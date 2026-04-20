"""
run_twisted_pf_lorenz63.py

Parallel particle smoother on the Lorenz-63 system — a chaotic nonlinear
SSM where no analytic filter exists.

Model
-----
  Lorenz-63 (Euler-discretised):
    x_t = f(x_{t-1}) + sigma_q * eps_t,    eps_t ~ N(0, I_3)
    y_t = x_t + sigma_r * eta_t,           eta_t ~ N(0, I_3)

  f(x) = x + dt * [sigma*(x2-x1), x1*(rho-x3)-x2, x1*x2-beta*x3]

Pipeline
--------
  1. Initialise twists from an EKF smoother (cheap, approximate).
  2. Optionally refine twists via gradient descent on the ELBO.
  3. Deploy the learned twists with the parallel DEER solver,
     warm-started from the twist means.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import schur as scipy_schur

from src.particle_filter import TwistedParallelParticleFilter

# Lorenz-63 parameters 
SIGMA_L = 10.0
RHO     = 28.0
BETA_L  = 8.0 / 3.0
DT      = 0.01

SIGMA_Q = 0.5     # process noise std (per dimension)
SIGMA_R = 2.0     # observation noise std (per dimension)

T  = 2000
D  = 3
N  = 512

def lorenz_deriv(x):
    return jnp.array([
        SIGMA_L * (x[1] - x[0]),
        x[0] * (RHO - x[2]) - x[1],
        x[0] * x[1] - BETA_L * x[2],
    ])

def lorenz_step(x):
    """Euler step."""
    return x + DT * lorenz_deriv(x)

def lorenz_jacobian(x):
    """Jacobian of the Euler-discretised Lorenz step."""
    J = jnp.array([
        [-SIGMA_L,  SIGMA_L,  0.0],
        [RHO - x[2], -1.0,   -x[0]],
        [x[1],        x[0],  -BETA_L],
    ])
    return jnp.eye(D) + DT * J


# EKF smoother (provides twist initialisation) 
def ekf_smoother(ys, x0, sigma_q, sigma_r):
    """Extended Kalman smoother for Lorenz-63."""
    _T = ys.shape[0]
    Q = sigma_q**2 * jnp.eye(D)
    R = sigma_r**2 * jnp.eye(D)

    def fwd_step(carry, y):
        m, P = carry
        m_pred = lorenz_step(m)
        J = lorenz_jacobian(m)
        P_pred = J @ P @ J.T + Q
        S = P_pred + R
        K = jnp.linalg.solve(S.T, P_pred.T).T
        m_new = m_pred + K @ (y - m_pred)
        P_new = (jnp.eye(D) - K) @ P_pred
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new)

    P0 = jnp.eye(D)
    _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(
        fwd_step, (x0, P0), ys
    )

    def bwd_step(carry, t):
        m_s, P_s = carry
        J = lorenz_jacobian(m_filts[t])
        G = jnp.linalg.solve(P_preds[t + 1].T, (P_filts[t] @ J.T).T).T
        m_s_new = m_filts[t] + G @ (m_s - m_preds[t + 1])
        P_s_new = P_filts[t] + G @ (P_s - P_preds[t + 1]) @ G.T
        return (m_s_new, P_s_new), (m_s_new, P_s_new)

    ts = jnp.arange(_T - 2, -1, -1)
    _, (ms_rev, Ps_rev) = jax.lax.scan(
        bwd_step, (m_filts[-1], P_filts[-1]), ts
    )

    smooth_means = jnp.concatenate([ms_rev[::-1], m_filts[-1:]])
    smooth_covs  = jnp.concatenate([Ps_rev[::-1], P_filts[-1:]])
    return smooth_means, smooth_covs


# Extended backward information filter 
def ebif(ys, ekf_means, sigma_q, sigma_r):
    """
    Extended backward information filter for Lorenz-63.

    Computes the optimal twist psi_t(x_t) = p(y_{t:T} | x_t) using a
    Gaussian (information-form) backward recursion, linearising the Lorenz
    transition around the EKF smoother means.

    Returns twist_means (T, D) and twist_vars (T, D) — diagonal of J^{-1}.
    """
    Q = sigma_q**2 * jnp.eye(D)
    R_inv = (1.0 / sigma_r**2) * jnp.eye(D)

    def step(carry, inputs):
        J, h = carry
        y_t, lin_pt = inputs

        # Linearise transition around lin_pt:
        #   x_{t+1} ≈ f(lin_pt) + F (x_t - lin_pt)  =>  A=F, b=f(lin_pt)-F*lin_pt
        F = lorenz_jacobian(lin_pt)
        b = lorenz_step(lin_pt) - F @ lin_pt

        # Backward prediction through transition
        J_inv = jnp.linalg.inv(J)
        P = J_inv + Q                          # J_{t+1}^{-1} + Q
        P_inv = jnp.linalg.inv(P)
        J_pred = F.T @ P_inv @ F
        h_pred = F.T @ P_inv @ (J_inv @ h - b)

        # Add observation: y = x + noise(R)
        J_new = J_pred + R_inv
        h_new = h_pred + R_inv @ y_t
        return (J_new, h_new), (J_new, h_new)

    # Initialise at t = T: beta_T(x_T) = p(y_T | x_T)
    J_T = R_inv
    h_T = R_inv @ ys[-1]

    # Backward scan: t = T-1, ..., 0
    # Linearisation points: EKF smoother means at t (not t+1)
    _, (Js_rev, hs_rev) = jax.lax.scan(
        step, (J_T, h_T),
        (ys[-2::-1], ekf_means[-2::-1]),
    )

    Js = jnp.concatenate([Js_rev[::-1], J_T[None]], axis=0)   # (T, D, D)
    hs = jnp.concatenate([hs_rev[::-1], h_T[None]], axis=0)   # (T, D)

    def extract(J, h):
        J_inv = jnp.linalg.inv(J)
        return J_inv @ h, jnp.diag(J_inv)

    twist_means, twist_vars = jax.vmap(extract)(Js, hs)
    return twist_means, twist_vars


# Twisted PF step and weight functions (full-covariance twist) 
# Twist parameterisation: psi_t(x_t) = N(x_t; mu_t, Sigma_t) with
# Sigma_t = L_t L_t^T,  L_t lower-triangular, diag(L_t) = exp(chol_log_diag).
# We parameterise L via (chol_log_diag, chol_offdiag) so Sigma stays SPD.
def build_chol(chol_log_diag, chol_offdiag):
    """Lower-triangular Cholesky from (log_diag, strict lower offdiag)."""
    return jnp.tril(chol_offdiag, k=-1) + jnp.diag(jnp.exp(chol_log_diag))


def _twist_posterior(f_prev, twist, sigma_q):
    """Gaussian posterior of q(x_t|x_{t-1}) ∝ N(x_t;f_prev,sigma_q^2 I)·psi_t(x_t)."""
    mu_twist = twist['mean']
    L_twist  = build_chol(twist['chol_log_diag'], twist['chol_offdiag'])

    prec_q         = 1.0 / sigma_q**2
    # Sigma_twist^{-1} via Cholesky solve
    Sigma_twist_inv = jax.scipy.linalg.cho_solve((L_twist, True), jnp.eye(D))
    prec_post      = prec_q * jnp.eye(D) + Sigma_twist_inv
    Sigma_post     = jnp.linalg.inv(prec_post)
    mean_post      = Sigma_post @ (prec_q * f_prev + Sigma_twist_inv @ mu_twist)
    L_post         = jnp.linalg.cholesky(Sigma_post)
    return mean_post, L_post


def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    sigma_q  = params['sigma_q']
    f_prev   = lorenz_step(x_prev)
    mean_post, L_post = _twist_posterior(f_prev, twist, sigma_q)
    return mean_post + L_post @ noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    sigma_q = params['sigma_q']
    sigma_r = params['sigma_r']

    f_prev = lorenz_step(x_prev)

    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sigma_r**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_r**2))
    log_trans = (-0.5 * jnp.sum((x_next - f_prev)**2) / sigma_q**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_q**2))

    mean_post, L_post = _twist_posterior(f_prev, twist, sigma_q)
    diff = x_next - mean_post
    z    = jax.scipy.linalg.solve_triangular(L_post, diff, lower=True)
    log_prop = (-0.5 * jnp.sum(z**2)
                - jnp.sum(jnp.log(jnp.diag(L_post)))
                - 0.5 * D * jnp.log(2 * jnp.pi))
    return log_lik + log_trans - log_prop


def weighted_mean(states, log_weights):
    """Importance-weighted smoothing mean using total path weights."""
    total_log_w = jnp.sum(log_weights, axis=1)          # (N,)
    w = jax.nn.softmax(total_log_w)                      # (N,)
    return jnp.einsum('n,ntd->td', w, states)            # (T, D)



# Baselines: Bootstrap PF (filter) and Bootstrap PS (FFBSi smoother)
def bootstrap_pf_lorenz(ys, x0_init, sigma_q, sigma_r, key, N):
    """
    Standard bootstrap particle filter for Lorenz-63.

    Stores pre-resampling particles and normalised log-weights at every step
    so they can be used by a backward-simulation smoother (FFBSi).

    Returns
    -------
    filt_means : (T, D)   importance-weighted filter means
    all_parts  : (T, N, D) pre-resampling particles at each timestep
    all_log_w  : (T, N)    normalised pre-resampling log-weights
    ess        : (T,)      effective sample size per step
    log_ml     : scalar    bootstrap log marginal likelihood estimate
    """
    def _systematic_resample(key, log_w, particles):
        w = jax.nn.softmax(log_w)
        n = w.shape[0]
        u = (jnp.arange(n) + jr.uniform(key)) / n
        cumw = jnp.cumsum(w)
        idx = jnp.clip(jnp.searchsorted(cumw, u), 0, n - 1)
        return particles[idx]

    def step(carry, yt):
        particles, key, log_ml_acc = carry
        key, trans_key, resamp_key = jr.split(key, 3)

        # Propagate through Lorenz dynamics
        particles_new = jax.vmap(lorenz_step)(particles) + sigma_q * jr.normal(
            trans_key, particles.shape)

        # Weight by observation likelihood
        log_w = -0.5 * jnp.sum((yt - particles_new)**2, axis=-1) / sigma_r**2
        log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)

        # Filter mean and ESS
        w_soft = jnp.exp(log_w_norm)
        filt_mean = jnp.sum(w_soft[:, None] * particles_new, axis=0)
        ess = 1.0 / jnp.sum(w_soft**2)

        # Marginal likelihood increment
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))

        # Resample for next step
        particles_resamp = _systematic_resample(resamp_key, log_w, particles_new)

        return ((particles_resamp, key, log_ml_acc + log_ml_inc),
                (particles_new, log_w_norm, filt_mean, ess))

    key, init_key = jr.split(key)
    particles0 = x0_init[None, :] + 0.5 * jr.normal(init_key, (N, D))
    (_, _, log_ml), (all_parts, all_log_w, filt_means, ess) = jax.lax.scan(
        step, (particles0, key, 0.0), ys)
    return filt_means, all_parts, all_log_w, ess, log_ml


def ffbsi_smoother(all_parts, all_log_w, sigma_q, key, M):
    """
    Forward Filtering Backward Simulation — particle smoother baseline.

    For each of M backward trajectories:
      1. Sample x_T from filter weights at time T.
      2. Walk backward sampling ancestor from weights w^(f)_i * p(x_{t+1}|x_i).

    Returns (M, T, D) array of smoothed trajectories.
    """
    T_steps, N, D_ = all_parts.shape
    sq2 = sigma_q**2

    def backward_one(m_key):
        m_key, init_key = jr.split(m_key)
        idx_T = jr.categorical(init_key, all_log_w[-1])
        x_T = all_parts[-1, idx_T]

        def bwd_step(carry, inputs):
            x_next, key = carry
            parts_t, log_w_t, step_key = inputs
            # Predicted particles: f(x_i) = lorenz_step(x_i)
            preds = jax.vmap(lorenz_step)(parts_t)                  # (N, D)
            log_trans = -0.5 * jnp.sum((x_next - preds)**2, axis=-1) / sq2
            log_smooth = log_w_t + log_trans
            idx = jr.categorical(step_key, log_smooth)
            x_t = parts_t[idx]
            key, _ = jr.split(key)
            return (x_t, key), x_t

        step_keys = jr.split(m_key, T_steps - 1)
        # Backward scan over t = T-2, T-3, ..., 0
        (_, _), path_rev = jax.lax.scan(
            bwd_step,
            (x_T, m_key),
            (all_parts[:-1][::-1], all_log_w[:-1][::-1], step_keys),
        )
        # path_rev is in reversed order; concat with x_T at the end
        path = jnp.concatenate([path_rev[::-1], x_T[None]], axis=0)
        return path                                                 

    keys = jr.split(key, M)
    return jax.vmap(backward_one)(keys)                               



# Simulate data
key = jr.PRNGKey(0)
key, k1, k2, k3 = jr.split(key, 4)

x0_true = jnp.array([1.0, 1.0, 1.0])
def burn_step(x, _):
    return lorenz_step(x), None
x0_true, _ = jax.lax.scan(burn_step, x0_true, None, length=5000)

def sim_step(x, noise_obs):
    q, r = noise_obs
    xn = lorenz_step(x) + SIGMA_Q * q
    y  = xn + SIGMA_R * r
    return xn, (xn, y)

_, (true_xs, ys) = jax.lax.scan(
    sim_step, x0_true,
    (jr.normal(k2, (T, D)), jr.normal(k3, (T, D)))
)

# 
# EKF smoother → linearisation points, then backward info filter
# 
print("Running EKF smoother ...")
ekf_means, ekf_covs = jax.jit(ekf_smoother)(ys, x0_true, SIGMA_Q, SIGMA_R)
ekf_scales = jnp.sqrt(jax.vmap(jnp.diag)(ekf_covs))          # (T, D)

ekf_rmse = float(jnp.sqrt(jnp.mean((ekf_means - true_xs)**2)))
print(f"  EKF smoother RMSE: {ekf_rmse:.4f}")

print("Running extended backward information filter ...")
ebif_means, ebif_vars = jax.jit(ebif)(ys, ekf_means, SIGMA_Q, SIGMA_R)
ebif_rmse = float(jnp.sqrt(jnp.mean((ebif_means - true_xs)**2)))
print(f"  EBIF twist RMSE:   {ebif_rmse:.4f}")

#  Schur basis: approximately diagonalise the average step Jacobian 
# The DEER solver uses a diagonal Jacobian approximation.  Rotating into
# coordinates where the average Jacobian is upper-triangular (Schur form)
# concentrates most coupling above the diagonal, making the diagonal
# Hutchinson estimator far more accurate for coupled dynamics like Lorenz.
avg_jac = np.array(jnp.mean(jax.vmap(lorenz_jacobian)(ekf_means), axis=0))
_, Q_schur = scipy_schur(avg_jac, output='real')
R_basis     = jnp.array(Q_schur.T)       # rotation INTO Schur coords
R_basis_inv = jnp.array(Q_schur)          # rotation BACK to original
schur_basis = (R_basis, R_basis_inv)
print(f"  Schur basis computed (avg Jacobian diag: {np.diag(Q_schur.T @ avg_jac @ Q_schur).round(4)})")


# Baseline: Bootstrap PF (filter) and Bootstrap PS (FFBSi smoother)
print("\n--- Baselines: Bootstrap PF / FFBSi smoother ---")
key, bpf_key, ffbsi_key = jr.split(key, 3)
run_bpf = jax.jit(bootstrap_pf_lorenz, static_argnames=("N",))
# Warmup
_ = run_bpf(ys, x0_true, SIGMA_Q, SIGMA_R, bpf_key, N)
jax.block_until_ready(_[0])

t0 = time.perf_counter()
bpf_filt_means, bpf_parts, bpf_log_w, bpf_ess, bpf_log_ml = run_bpf(
    ys, x0_true, SIGMA_Q, SIGMA_R, bpf_key, N)
jax.block_until_ready(bpf_filt_means)
t_bpf = time.perf_counter() - t0

rmse_bpf_filt = float(jnp.sqrt(jnp.mean((bpf_filt_means - true_xs)**2)))
print(f"  Bootstrap PF (filter):    time={t_bpf:.4f}s  RMSE={rmse_bpf_filt:.4f}  "
      f"ESS={float(bpf_ess.mean()):.0f}/{N}")

# FFBSi smoother with M = N backward trajectories
M_BACKWARD = N
run_ffbsi = jax.jit(ffbsi_smoother, static_argnames=("M",))
_ = run_ffbsi(bpf_parts, bpf_log_w, SIGMA_Q, ffbsi_key, M=M_BACKWARD)
jax.block_until_ready(_)

t0 = time.perf_counter()
bps_trajs = run_ffbsi(bpf_parts, bpf_log_w, SIGMA_Q, ffbsi_key, M=M_BACKWARD)
bps_trajs.block_until_ready()
t_bps = time.perf_counter() - t0
bps_means = bps_trajs.mean(axis=0)                                    # (T, D)
rmse_bps = float(jnp.sqrt(jnp.mean((bps_means - true_xs)**2)))
print(f"  Bootstrap PS (FFBSi):     time={t_bps:.4f}s  RMSE={rmse_bps:.4f}  "
      f"(M={M_BACKWARD} paths)")

# Build particle filter
tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)
model_params = {'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R}

key, s0_key, noise_key = jr.split(key, 3)
x0s   = x0_true[None, :] + 0.5 * jr.normal(s0_key, (N, D))
noise = jr.normal(noise_key, (N, T, D))


# A) Full-covariance twists initialised from EKF smoother
# Use the EKF smoother posterior Gaussians directly as full-cov twists.
# This captures the strong (x,y) coupling in Lorenz dynamics that a
# diagonal twist fundamentally cannot represent.
L_ekf = jax.vmap(jnp.linalg.cholesky)(ekf_covs + 1e-6 * jnp.eye(D))  # (T, D, D)
chol_log_diag_init = jnp.log(jax.vmap(jnp.diag)(L_ekf))              # (T, D)
chol_offdiag_init  = jax.vmap(lambda M: jnp.tril(M, k=-1))(L_ekf)    # (T, D, D)

twist_ekf = {
    'mean':           ekf_means,
    'chol_log_diag':  chol_log_diag_init,
    'chol_offdiag':   chol_offdiag_init,
}
tape_ekf = (ys, noise, twist_ekf)

print(f"\n--- EKF twist + Schur basis (T={T}, D={D}, N={N}) ---")

# Sequential
run_seq = jax.jit(tpf.run_sequential)
run_seq(x0s, tape_ekf, model_params).block_until_ready()
t0 = time.perf_counter()
states_ekf_seq = run_seq(x0s, tape_ekf, model_params).block_until_ready()
t_ekf_seq = time.perf_counter() - t0

log_ws_ekf_seq = jax.jit(tpf.compute_log_weights)(
    x0s, states_ekf_seq, tape_ekf, model_params)
mean_ekf_seq   = weighted_mean(states_ekf_seq, log_ws_ekf_seq)
rmse_ekf_seq   = float(jnp.sqrt(jnp.mean((mean_ekf_seq - true_xs)**2)))

print(f"  Sequential:  time={t_ekf_seq:.4f}s  RMSE={rmse_ekf_seq:.4f}")

# Parallel (warm-started from EKF means)
key, dk = jr.split(key)
params_par = {**model_params, 'key': dk}
yinit_ekf = ekf_means   

run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
s, _ = run_par(x0s, tape_ekf, params_par, max_iter=100, num_samples=5,
               yinit_guess=yinit_ekf, basis=schur_basis)
s.block_until_ready()
t0 = time.perf_counter()
states_ekf_par, iters_ekf = run_par(
    x0s, tape_ekf, params_par, max_iter=100, num_samples=5,
    yinit_guess=yinit_ekf, basis=schur_basis)
states_ekf_par.block_until_ready()
t_ekf_par = time.perf_counter() - t0
deer_iters_ekf = int(iters_ekf[0])

log_ws_ekf_par = jax.jit(tpf.compute_log_weights)(
    x0s, states_ekf_par, tape_ekf, model_params)
mean_ekf_par   = weighted_mean(states_ekf_par, log_ws_ekf_par)
rmse_ekf_par   = float(jnp.sqrt(jnp.mean((mean_ekf_par - true_xs)**2)))

print(f"  Parallel:    time={t_ekf_par:.4f}s  RMSE={rmse_ekf_par:.4f}"
      f"  ({deer_iters_ekf} DEER iters, {t_ekf_seq/t_ekf_par:.1f}x speedup)")


# B) Learned full-covariance twists (EKF init + gradient refinement)
twist_mean     = ekf_means.copy()               
twist_log_diag = chol_log_diag_init              
twist_offdiag  = chol_offdiag_init              

def loss_fn(twist_mean, twist_log_diag, twist_offdiag):
    twist_params = {
        'mean':          twist_mean,
        'chol_log_diag': twist_log_diag,
        'chol_offdiag':  twist_offdiag,
    }
    tape   = (ys, noise, twist_params)
    states = tpf.run_sequential(x0s, tape, model_params)
    log_ws = tpf.compute_log_weights(x0s, states, tape, model_params)
    total  = jnp.sum(log_ws, axis=1)
    elbo   = jax.scipy.special.logsumexp(total) - jnp.log(float(N))
    return -elbo

grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))

def adam_init_tree(*params):
    return tuple((jnp.zeros_like(p), jnp.zeros_like(p)) for p in params) + (0,)

def adam_step_tree(grads, state, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
    *mv_pairs, t = state
    t = t + 1
    updates, new_pairs = [], []
    for g, (m, v) in zip(grads, mv_pairs):
        m_new = b1 * m + (1 - b1) * g
        v_new = b2 * v + (1 - b2) * g**2
        m_hat = m_new / (1 - b1**t)
        v_hat = v_new / (1 - b2**t)
        updates.append(-lr * m_hat / (jnp.sqrt(v_hat) + eps))
        new_pairs.append((m_new, v_new))
    return tuple(updates), tuple(new_pairs) + (t,)

N_STEPS   = 1500
LR_MAX    = 1e-2
LR_MIN    = 1e-4

import os, pickle
CACHE_PATH = "examples/_lorenz_learned_twist.pkl"

if os.path.exists(CACHE_PATH):
    print(f"\n--- Loading cached learned twist from {CACHE_PATH} ---")
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    twist_mean     = jnp.array(cached['mean'])
    twist_log_diag = jnp.array(cached['log_diag'])
    twist_offdiag  = jnp.array(cached['offdiag'])
    losses         = cached['losses']
    t_learn        = cached['t_learn']
    print(f"  cache loaded  (final neg-ELBO = {losses[-1]:.2f}, learning took {t_learn:.1f}s)")
else:
    opt_state = adam_init_tree(twist_mean, twist_log_diag, twist_offdiag)
    losses    = []

    print(f"\n--- Learning full-cov twist parameters ({N_STEPS} steps) ---")
    t_learn_start = time.perf_counter()

    for i in range(N_STEPS):
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + jnp.cos(jnp.pi * i / N_STEPS))
        loss, grads = grad_fn(twist_mean, twist_log_diag, twist_offdiag)
        updates, opt_state = adam_step_tree(grads, opt_state, lr=lr)
        twist_mean     = twist_mean     + updates[0]
        twist_log_diag = twist_log_diag + updates[1]
        twist_offdiag  = twist_offdiag  + updates[2]
        losses.append(float(loss))
        if (i + 1) % 100 == 0:
            print(f"  step {i+1:>4}/{N_STEPS}  neg-ELBO = {loss:.2f}")

    t_learn = time.perf_counter() - t_learn_start
    print(f"Learning complete in {t_learn:.1f}s")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            'mean':     np.array(twist_mean),
            'log_diag': np.array(twist_log_diag),
            'offdiag':  np.array(twist_offdiag),
            'losses':   losses,
            't_learn':  t_learn,
        }, f)
    print(f"  cached learned twist to {CACHE_PATH}")

twist_learned = {
    'mean':          twist_mean,
    'chol_log_diag': twist_log_diag,
    'chol_offdiag':  twist_offdiag,
}
tape_learned  = (ys, noise, twist_learned)

# Sequential with learned twists
run_seq(x0s, tape_learned, model_params).block_until_ready()
t0 = time.perf_counter()
states_learn_seq = run_seq(x0s, tape_learned, model_params).block_until_ready()
t_learn_seq = time.perf_counter() - t0

log_ws_learn_seq = jax.jit(tpf.compute_log_weights)(
    x0s, states_learn_seq, tape_learned, model_params)
mean_learn_seq   = weighted_mean(states_learn_seq, log_ws_learn_seq)
rmse_learn_seq   = float(jnp.sqrt(jnp.mean((mean_learn_seq - true_xs)**2)))

print(f"  Sequential:  time={t_learn_seq:.4f}s  RMSE={rmse_learn_seq:.4f}")

# Parallel with learned twists.
# The learned chol_offdiag injects cross-coord coupling into the effective
# Jacobian (Sigma_post · prec_q · J_f), whose spectrum has complex
# conjugate eigenvalues (≈ 0.91 ± 0.013j).  A real Schur basis cannot
# diagonalise complex pairs, so the basis HURTS here — we drop it and
# instead damp the Jacobian slightly (damp=0.95) to pull the effective
# spectral radius below 1.  This cuts iters from ~58 to ~29.
key, dk = jr.split(key)
params_par = {**model_params, 'key': dk}
yinit_learned = twist_mean   # (T, D)

s, _ = run_par(x0s, tape_learned, params_par, max_iter=100, num_samples=5,
               yinit_guess=yinit_learned, damp_factor=0.95)
s.block_until_ready()
t0 = time.perf_counter()
states_learn_par, iters_learn = run_par(
    x0s, tape_learned, params_par, max_iter=100, num_samples=5,
    yinit_guess=yinit_learned, damp_factor=0.95)
states_learn_par.block_until_ready()
t_learn_par = time.perf_counter() - t0
deer_iters_learn = int(iters_learn[0])

log_ws_learn_par = jax.jit(tpf.compute_log_weights)(
    x0s, states_learn_par, tape_learned, model_params)
mean_learn_par   = weighted_mean(states_learn_par, log_ws_learn_par)
rmse_learn_par   = float(jnp.sqrt(jnp.mean((mean_learn_par - true_xs)**2)))

print(f"  Parallel:    time={t_learn_par:.4f}s  RMSE={rmse_learn_par:.4f}"
      f"  ({deer_iters_learn} DEER iters, {t_learn_seq/t_learn_par:.1f}x speedup)")

# Per-step ESS 
log_w_norm   = log_ws_ekf_par - jax.scipy.special.logsumexp(
    log_ws_ekf_par, axis=0, keepdims=True)
ess_per_step = 1.0 / jnp.sum(jnp.exp(2 * log_w_norm), axis=0)
mean_ess     = float(jnp.mean(ess_per_step))


# Summary
print(f"\n{'='*70}")
print(f"Summary — Lorenz-63, T={T}, D={D}, N={N}")
print(f"{'='*70}")
print(f"{'Method':<28} {'Time':>8} {'RMSE':>8} {'DEER':>6} {'Speedup':>8}")
print(f"{'-'*70}")
print(f"{'EKF smoother (reference)':<28} {'—':>8} {ekf_rmse:>8.4f} {'—':>6} {'—':>8}")
print(f"{'Bootstrap PF (filter)':<28} {t_bpf:>7.4f}s {rmse_bpf_filt:>8.4f} {'—':>6} {'—':>8}")
print(f"{'Bootstrap PS (FFBSi)':<28} {t_bps:>7.4f}s {rmse_bps:>8.4f} {'—':>6} {'—':>8}")
print(f"{'EKF twist (seq)':<28} {t_ekf_seq:>7.4f}s {rmse_ekf_seq:>8.4f} {'—':>6} {'1.0x':>8}")
print(f"{'EKF twist (par, warm)':<28} {t_ekf_par:>7.4f}s {rmse_ekf_par:>8.4f}"
      f" {deer_iters_ekf:>5}  {t_ekf_seq/t_ekf_par:>7.1f}x")
print(f"{'Learned twist (seq)':<28} {t_learn_seq:>7.4f}s {rmse_learn_seq:>8.4f} {'—':>6} {'1.0x':>8}")
print(f"{'Learned twist (par, warm)':<28} {t_learn_par:>7.4f}s {rmse_learn_par:>8.4f}"
      f" {deer_iters_learn:>5}  {t_learn_seq/t_learn_par:>7.1f}x")
print(f"{'-'*70}")
print(f"Per-step ESS (mean): {mean_ess:.1f} / {N}")
print(f"{'='*70}")


# Use the best parallel result for plotting
best_par_means = mean_ekf_par
best_speedup   = t_ekf_seq / t_ekf_par
best_iters     = deer_iters_ekf

fig = plt.figure(figsize=(16, 10))
t_idx = jnp.arange(T)
dim_names = ['x', 'y', 'z']

# Row 1: three state dimensions
for d in range(3):
    ax = fig.add_subplot(3, 3, d + 1)
    ax.plot(t_idx, true_xs[:, d], 'k-', lw=1, alpha=0.7, label='true')
    ax.plot(t_idx, ekf_means[:, d], 'g-', lw=0.8, alpha=0.6, label='EKF')
    ax.plot(t_idx, best_par_means[:, d], 'r-', lw=0.8, alpha=0.8,
            label='parallel PF')
    ax.set_ylabel(f'${dim_names[d]}$')
    if d == 0:
        ax.legend(fontsize=8)
    ax.set_title(f'State ${dim_names[d]}$')
    ax.grid(True, alpha=0.3)

# Row 2: loss curve, per-step ESS, 3D attractor
ax = fig.add_subplot(3, 3, 4)
ax.plot(losses, 'b-', lw=1)
ax.set_xlabel('Step')
ax.set_ylabel('Neg-ELBO')
ax.set_title('Twist learning')
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 5)
ax.plot(t_idx, ess_per_step, 'b-', lw=0.8)
ax.axhline(N, color='k', ls='--', lw=0.5, label=f'N={N}')
ax.set_xlabel('Timestep')
ax.set_ylabel('ESS')
ax.set_title(f'Per-step ESS (mean={mean_ess:.0f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 6, projection='3d')
ax.plot(true_xs[:, 0], true_xs[:, 1], true_xs[:, 2],
        'k-', lw=0.5, alpha=0.5, label='true')
ax.plot(best_par_means[:, 0], best_par_means[:, 1], best_par_means[:, 2],
        'r-', lw=0.5, alpha=0.7, label='parallel PF')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Lorenz attractor')
ax.legend(fontsize=8)

# Row 3: twist means vs true for each dimension
for d in range(3):
    ax = fig.add_subplot(3, 3, 7 + d)
    ax.plot(t_idx, true_xs[:, d], 'k-', lw=1, alpha=0.7, label='true')
    ax.plot(t_idx, ekf_means[:, d], 'g-', lw=0.8, alpha=0.5, label='EKF mean')
    ax.plot(t_idx, twist_mean[:, d], 'r-', lw=0.8, alpha=0.7, label='learned twist')
    scale = jnp.exp(twist_log_diag[:, d])
    ax.fill_between(t_idx, twist_mean[:, d] - scale,
                    twist_mean[:, d] + scale, color='r', alpha=0.1)
    ax.set_xlabel('Timestep')
    ax.set_title(f'Learned twist ${dim_names[d]}$')
    if d == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Lorenz-63 parallel particle smoother — T={T}, D={D}, N={N}, '
    f'{best_iters} DEER iters, {best_speedup:.0f}x speedup',
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig('twisted_pf_lorenz63.png', dpi=150, bbox_inches='tight')
plt.show()
