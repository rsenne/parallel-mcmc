"""
run_twisted_pf_stochvol.py

Stochastic volatility model — a nonlinear SSM where the Kalman filter
cannot be applied.  Demonstrates:

  1. Learning twist (backward) parameters via gradient descent through
     the differentiable sequential particle smoother.
  2. Deploying the learned twists with the parallel DEER solver for
     fast inference (O(log T) depth).

Model
-----
  x_t = phi * x_{t-1} + sigma_q * eps_t          (log-volatility)
  y_t = beta * exp(x_t / 2) * eta_t              (observations)

The twisted proposal modifies the transition prior with a per-timestep
Gaussian "twist" that approximates the backward smoothing message.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter

PHI     = 0.95
SIGMA_Q = 0.2
BETA    = 1.0

T  = 1000
D  = 1
N  = 256

def twisted_step_fn(x_prev, driver, params):
    """Twisted transition (same form for any model with linear-Gaussian transition)."""
    obs, noise, twist = driver
    phi     = params['phi']
    sigma_q = params['sigma_q']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    return mean_post + jnp.sqrt(var_post) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    """Log importance weight — SV observation model."""
    phi     = params['phi']
    sigma_q = params['sigma_q']
    beta    = params['beta']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])

    # log p(y_t | x_t) — stochastic volatility likelihood
    log_lik = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(beta)
               - x_next[0] / 2.0
               - obs[0]**2 / (2.0 * beta**2 * jnp.exp(x_next[0])))

    # log p(x_t | x_{t-1})
    log_trans = (-0.5 * (x_next[0] - phi * x_prev[0])**2 / sigma_q**2
                 - 0.5 * jnp.log(2.0 * jnp.pi * sigma_q**2))

    # log q_twist(x_t | x_{t-1})
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    log_prop = (-0.5 * (x_next[0] - mean_post[0])**2 / var_post[0]
                - 0.5 * jnp.log(2.0 * jnp.pi * var_post[0]))

    return log_lik + log_trans - log_prop

def approx_backward_info_filter_sv(ys, phi, sigma_q, beta, n_iter=3):
    """
    Approximate backward information filter for the stochastic volatility model.

    Iteratively linearises the observation log-likelihood
        log p(y|x) = -0.5*log(2pi) - log(beta) - x/2 - y^2/(2*beta^2*exp(x))
    around the current backward message mean.

    Returns twist_means (T,) and twist_vars (T,).
    """
    T = ys.shape[0]
    sq2 = sigma_q**2

    def obs_info(y, x0):
        """Gaussian approx of observation info at linearisation point x0."""
        # Clamp x0 to avoid overflow in exp(-x0) for large negative x0
        x0_c = jnp.clip(x0, -20.0, 20.0)
        r = y**2 / (2.0 * beta**2 * jnp.exp(x0_c))
        r = jnp.clip(r, 1e-8, 100.0)             # clip once for consistency
        J_obs = r                                  # -d^2/dx^2 log p(y|x)
        g = -0.5 + r                              # d/dx log p(y|x) at x0
        h_obs = g + J_obs * x0_c
        return J_obs, h_obs

    # Initialise twist means with stationary mean (zero)
    twist_means = jnp.zeros(T)

    for _ in range(n_iter):
        def step(carry, inputs):
            J, h = carry
            y_t, lin_pt = inputs
            # Predict backward through linear transition x_{t+1} = phi*x_t + noise
            denom = 1.0 + sq2 * J
            J_pred = phi**2 * J / denom
            h_pred = phi * h / denom
            # Add linearised observation
            J_obs, h_obs = obs_info(y_t, lin_pt)
            J_obs = jnp.maximum(J_obs, 1e-6)   # ensure positive
            return (J_pred + J_obs, h_pred + h_obs), (J_pred + J_obs, h_pred + h_obs)

        # Last timestep
        J_obs_T, h_obs_T = obs_info(ys[-1], twist_means[-1])
        J_obs_T = jnp.maximum(J_obs_T, 1e-6)
        J_T = J_obs_T
        h_T = h_obs_T

        _, (Js_rev, hs_rev) = jax.lax.scan(
            step, (J_T, h_T),
            (ys[-2::-1], twist_means[-2::-1]),
        )

        Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_T])])
        hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_T])])
        twist_means = hs / Js

    twist_vars = 1.0 / Js
    return twist_means, twist_vars


key = jr.PRNGKey(0)
key, k1, k2, k3 = jr.split(key, 4)
x0_true = jr.normal(k1) * 0.5

def sim_sv_step(x, noise_obs):
    q, r = noise_obs
    xn = PHI * x + SIGMA_Q * q
    y  = BETA * jnp.exp(xn / 2.0) * r
    return xn, (xn, y)

_, (true_xs, ys) = jax.lax.scan(
    sim_sv_step, x0_true, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
)
ys_2d = ys[:, None]

tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)
params = {'phi': PHI, 'sigma_q': SIGMA_Q, 'beta': BETA}

# Fixed particles and noise
key, s0_key, noise_key = jr.split(key, 3)
x0s   = jr.normal(s0_key, (N, D)) * 0.5
noise = jr.normal(noise_key, (N, T, D))

# Initialise twists from approximate backward info filter
print("Running approximate backward information filter ...")
bif_means, bif_vars = jax.jit(approx_backward_info_filter_sv, static_argnames=('n_iter',))(ys, PHI, SIGMA_Q, BETA, n_iter=10)
bif_rmse = float(jnp.sqrt(jnp.mean((bif_means - true_xs)**2)))
print(f"  BIF twist RMSE: {bif_rmse:.4f}")

twist_mean      = bif_means[:, None]              # (T, 1)
twist_log_scale = 0.5 * jnp.log(bif_vars)[:, None]  # (T, 1)


def loss_fn(twist_mean, twist_log_scale):
    twist_params = {'mean': twist_mean, 'log_scale': twist_log_scale}
    tape   = (ys_2d, noise, twist_params)
    states = tpf.run_sequential(x0s, tape, params)                 # (N, T, D)
    log_ws = tpf.compute_log_weights(x0s, states, tape, params)    # (N, T)
    total  = jnp.sum(log_ws, axis=1)                               # (N,)
    elbo   = jax.scipy.special.logsumexp(total) - jnp.log(float(N))
    return -elbo

grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

def adam_init(p1, p2):
    return (jnp.zeros_like(p1), jnp.zeros_like(p1),
            jnp.zeros_like(p2), jnp.zeros_like(p2), 0)

def adam_step(g1, g2, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m1, v1, m2, v2, t = state
    t  = t + 1
    m1 = b1 * m1 + (1 - b1) * g1
    v1 = b2 * v1 + (1 - b2) * g1**2
    m2 = b1 * m2 + (1 - b1) * g2
    v2 = b2 * v2 + (1 - b2) * g2**2
    u1 = -lr * (m1 / (1 - b1**t)) / (jnp.sqrt(v1 / (1 - b2**t)) + eps)
    u2 = -lr * (m2 / (1 - b2**t)) / (jnp.sqrt(v2 / (1 - b2**t)) + eps)
    return u1, u2, (m1, v1, m2, v2, t)

N_STEPS   = 500
opt_state = adam_init(twist_mean, twist_log_scale)
losses    = []

print(f"Learning twist parameters (T={T}, N={N}) ...")
t_learn_start = time.perf_counter()

for i in range(N_STEPS):
    loss, (g_mean, g_scale) = grad_fn(twist_mean, twist_log_scale)
    u_mean, u_scale, opt_state = adam_step(g_mean, g_scale, opt_state, lr=1e-2)
    twist_mean      = twist_mean + u_mean
    twist_log_scale = twist_log_scale + u_scale
    losses.append(float(loss))
    if (i + 1) % 25 == 0:
        print(f"  step {i+1:>4}/{N_STEPS}  neg-ELBO = {loss:.2f}")

t_learn = time.perf_counter() - t_learn_start
print(f"Learning complete in {t_learn:.1f}s\n")

twist_learned = {'mean': twist_mean, 'log_scale': twist_log_scale}
tape_learned  = (ys_2d, noise, twist_learned)

# Sequential
run_seq = jax.jit(tpf.run_sequential)
run_seq(x0s, tape_learned, params).block_until_ready()

t0 = time.perf_counter()
states_seq = run_seq(x0s, tape_learned, params).block_until_ready()
t_seq = time.perf_counter() - t0

# Parallel
key, dk = jr.split(key)
params['key'] = dk
run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
s, i_ = run_par(x0s, tape_learned, params, max_iter=100)
s.block_until_ready()

t0 = time.perf_counter()
states_par, iters = run_par(x0s, tape_learned, params, max_iter=100)
states_par.block_until_ready()
t_par = time.perf_counter() - t0
deer_iters = int(iters[0])

def _path_log_weights(states, tape, pars):
    """Cumulative path log-weights for each particle: shape (N,)."""
    log_ws = tpf.compute_log_weights(x0s, states, tape, pars)   # (N, T)
    return jnp.sum(log_ws, axis=1)                               # (N,)


def _weighted_mean_from_total(states, total_log_w):
    """Importance-weighted smoother mean given pre-computed total log-weights."""
    w = jax.nn.softmax(total_log_w)
    return jnp.einsum('n,ntd->td', w, states)[:, 0]


def _systematic_resample(key, log_w):
    """Systematic-resampling indices.  log_w: (N,) → idx: (N,)."""
    w = jax.nn.softmax(log_w)
    n = w.shape[0]
    u = (jnp.arange(n) + jr.uniform(key)) / n
    cumw = jnp.cumsum(w)
    return jnp.clip(jnp.searchsorted(cumw, u), 0, n - 1)


def _resampled_mean(states, total_log_w, key):
    """Post-hoc residual-resampling smoother mean: resample once on path
    weights, then take a *uniform* mean over the resampled trajectories.
    DEER convergence is unaffected — the resample is post-DEER and does not
    feed back into the proposal."""
    idx = _systematic_resample(key, total_log_w)   # (N,)
    return states[idx].mean(axis=0)[:, 0]            # (T,)


# Strip the 'key' from params for the deterministic weight evaluation
_pars_w = {k: v for k, v in params.items() if k != 'key'}

# Path log-weights for each set of trajectories
log_w_seq  = _path_log_weights(states_seq, tape_learned, _pars_w)
log_w_par  = _path_log_weights(states_par, tape_learned, _pars_w)

seq_means = _weighted_mean_from_total(states_seq, log_w_seq)
par_means = _weighted_mean_from_total(states_par, log_w_par)

rmse_seq = float(jnp.sqrt(jnp.mean((seq_means - true_xs)**2)))
rmse_par = float(jnp.sqrt(jnp.mean((par_means - true_xs)**2)))

# Post-hoc systematic resampling on path weights — single pass, post-DEER.
key, rs_key = jr.split(key)
par_means_rs = _resampled_mean(states_par, log_w_par, rs_key)
rmse_par_rs  = float(jnp.sqrt(jnp.mean((par_means_rs - true_xs)**2)))

# Effective sample size of the path weights (same for seq/par by bit-equality)
ess_par = float(1.0 / jnp.sum(jax.nn.softmax(log_w_par) ** 2))


# Also run with NO twist (prior proposal) for reference
twist_none = {'mean': jnp.zeros((T, D)), 'log_scale': 3.0 * jnp.ones((T, D))}
tape_none  = (ys_2d, noise, twist_none)
states_none = run_seq(x0s, tape_none, _pars_w).block_until_ready()
log_w_none  = _path_log_weights(states_none, tape_none, _pars_w)
none_means  = _weighted_mean_from_total(states_none, log_w_none)
rmse_none   = float(jnp.sqrt(jnp.mean((none_means - true_xs)**2)))

print(f"T={T}, N={N}")
print(f"No twist (prior):              RMSE vs true = {rmse_none:.4f}")
print(f"Learned twist (seq):           time={t_seq:.4f}s   RMSE = {rmse_seq:.4f}")
print(f"Learned twist (par):           time={t_par:.4f}s   RMSE = {rmse_par:.4f}"
      f"  ({deer_iters} DEER iters)")
print(f"Learned twist (par + resamp):  time≈{t_par:.4f}s   RMSE = {rmse_par_rs:.4f}"
      f"  (path ESS = {ess_par:.1f}/{N})")
print(f"Speedup: {t_seq / t_par:.1f}x")

# Baselines: Bootstrap PF + FFBSi smoother
from src.baselines import bootstrap_pf, ffbsi_smoother

def sv_transition_mean(x_prev, p):
    return p['phi'] * x_prev

def sv_log_lik(y_t, x_t, p):
    # y | x ~ N(0, beta^2 * exp(x))
    var = p['beta']**2 * jnp.exp(x_t)
    return -0.5 * jnp.sum(y_t**2 / var + jnp.log(2 * jnp.pi * var))

key, bpf_key, ffbsi_key, init_key = jr.split(key, 4)
x0_particles = jr.normal(init_key, (N, D)) * 0.5
bpf_params = {'phi': PHI, 'sigma_q': SIGMA_Q, 'beta': BETA}

run_bpf = jax.jit(bootstrap_pf,
                  static_argnames=("transition_mean", "log_lik"))
_ = run_bpf(ys_2d, x0_particles, sv_transition_mean, sv_log_lik,
            SIGMA_Q, bpf_params, bpf_key)

t0 = time.perf_counter()
bpf_filt, bpf_parts, bpf_log_w, bpf_ess, bpf_log_ml = run_bpf(
    ys_2d, x0_particles, sv_transition_mean, sv_log_lik,
    SIGMA_Q, bpf_params, bpf_key)
bpf_filt.block_until_ready()
t_bpf = time.perf_counter() - t0
rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_filt[:, 0] - true_xs)**2)))
print(f"\nBootstrap PF (filter):  time={t_bpf:.3f}s   RMSE vs true = {rmse_bpf:.4f}"
      f"  ESS mean={float(bpf_ess.mean()):.0f}/{N}")

M_BACKWARD = N
run_ffbsi = jax.jit(ffbsi_smoother,
                    static_argnames=("M", "transition_mean"))
_ = run_ffbsi(bpf_parts, bpf_log_w, sv_transition_mean,
              SIGMA_Q, bpf_params, ffbsi_key, M=M_BACKWARD)

t0 = time.perf_counter()
bps_trajs = run_ffbsi(bpf_parts, bpf_log_w, sv_transition_mean,
                       SIGMA_Q, bpf_params, ffbsi_key, M=M_BACKWARD)
bps_trajs.block_until_ready()
t_bps = time.perf_counter() - t0
bps_means = bps_trajs.mean(axis=0)[:, 0]
rmse_bps = float(jnp.sqrt(jnp.mean((bps_means - true_xs)**2)))
print(f"Bootstrap PS (FFBSi):   time={t_bps:.3f}s   RMSE vs true = {rmse_bps:.4f}"
      f"  (M={M_BACKWARD})")
print(f"Twisted PF vs FFBSi:    RMSE  {rmse_par:.4f} vs {rmse_bps:.4f}"
      f"   time  {t_par:.4f}s vs {t_bps:.3f}s   speedup {t_bps/t_par:.1f}x")


# ---------------------------------------------------------------------------
# MSE-to-target twist:  fit Gaussian per-timestep moments to FFBSi marginals.
# This mirrors the PLDS amortized-twist methodology (MSE to a strong target),
# adapted to the single-θ smoother case.  No Adam training is required: with
# a Gaussian family and Gaussian targets the MSE optimum is closed-form —
# the fitted moments ARE the targets.  Smoothing the per-timestep variance
# estimates a little keeps the proposal from being over-tight where FFBSi
# happens to have a very narrow set of trajectories.
# ---------------------------------------------------------------------------
ffbsi_means_per_t = bps_trajs.mean(axis=0)[:, 0]                       # (T,)
ffbsi_vars_per_t  = bps_trajs.var(axis=0)[:, 0] + 1e-3                  # (T,)
# Mild temporal smoothing of the variance (5-step moving average) — pure MC
# noise on the per-step std otherwise.  Mean is left untouched.
def _smooth5(a):
    a = jnp.asarray(a)
    pad = jnp.concatenate([a[:1], a[:1], a, a[-1:], a[-1:]])
    return (pad[:-4] + pad[1:-3] + pad[2:-2] + pad[3:-1] + pad[4:]) / 5.0
ffbsi_vars_smooth = _smooth5(ffbsi_vars_per_t)

twist_mse = {
    'mean':      ffbsi_means_per_t[:, None],                            # (T, 1)
    'log_scale': 0.5 * jnp.log(ffbsi_vars_smooth)[:, None],              # (T, 1)
}
tape_mse = (ys_2d, noise, twist_mse)

# Sequential and parallel runs with the MSE-to-FFBSi twist
_ = run_seq(x0s, tape_mse, _pars_w).block_until_ready()
t0 = time.perf_counter()
states_seq_mse = run_seq(x0s, tape_mse, _pars_w).block_until_ready()
t_seq_mse = time.perf_counter() - t0

key, dk_mse = jr.split(key)
params_mse = {**_pars_w, 'key': dk_mse}
_, _ = run_par(x0s, tape_mse, params_mse, max_iter=100)
t0 = time.perf_counter()
states_par_mse, iters_mse = run_par(x0s, tape_mse, params_mse, max_iter=100)
states_par_mse.block_until_ready()
t_par_mse = time.perf_counter() - t0
deer_iters_mse = int(iters_mse[0])

log_w_seq_mse = _path_log_weights(states_seq_mse, tape_mse, _pars_w)
log_w_par_mse = _path_log_weights(states_par_mse, tape_mse, _pars_w)
seq_means_mse = _weighted_mean_from_total(states_seq_mse, log_w_seq_mse)
par_means_mse = _weighted_mean_from_total(states_par_mse, log_w_par_mse)
rmse_seq_mse  = float(jnp.sqrt(jnp.mean((seq_means_mse - true_xs)**2)))
rmse_par_mse  = float(jnp.sqrt(jnp.mean((par_means_mse - true_xs)**2)))
ess_par_mse   = float(1.0 / jnp.sum(jax.nn.softmax(log_w_par_mse) ** 2))

print(f"\nMSE twist (FFBSi target):")
print(f"  seq:  time={t_seq_mse:.4f}s   RMSE = {rmse_seq_mse:.4f}")
print(f"  par:  time={t_par_mse:.4f}s   RMSE = {rmse_par_mse:.4f}"
      f"  ({deer_iters_mse} DEER iters,  path ESS = {ess_par_mse:.1f}/{N})")
print(f"  vs FFBSi:  RMSE {rmse_par_mse:.4f} vs {rmse_bps:.4f}"
      f"   time {t_par_mse:.4f}s vs {t_bps:.3f}s   speedup {t_bps/t_par_mse:.1f}x")


# ---------------------------------------------------------------------------
# Normalizing-flow conditional twist
#
# Architecture:
#   1. Backward GRU on y_{1:T}        →   h_t  (shared across particles)
#   2. Per-(particle, t) flow-param MLP:
#        cond_t = h_t                                   (precomputed)
#        (a, b, k) = MLP([x_{t-1}, h_t])                 (per-particle)
#   3. Flow:  x_t = a + e^b · (z + tanh(k) · tanh(z)),  z ~ N(0, 1)
#
# Always monotonic in z:
#   ∂x_t/∂z = e^b · (1 + tanh(k) · sech^2(z))   >  e^b · (1 − |tanh(k)|)  > 0
# Smoothly differentiable in x_{t-1} via the MLP's gradient → DEER-friendly.
#
# Trained by maximum likelihood on FFBSi-paired (x_{t-1}, x_t) samples.
# Density via 8 Newton iterations to invert the flow.
# ---------------------------------------------------------------------------
H_GRU_NN = 32
H_MLP_NN = 32
NEWTON_ITERS = 8


def _init_linear_nn(key, d_in, d_out, scale=0.02):
    return {'W': jr.normal(key, (d_in, d_out)) * scale,
            'b': jnp.zeros(d_out)}


def init_flow_params(key):
    keys = jr.split(key, 8)
    p = {}
    # Backward GRU on y
    d_gru_in = 1
    p['gru_Wz'] = _init_linear_nn(keys[0], d_gru_in + H_GRU_NN, H_GRU_NN, 0.02)
    p['gru_Wr'] = _init_linear_nn(keys[1], d_gru_in + H_GRU_NN, H_GRU_NN, 0.02)
    p['gru_Wh'] = _init_linear_nn(keys[2], d_gru_in + H_GRU_NN, H_GRU_NN, 0.02)
    # Conditional MLP that maps (x_prev, h_t) -> (a, b, k)
    p['mlp1']   = _init_linear_nn(keys[3], 1 + H_GRU_NN, H_MLP_NN, 0.1)
    p['mlp2']   = _init_linear_nn(keys[4], H_MLP_NN, 3, 0.05)
    # Initialise the readout so the flow starts close to the identity Gaussian
    # with mean = x_prev * phi (we'll add the prior mean inside the flow)
    p['mlp2']['b'] = jnp.array([0.0, jnp.log(SIGMA_Q), 0.0])  # a, b=log(σ_q), k
    return p


def linear_(p, x):
    return x @ p['W'] + p['b']


def compute_gru_hiddens(net_params, ys_flat):
    """Backward GRU on y_{1:T}.  Returns (T, H_GRU_NN) tensor of h_t."""
    def gru_step(h, y_t):
        inp = jnp.array([y_t])
        combined = jnp.concatenate([inp, h])
        z = jax.nn.sigmoid(linear_(net_params['gru_Wz'], combined))
        r = jax.nn.sigmoid(linear_(net_params['gru_Wr'], combined))
        combined_r = jnp.concatenate([inp, r * h])
        h_tilde = jnp.tanh(linear_(net_params['gru_Wh'], combined_r))
        h_new = (1.0 - z) * h + z * h_tilde
        return h_new, h_new
    h0 = jnp.zeros(H_GRU_NN)
    _, hiddens_rev = jax.lax.scan(gru_step, h0, ys_flat[::-1])
    return hiddens_rev[::-1]                                       # (T, H_GRU_NN)


def flow_params_from_xprev(net_params, x_prev_scalar, h_t):
    """Returns (a, b, k) at one (particle, t)."""
    inp = jnp.concatenate([jnp.array([x_prev_scalar]), h_t])
    h1 = jax.nn.tanh(linear_(net_params['mlp1'], inp))
    out = linear_(net_params['mlp2'], h1)
    return out[0], out[1], out[2]


def flow_forward(z, a, b, k):
    """x = a + e^b · (z + tanh(k) · tanh(z))."""
    return a + jnp.exp(b) * (z + jnp.tanh(k) * jnp.tanh(z))


def flow_log_jac_dz(z, b, k):
    """log |dx/dz| = b + log(1 + tanh(k) · sech^2(z))."""
    return b + jnp.log(1.0 + jnp.tanh(k) * (1.0 / jnp.cosh(z) ** 2) + 1e-9)


def flow_inverse(x, a, b, k, n_iter=NEWTON_ITERS):
    """Newton-iterate to find z such that flow_forward(z, a, b, k) = x."""
    y = (x - a) / jnp.exp(b)
    z = y / (1.0 + jnp.tanh(k))                # initial guess (close near 0)
    for _ in range(n_iter):
        f = z + jnp.tanh(k) * jnp.tanh(z) - y
        df = 1.0 + jnp.tanh(k) * (1.0 / jnp.cosh(z) ** 2)
        z = z - f / df
    return z


def flow_log_density(x, a, b, k):
    """log q(x | x_prev, h_t) = log φ(z) − log |dx/dz| with z = T^{-1}(x)."""
    z = flow_inverse(x, a, b, k)
    log_phi = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi)
    return log_phi - flow_log_jac_dz(z, b, k)


# DEER-compatible step + log-weight functions for the flow twist.
# We pass `h_t` as the `twist` argument of the driver tape so that DEER's
# step function only needs to do per-(particle, t) MLP calls.
def flow_step_fn(state, driver, params):
    obs, noise, twist_t = driver       # twist_t = {'h': (H_GRU_NN,)}
    x_prev = state[0]
    z = noise[0]
    a, b, k = flow_params_from_xprev(params['flow_net'], x_prev, twist_t['h'])
    x_new = flow_forward(z, a, b, k)
    return jnp.array([x_new])


def flow_log_weight_fn(x_prev, x_next, obs, twist_t, params):
    phi = params['phi']; sq = params['sigma_q']; beta = params['beta']
    x_prev_s = x_prev[0]; x_t = x_next[0]
    a, b, k = flow_params_from_xprev(params['flow_net'], x_prev_s, twist_t['h'])
    # Observation likelihood (stochvol)
    log_lik = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(beta)
               - x_t / 2.0
               - obs[0]**2 / (2.0 * beta**2 * jnp.exp(x_t)))
    # Transition prior
    log_trans = (-0.5 * (x_t - phi * x_prev_s)**2 / sq**2
                 - 0.5 * jnp.log(2.0 * jnp.pi * sq**2))
    # Flow density
    log_q = flow_log_density(x_t, a, b, k)
    return log_lik + log_trans - log_q


tpf_flow = TwistedParallelParticleFilter(flow_step_fn, flow_log_weight_fn, D=1)


# Training: maximum likelihood on FFBSi-paired (x_{t-1}, x_t) samples.
# bps_trajs has shape (M, T, D=1); pairs are (bps_trajs[:, t-1], bps_trajs[:, t]).
ffbsi_xs = bps_trajs[:, :, 0]                                       # (M, T)


def flow_nll(net_params):
    """NLL of FFBSi pairs under the flow.  Sum over (m, t) of -log q(x_t|x_{t-1})."""
    h_seq = compute_gru_hiddens(net_params, ys_2d[:, 0])             # (T, H_GRU_NN)

    def per_pair_logq(x_prev_s, x_t_s, h_t):
        a, b, k = flow_params_from_xprev(net_params, x_prev_s, h_t)
        return flow_log_density(x_t_s, a, b, k)

    # vmap over m (M trajectories) and t (T-1 transitions)
    def per_trajectory(traj):
        return jax.vmap(per_pair_logq)(traj[:-1], traj[1:], h_seq[1:])

    log_q_all = jax.vmap(per_trajectory)(ffbsi_xs)                   # (M, T-1)
    return -jnp.mean(log_q_all)


_flow_grad = jax.jit(jax.value_and_grad(flow_nll))


def _adam_init_tree(params):
    return (jax.tree_util.tree_map(jnp.zeros_like, params),
            jax.tree_util.tree_map(jnp.zeros_like, params),
            0)


def _adam_step_tree(grads, params, state, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m_new = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v_new = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi**2, v, grads)
    bc1, bc2 = 1.0 - b1**t, 1.0 - b2**t
    upd = jax.tree_util.tree_map(
        lambda mi, vi: -lr * (mi / bc1) / (jnp.sqrt(vi / bc2) + eps),
        m_new, v_new)
    new_params = jax.tree_util.tree_map(lambda p_, u_: p_ + u_, params, upd)
    return new_params, (m_new, v_new, t)


key, flow_init_key = jr.split(key)
flow_net_params = init_flow_params(flow_init_key)
flow_opt_state = _adam_init_tree(flow_net_params)

print("\nLearning normalizing-flow twist (conditional on x_{t-1}, ML on FFBSi pairs) ...")
N_STEPS_FLOW = 1500
flow_losses = []
t0 = time.perf_counter()
for i in range(N_STEPS_FLOW):
    loss_v, grads = _flow_grad(flow_net_params)
    flow_net_params, flow_opt_state = _adam_step_tree(
        grads, flow_net_params, flow_opt_state, lr=3e-3)
    flow_losses.append(float(loss_v))
    if (i + 1) % 100 == 0:
        print(f"  step {i+1:>4}/{N_STEPS_FLOW}  NLL = {loss_v:.4f}")
t_flow_train = time.perf_counter() - t0
print(f"  trained in {t_flow_train:.1f}s")


# Build the tape with per-(t, particle) GRU hidden as the "twist" pytree.
# We share h_t across particles; the per-particle dependence comes from
# the MLP applied to (x_prev, h_t) inside the step.
flow_h_seq = compute_gru_hiddens(flow_net_params, ys_2d[:, 0])      # (T, H_GRU_NN)
twist_flow = {'h': flow_h_seq}                                       # leaves shape (T, H_GRU_NN)
key, flow_tape_key = jr.split(key)
tape_flow = tpf_flow.make_tape(ys_2d, twist_flow, flow_tape_key, N)

# Pass the trained flow_net_params into the model params dict so the step
# function can pull them out.
flow_pars = {**_pars_w, 'flow_net': flow_net_params}

run_seq_flow = jax.jit(tpf_flow.run_sequential)
_ = run_seq_flow(x0s, tape_flow, flow_pars).block_until_ready()
t0 = time.perf_counter()
states_seq_flow = run_seq_flow(x0s, tape_flow, flow_pars).block_until_ready()
t_seq_flow = time.perf_counter() - t0

key, dk_flow = jr.split(key)
flow_pars_par = {**flow_pars, 'key': dk_flow}
run_par_flow = jax.jit(tpf_flow.run_parallel,
                        static_argnames=("max_iter", "num_samples"))
_, _ = run_par_flow(x0s, tape_flow, flow_pars_par, max_iter=100)
t0 = time.perf_counter()
states_par_flow, iters_flow = run_par_flow(
    x0s, tape_flow, flow_pars_par, max_iter=100)
states_par_flow.block_until_ready()
t_par_flow = time.perf_counter() - t0
deer_iters_flow = int(iters_flow[0])

log_w_seq_flow = tpf_flow.compute_log_weights(x0s, states_seq_flow, tape_flow, flow_pars).sum(axis=1)
log_w_par_flow = tpf_flow.compute_log_weights(x0s, states_par_flow, tape_flow, flow_pars).sum(axis=1)
seq_means_flow = jnp.einsum('n,ntd->td', jax.nn.softmax(log_w_seq_flow), states_seq_flow)[:, 0]
par_means_flow = jnp.einsum('n,ntd->td', jax.nn.softmax(log_w_par_flow), states_par_flow)[:, 0]
rmse_seq_flow = float(jnp.sqrt(jnp.mean((seq_means_flow - true_xs)**2)))
rmse_par_flow = float(jnp.sqrt(jnp.mean((par_means_flow - true_xs)**2)))
ess_par_flow  = float(1.0 / jnp.sum(jax.nn.softmax(log_w_par_flow) ** 2))

# Use the seq results for fallback fields needed by the figure block below.
rmse_par_nn = rmse_par_flow
par_means_nn = par_means_flow
t_par_nn = t_par_flow

print(f"\nFlow twist (conditional on x_{{t-1}}, ML on FFBSi pairs):")
print(f"  seq:  time={t_seq_flow:.4f}s   RMSE = {rmse_seq_flow:.4f}")
print(f"  par:  time={t_par_flow:.4f}s   RMSE = {rmse_par_flow:.4f}"
      f"  ({deer_iters_flow} DEER iters,  path ESS = {ess_par_flow:.1f}/{N})")
print(f"  vs FFBSi:  RMSE {rmse_par_flow:.4f} vs {rmse_bps:.4f}"
      f"   time {t_par_flow:.4f}s vs {t_bps:.3f}s   speedup {t_bps/t_par_flow:.1f}x")




# Plot — paper_style 2x2 layout
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "figures"))
import paper_style as ps  # noqa: E402
import numpy as _np
ps.apply()

t_idx = _np.arange(T)
true_xs_np        = _np.asarray(true_xs)
seq_means_np      = _np.asarray(seq_means)
par_means_np      = _np.asarray(par_means)
par_means_mse_np  = _np.asarray(par_means_mse)
par_means_nn_np   = _np.asarray(par_means_nn)
none_means_np     = _np.asarray(none_means)
bps_means_np      = _np.asarray(bps_means)
bpf_filt_np       = _np.asarray(bpf_filt[:, 0])
twist_mean_np      = _np.asarray(twist_mean[:, 0])
twist_log_scale_np = _np.asarray(twist_log_scale[:, 0])

WIN = slice(400, 700)

fig, axes = plt.subplots(2, 2, figsize=(ps.COL_FULL, 4.4),
                         constrained_layout=True)

# (a) Smoothing estimates zoomed to a 300-step window — actually legible
ax = axes[0, 0]
ax.plot(t_idx[WIN], true_xs_np[WIN], color='black', lw=1.0, alpha=0.85,
        label='true $x_t$')
ax.plot(t_idx[WIN], none_means_np[WIN], color=ps.PALETTE['bpf'], lw=0.8,
        alpha=0.55, label=f'no twist (RMSE {rmse_none:.3f})')
ax.plot(t_idx[WIN], par_means_np[WIN], color=ps.PALETTE['par'], lw=0.8,
        alpha=0.6, label=f'ELBO twist (RMSE {rmse_par:.3f})')
ax.plot(t_idx[WIN], par_means_mse_np[WIN], color=ps.PALETTE['twisted_seq'],
        lw=0.8, alpha=0.7,
        label=f'per-$t$ MSE twist (RMSE {rmse_par_mse:.3f})')
ax.plot(t_idx[WIN], par_means_nn_np[WIN], color=ps.PALETTE['par_alt'],
        lw=1.1, alpha=0.95,
        label=f'NN MSE twist (RMSE {rmse_par_nn:.3f})')
ax.set_xlabel('Timestep')
ax.set_ylabel('$x_t$')
ax.set_title(f'Smoothing estimates (window {WIN.start}--{WIN.stop})')
ax.legend(loc='lower left', fontsize=5.5, ncol=1, framealpha=0.9)

# (b) Twist-learning ELBO curve
ax = axes[0, 1]
losses_arr = _np.asarray(losses)
ax.plot(_np.arange(len(losses_arr)), losses_arr,
        color=ps.PALETTE['par_alt'], lw=1.0)
ax.set_xlabel('Adam step')
ax.set_ylabel('Neg-ELBO')
ax.set_title(f'Twist training ({len(losses_arr)} steps)')

# (c) RMSE bar chart — methods on x-axis
ax = axes[1, 0]
methods = ['BPF', 'FFBSi', 'No\ntwist',
           'ELBO\ntwist', 'per-$t$\nMSE', 'NN\nMSE']
rmses   = [rmse_bpf,   rmse_bps, rmse_none, rmse_par, rmse_par_mse, rmse_par_nn]
times   = [t_bpf,      t_bps,    t_seq,     t_par,    t_par_mse,    t_par_nn]
colors  = [ps.PALETTE['bpf'], ps.PALETTE['ffbsi'],
           ps.PALETTE['standard'], ps.PALETTE['par'],
           ps.PALETTE['twisted_seq'], ps.PALETTE['par_alt']]
xpos = _np.arange(len(methods))
ax.bar(xpos, rmses, color=colors, edgecolor='k', lw=0.5,
       alpha=0.88, width=0.7)
ax.set_xticks(xpos)
ax.set_xticklabels(methods, fontsize=6.0)
ax.set_ylabel('RMSE vs.\\ true $x$')
ax.set_title('Smoother accuracy')
top = max(rmses) * 1.18
ax.set_ylim(0, top)
for xi, r in zip(xpos, rmses):
    ax.text(xi, r + top * 0.01, f'{r:.3f}', ha='center', va='bottom',
            fontsize=5.5, color='0.15')

# (d) Wall-clock bar chart — same x-categories, log y
ax = axes[1, 1]
ax.bar(xpos, times, color=colors, edgecolor='k', lw=0.5,
       alpha=0.88, width=0.7)
ax.set_yscale('log')
ax.set_xticks(xpos)
ax.set_xticklabels(methods, fontsize=6.0)
ax.set_ylabel('Wall-clock (s)')
ax.set_title('Wall-clock cost')
ymax = max(times) * 12.0
ax.set_ylim(min(times) * 0.4, ymax)
for xi, v in zip(xpos, times):
    ax.text(xi, v * 1.5, f'{v*1e3:.1f} ms' if v < 1.0 else f'{v:.2f} s',
            ha='center', va='bottom', fontsize=5.5, color='0.15')

# Headline speedup: NN-MSE-twist par vs FFBSi (both fair smoothers)
sp_nn_vs_ffbsi = t_bps / t_par_nn if t_par_nn > 0 else 0.0
ax.annotate(
    f'NN MSE twist:\n$\\mathbf{{{sp_nn_vs_ffbsi:.0f}\\times}}$ vs. FFBSi',
    xy=(5, t_par_nn),
    xytext=(0.04, 0.97), textcoords='axes fraction',
    fontsize=7, color=ps.PALETTE['par_alt'],
    arrowprops=dict(arrowstyle='->', color=ps.PALETTE['par_alt'],
                    lw=0.7, alpha=0.85, shrinkA=2, shrinkB=2),
    ha='left', va='top',
    bbox=dict(boxstyle='round,pad=0.22', fc='white',
              ec=ps.PALETTE['par_alt'], lw=0.5, alpha=0.92),
)

fig.suptitle(
    f'Twisted parallel particle smoother on stochastic volatility  '
    f'($T{{=}}{T}$, $N{{=}}{N}$).  '
    f'Learned twist: {deer_iters} DEER iters, '
    f'{t_seq/t_par:.0f}$\\times$ speedup over Tw seq.',
    fontsize=9.5, y=1.03,
)
plt.savefig('twisted_pf_stochvol.png')
plt.savefig('twisted_pf_stochvol.svg')
plt.close(fig)
print('Saved plot to twisted_pf_stochvol.png and twisted_pf_stochvol.svg')
