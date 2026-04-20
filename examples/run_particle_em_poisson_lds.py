"""
run_particle_em_poisson_lds.py

Particle EM for a D-dimensional Poisson linear dynamical system (PLDS).

Model
-----
  x_t = A x_{t-1} + sigma_q * eps_t,   eps_t ~ N(0, I_D)
  y_{t,d} ~ Poisson(exp(x_{t,d}))       independently for d = 1, ..., D

The E-step uses a twisted parallel particle smoother to compute
sufficient statistics (expected outer products of smoothed states).
The M-step is closed-form matrix least squares for A and sigma_q.

Compares three E-step implementations:
  1. Bootstrap PF  (sequential, resampling)
  2. Twisted PF    (sequential, no resampling)
  3. Twisted PF    (parallel via DEER)
"""

import time
from functools import partial
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from src.particle_filter import TwistedParallelParticleFilter
from src.baselines import bootstrap_pf, ffbsi_smoother

# Model configuration
D = 5            # latent state dimension
T = 2000         # time series length
N = 256          # particles
N_EM = 50        # EM iterations

#  Generate a random stable transition matrix 
def make_stable_A(key, D, eig_range=(0.80, 0.95)):
    """
    Random D x D matrix with eigenvalues uniformly in eig_range.
    Uses A = Q @ diag(lambdas) @ Q^T with random orthogonal Q,
    giving a symmetric matrix with prescribed spectrum.
    """
    k1, k2 = jr.split(key)
    # Random orthogonal matrix via QR
    M = jr.normal(k1, (D, D))
    Q, _ = jnp.linalg.qr(M)
    # Eigenvalues uniformly in the specified range
    lambdas = jr.uniform(k2, (D,), minval=eig_range[0], maxval=eig_range[1])
    return Q @ jnp.diag(lambdas) @ Q.T

# 
# EKF for Poisson observations (provides BIF linearisation points)
# 
def ekf_poisson(ys, A, sigma_q, x0, P0):
    """
    Extended Kalman filter for the PLDS.

    Observation model: y_d ~ Poisson(exp(x_d)).
    Linearisation of the observation log-likelihood at the predicted mean:
      d/dx exp(x) = exp(x) ⇒ H = diag(exp(m_pred))
      Var[y|x] = exp(x)     ⇒ R = diag(exp(m_pred))
    """
    Q = sigma_q**2 * jnp.eye(D)

    def step(carry, y):
        m, P = carry
        # Predict
        m_pred = A @ m
        P_pred = A @ P @ A.T + Q
        # Linearise Poisson observation at m_pred
        lam = jnp.exp(jnp.clip(m_pred, -10.0, 10.0))   # predicted rate
        H = jnp.diag(lam)                                 # Jacobian
        R = jnp.diag(lam)                                 # obs noise cov
        # Update
        S = H @ P_pred @ H + R
        K = jnp.linalg.solve(S.T, (P_pred @ H).T).T
        innov = y - lam                                    # y - E[y|m_pred]
        m_new = m_pred + K @ innov
        P_new = (jnp.eye(D) - K @ H) @ P_pred
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new)

    _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(
        step, (x0, P0), ys
    )
    return m_preds, P_preds, m_filts, P_filts


def ekf_smoother_poisson(ys, A, sigma_q, x0, P0):
    """EKF smoother: forward filter then backward RTS pass."""
    Q = sigma_q**2 * jnp.eye(D)
    m_preds, P_preds, m_filts, P_filts = ekf_poisson(ys, A, sigma_q, x0, P0)

    def bwd_step(carry, t):
        m_s, P_s = carry
        G = jnp.linalg.solve(P_preds[t + 1].T, (P_filts[t] @ A.T).T).T
        m_s_new = m_filts[t] + G @ (m_s - m_preds[t + 1])
        P_s_new = P_filts[t] + G @ (P_s - P_preds[t + 1]) @ G.T
        return (m_s_new, P_s_new), (m_s_new, P_s_new)

    ts = jnp.arange(T - 2, -1, -1)
    _, (ms_rev, Ps_rev) = jax.lax.scan(
        bwd_step, (m_filts[-1], P_filts[-1]), ts
    )
    smooth_means = jnp.concatenate([ms_rev[::-1], m_filts[-1:]])
    smooth_covs  = jnp.concatenate([Ps_rev[::-1], P_filts[-1:]])
    return smooth_means, smooth_covs



# Extended backward information filter (BIF) for Poisson observations
def ebif_poisson(ys, A, sigma_q, ekf_means):
    """
    Extended BIF: backward Gaussian message ψ_t(x_t) ≈ p(y_{t:T}|x_t).

    Poisson observation information at linearisation point x0:
      J_obs = diag(exp(x0))
      h_obs = y - exp(x0) + diag(exp(x0)) @ x0

    Returns full-covariance twist parameters (Cholesky form) for
    the TwistedParallelParticleFilter.
    """
    Q = sigma_q**2 * jnp.eye(D)

    def obs_info(y, x0):
        lam = jnp.exp(jnp.clip(x0, -10.0, 10.0))
        J_obs = jnp.diag(lam)
        h_obs = y - lam + lam * x0        # gradient + Hessian * x0
        return J_obs, h_obs

    def step(carry, inputs):
        J, h = carry
        y_t, lin_pt = inputs
        # Backward prediction through transition x_t = A x_{t-1} + noise(Q)
        J_inv = jnp.linalg.inv(J)
        P = J_inv + Q
        P_inv = jnp.linalg.inv(P)
        J_pred = A.T @ P_inv @ A
        h_pred = A.T @ P_inv @ (J_inv @ h)
        # Add observation information
        J_obs, h_obs = obs_info(y_t, lin_pt)
        return (J_pred + J_obs, h_pred + h_obs), (J_pred + J_obs, h_pred + h_obs)

    J_obs_T, h_obs_T = obs_info(ys[-1], ekf_means[-1])
    _, (Js_rev, hs_rev) = jax.lax.scan(
        step, (J_obs_T, h_obs_T),
        (ys[-2::-1], ekf_means[-2::-1]),
    )

    Js = jnp.concatenate([Js_rev[::-1], J_obs_T[None]], axis=0)   # (T, D, D)
    hs = jnp.concatenate([hs_rev[::-1], h_obs_T[None]], axis=0)   # (T, D)

    # Convert information-form (J, h) → Cholesky twist params
    def to_twist(J_t, h_t):
        Sigma = jnp.linalg.inv(J_t)
        mu = Sigma @ h_t
        L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(D))
        log_diag = jnp.log(jnp.diag(L))
        offdiag = jnp.tril(L, k=-1)
        return mu, log_diag, offdiag

    means, log_diags, offdiags = jax.vmap(to_twist)(Js, hs)
    return {
        'mean':          means,          # (T, D)
        'chol_log_diag': log_diags,      # (T, D)
        'chol_offdiag':  offdiags,       # (T, D, D)
    }


# Twisted PF step + weight functions (full-covariance twist)
def build_chol(chol_log_diag, chol_offdiag):
    return jnp.tril(chol_offdiag, k=-1) + jnp.diag(jnp.exp(chol_log_diag))


def _twist_posterior(f_prev, twist, sigma_q):
    """Full-cov Gaussian posterior: q(x_t|x_{t-1}) ∝ N(f,Q) · ψ_t(x_t)."""
    mu_twist = twist['mean']
    L_twist  = build_chol(twist['chol_log_diag'], twist['chol_offdiag'])

    prec_q = 1.0 / sigma_q**2
    Sigma_twist_inv = jax.scipy.linalg.cho_solve((L_twist, True), jnp.eye(D))
    prec_post  = prec_q * jnp.eye(D) + Sigma_twist_inv
    Sigma_post = jnp.linalg.inv(prec_post)
    mean_post  = Sigma_post @ (prec_q * f_prev + Sigma_twist_inv @ mu_twist)
    L_post     = jnp.linalg.cholesky(Sigma_post)
    return mean_post, L_post


def twisted_step_fn(x_prev, driver, params):
    """Per-particle twisted transition."""
    obs, noise, twist = driver
    A = params['A']
    sigma_q = params['sigma_q']
    f_prev = A @ x_prev
    mean_post, L_post = _twist_posterior(f_prev, twist, sigma_q)
    return mean_post + L_post @ noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    """Per-particle incremental log importance weight."""
    A = params['A']
    sigma_q = params['sigma_q']
    f_prev = A @ x_prev

    # log p(y_t | x_t) — Poisson
    log_lik = jnp.sum(obs * x_next - jnp.exp(x_next) - jax.lax.lgamma(obs + 1))

    # log p(x_t | x_{t-1}) — Gaussian transition
    diff_trans = x_next - f_prev
    log_trans = (-0.5 * jnp.sum(diff_trans**2) / sigma_q**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_q**2))

    # log q(x_t | x_{t-1}) — twisted proposal
    mean_post, L_post = _twist_posterior(f_prev, twist, sigma_q)
    diff_prop = x_next - mean_post
    z = jax.scipy.linalg.solve_triangular(L_post, diff_prop, lower=True)
    log_prop = (-0.5 * jnp.sum(z**2)
                - jnp.sum(jnp.log(jnp.diag(L_post)))
                - 0.5 * D * jnp.log(2 * jnp.pi))

    return log_lik + log_trans - log_prop



# Bootstrap PF for comparison
def bootstrap_pf_smooth(ys, A, sigma_q, x0, key, N):
    """
    Bootstrap PF with systematic resampling.
    Returns weighted filter means and per-step particles / weights
    for computing EM sufficient statistics.
    """
    Q_chol = sigma_q * jnp.eye(D)

    def _systematic_resample(key, log_w, particles):
        w = jax.nn.softmax(log_w)
        n = w.shape[0]
        u = (jnp.arange(n) + jr.uniform(key)) / n
        cumw = jnp.cumsum(w)
        idx = jnp.clip(jnp.searchsorted(cumw, u), 0, n - 1)
        return particles[idx]

    def step(carry, yt):
        particles, key = carry
        key, trans_key, resamp_key = jr.split(key, 3)
        # Propagate
        particles_new = jax.vmap(lambda x: A @ x)(particles) + \
                        sigma_q * jr.normal(trans_key, particles.shape)
        # Poisson log-weights
        log_w = jax.vmap(
            lambda x: jnp.sum(yt * x - jnp.exp(x) - jax.lax.lgamma(yt + 1))
        )(particles_new)
        log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)
        w = jnp.exp(log_w_norm)
        filt_mean = jnp.sum(w[:, None] * particles_new, axis=0)

        # Resample
        particles_resamp = _systematic_resample(resamp_key, log_w, particles_new)
        return (particles_resamp, key), (filt_mean, particles_new, log_w_norm)

    key, init_key = jr.split(key)
    particles0 = x0[None, :] + 0.5 * jr.normal(init_key, (N, D))
    (_, _), (filt_means, all_parts, all_log_w) = jax.lax.scan(
        step, (particles0, key), ys
    )
    return filt_means, all_parts, all_log_w


# 
# EM sufficient statistics
# 
def compute_suff_stats_twisted(x0s, states, log_weights):
    """
    Compute EM sufficient statistics from twisted PF output.

    states      : (N, T, D)
    log_weights : (N, T)

    Returns E[x_t x_{t-1}^T], E[x_{t-1} x_{t-1}^T], E[x_t x_t^T]
    summed over t, weighted by path importance weights.
    """
    total_log_w = jnp.sum(log_weights, axis=1)          # (N,)
    w = jax.nn.softmax(total_log_w)                      # (N,)

    # Prepend x0s as "t=0" to form x_prev
    full = jnp.concatenate([x0s[:, None, :], states], axis=1)  # (N, T+1, D)
    x_prev = full[:, :-1, :]    # (N, T, D)
    x_curr = full[:, 1:, :]     # (N, T, D)

    # Weighted outer products summed over t
    # S_xx = sum_t E[x_t x_t^T]
    # S_xp = sum_t E[x_t x_{t-1}^T]
    # S_pp = sum_t E[x_{t-1} x_{t-1}^T]
    def per_particle(x_c, x_p, wi):
        # x_c, x_p: (T, D)
        S_xp = jnp.einsum('td,te->de', x_c, x_p) * wi   # (D, D)
        S_pp = jnp.einsum('td,te->de', x_p, x_p) * wi
        S_xx = jnp.einsum('td,te->de', x_c, x_c) * wi
        return S_xp, S_pp, S_xx

    S_xp, S_pp, S_xx = jax.vmap(per_particle)(x_curr, x_prev, w)
    return jnp.sum(S_xp, 0), jnp.sum(S_pp, 0), jnp.sum(S_xx, 0)


def compute_suff_stats_bootstrap(filt_means):
    """
    Simple sufficient statistics from bootstrap filter means.
    Uses plug-in estimates E[x_t] ≈ filter mean.
    """
    x_prev = filt_means[:-1]    # (T-1, D)
    x_curr = filt_means[1:]     # (T-1, D)
    S_xp = x_curr.T @ x_prev   # (D, D)
    S_pp = x_prev.T @ x_prev
    S_xx = x_curr.T @ x_curr
    return S_xp, S_pp, S_xx


def m_step(S_xp, S_pp, S_xx, T_eff):
    """M-step: closed-form updates for A and sigma_q (regularised)."""
    # Regularise S_pp to prevent singular solve
    S_pp_reg = S_pp + 1e-4 * jnp.eye(D)
    A_new = jnp.linalg.solve(S_pp_reg.T, S_xp.T).T
    # Clip spectral radius to keep dynamics stable
    eigs = jnp.linalg.eigvals(A_new)
    max_eig = jnp.max(jnp.abs(eigs))
    A_new = jnp.where(max_eig > 0.99, A_new * 0.99 / max_eig, A_new)
    residual = S_xx - A_new @ S_xp.T - S_xp @ A_new.T + A_new @ S_pp_reg @ A_new.T
    sigma_q_new = jnp.sqrt(jnp.clip(jnp.trace(residual) / (T_eff * D), 1e-4, 2.0))
    return A_new, sigma_q_new


def weighted_mean(states, log_weights):
    total_log_w = jnp.sum(log_weights, axis=1)
    w = jax.nn.softmax(total_log_w)
    return jnp.einsum('n,ntd->td', w, states)


# Simulate data 
key = jr.PRNGKey(0)
key, Akey, simkey = jr.split(key, 3)
A_TRUE = make_stable_A(Akey, D, eig_range=(0.80, 0.95))
SIGMA_Q_TRUE = 0.3

print(f"Poisson LDS: D={D}, T={T}, N={N}")
print(f"True A spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_TRUE)))):.4f}")
print(f"True sigma_q: {SIGMA_Q_TRUE}")

k1, k2 = jr.split(simkey)
x0_true = jnp.zeros(D)

def sim_step(x, noise):
    xn = A_TRUE @ x + SIGMA_Q_TRUE * noise
    yn = jr.poisson(jr.fold_in(k2, int(jnp.sum(xn * 1000) % 2**31)),
                    jnp.exp(jnp.clip(xn, -10, 10))).astype(jnp.float32)
    return xn, (xn, yn)

# Simulate with explicit keys per step for reproducibility
key, sim_key = jr.split(key)
sim_keys = jr.split(sim_key, T)

def sim_step_keyed(x, inputs):
    noise, step_key = inputs
    xn = A_TRUE @ x + SIGMA_Q_TRUE * noise
    yn = jr.poisson(step_key, jnp.exp(jnp.clip(xn, -10, 10))).astype(jnp.float32)
    return xn, (xn, yn)

noise_sim = jr.normal(k1, (T, D))
_, (true_xs, ys) = jax.lax.scan(sim_step_keyed, x0_true, (noise_sim, sim_keys))

print(f"Observation range: [{float(ys.min()):.0f}, {float(ys.max()):.0f}], "
      f"mean={float(ys.mean()):.2f}")

# Particle EM

# Initial parameter guesses — start from 0.85*I (knows scale, not structure)
A_init = 0.85 * jnp.eye(D)
sigma_q_init = 0.5

P0 = jnp.eye(D)
x0_guess = jnp.zeros(D)

# Setup twisted PF 
tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)

key, x0_key, noise_key = jr.split(key, 3)
x0s = jr.normal(x0_key, (N, D)) * 0.5
pf_noise = jr.normal(noise_key, (N, T, D))

#  EM loop: Twisted PF (sequential) 
print(f"\n{'='*70}")
print(f"Running Particle EM ({N_EM} iterations)")
print(f"{'='*70}")

def run_em_twisted(A_init, sigma_q_init, parallel=False, max_deer=100):
    """Run particle EM using twisted PF for E-step."""

    # JIT the whole E-step + M-step once — non-jit calls inside the loop
    # cause full re-trace per EM iter (~2s overhead per call for the
    # parallel PF; see debug_pls_pf_timing.py).
    @jax.jit
    def em_step_seq(A_curr, sq_curr):
        ekf_means, _ = ekf_smoother_poisson(ys, A_curr, sq_curr, x0_guess, P0)
        twist = ebif_poisson(ys, A_curr, sq_curr, ekf_means)
        tape = (ys, pf_noise, twist)
        model_params = {'A': A_curr, 'sigma_q': sq_curr}
        states = tpf.run_sequential(x0s, tape, model_params)
        log_ws = tpf.compute_log_weights(x0s, states, tape, model_params)
        S_xp, S_pp, S_xx = compute_suff_stats_twisted(x0s, states, log_ws)
        A_new, sq_new = m_step(S_xp, S_pp, S_xx, T)
        return A_new, sq_new

    @partial(jax.jit, static_argnames=("max_iter",))
    def em_step_par(A_curr, sq_curr, key, max_iter):
        ekf_means, _ = ekf_smoother_poisson(ys, A_curr, sq_curr, x0_guess, P0)
        twist = ebif_poisson(ys, A_curr, sq_curr, ekf_means)
        tape = (ys, pf_noise, twist)
        model_params = {'A': A_curr, 'sigma_q': sq_curr}
        params_par = {**model_params, 'key': key}
        states, iters = tpf.run_parallel(
            x0s, tape, params_par, yinit_guess=ekf_means, max_iter=max_iter,
        )
        log_ws = tpf.compute_log_weights(x0s, states, tape, model_params)
        S_xp, S_pp, S_xx = compute_suff_stats_twisted(x0s, states, log_ws)
        A_new, sq_new = m_step(S_xp, S_pp, S_xx, T)
        return A_new, sq_new, iters

    A_curr = A_init
    sq_curr = sigma_q_init
    history = {'A_err': [], 'sq_err': [], 'time': []}

    # Warmup compile
    if parallel:
        _ = em_step_par(A_curr, sq_curr, jr.PRNGKey(999), max_deer)
        jax.block_until_ready(_[0])
    else:
        _ = em_step_seq(A_curr, sq_curr)
        jax.block_until_ready(_[0])

    for it in range(N_EM):
        t0 = time.perf_counter()
        if parallel:
            A_new, sq_new, iters = em_step_par(A_curr, sq_curr,
                                                jr.PRNGKey(it), max_deer)
            jax.block_until_ready(A_new)
        else:
            A_new, sq_new = em_step_seq(A_curr, sq_curr)
            jax.block_until_ready(A_new)
        dt = time.perf_counter() - t0

        A_err = float(jnp.linalg.norm(A_new - A_TRUE, 'fro') /
                       jnp.linalg.norm(A_TRUE, 'fro'))
        sq_err = float(jnp.abs(sq_new - SIGMA_Q_TRUE) / SIGMA_Q_TRUE)

        history['A_err'].append(A_err)
        history['sq_err'].append(sq_err)
        history['time'].append(dt)

        if (it + 1) % 10 == 0 or it == 0:
            mode = "par" if parallel else "seq"
            iters_str = f"  iters={int(iters[0])}" if parallel else ""
            print(f"  [{mode}] it={it+1:>3}  "
                  f"||A-A*||/||A*||={A_err:.4f}  "
                  f"|sq-sq*|/sq*={sq_err:.4f}  "
                  f"time={dt:.3f}s{iters_str}")

        A_curr = A_new
        sq_curr = sq_new

    return A_curr, sq_curr, history


#  EM with bootstrap PF 
def run_em_bootstrap(A_init, sigma_q_init):
    """Run particle EM using bootstrap PF for E-step."""
    A_curr = A_init
    sq_curr = sigma_q_init
    history = {'A_err': [], 'sq_err': [], 'time': []}

    for it in range(N_EM):
        t0 = time.perf_counter()
        bpf_key = jr.PRNGKey(1000 + it)

        # E-step: bootstrap PF
        filt_means, _, _ = bootstrap_pf_smooth(
            ys, A_curr, sq_curr, x0_guess, bpf_key, N)

        # Sufficient statistics from filter means (plug-in)
        S_xp, S_pp, S_xx = compute_suff_stats_bootstrap(filt_means)

        # M-step
        A_new, sq_new = m_step(S_xp, S_pp, S_xx, T - 1)
        dt = time.perf_counter() - t0

        A_err = float(jnp.linalg.norm(A_new - A_TRUE, 'fro') /
                       jnp.linalg.norm(A_TRUE, 'fro'))
        sq_err = float(jnp.abs(sq_new - SIGMA_Q_TRUE) / SIGMA_Q_TRUE)
        history['A_err'].append(A_err)
        history['sq_err'].append(sq_err)
        history['time'].append(dt)

        if (it + 1) % 10 == 0 or it == 0:
            print(f"  [bpf] it={it+1:>3}  "
                  f"||A-A*||/||A*||={A_err:.4f}  "
                  f"|sq-sq*|/sq*={sq_err:.4f}  "
                  f"time={dt:.3f}s")

        A_curr = A_new
        sq_curr = sq_new

    return A_curr, sq_curr, history


#  Run all three EM variants 
print("\n--- Bootstrap PF EM ---")
A_bpf, sq_bpf, hist_bpf = run_em_bootstrap(A_init, sigma_q_init)

print("\n--- Twisted PF EM (sequential) ---")
A_tw_seq, sq_tw_seq, hist_tw_seq = run_em_twisted(A_init, sigma_q_init, parallel=False)

print("\n--- Twisted PF EM (parallel / DEER) ---")
A_tw_par, sq_tw_par, hist_tw_par = run_em_twisted(A_init, sigma_q_init, parallel=True)


# Summary
print(f"\n{'='*70}")
print(f"Summary — Poisson LDS Particle EM, D={D}, T={T}, N={N}, {N_EM} iters")
print(f"{'='*70}")

t_bpf_total = sum(hist_bpf['time'])
t_seq_total = sum(hist_tw_seq['time'])
t_par_total = sum(hist_tw_par['time'])

print(f"\n{'Method':<28} {'Total time':>10} {'Final ||A-A*||':>14} {'Final |sq-sq*|':>14}")
print("-" * 70)
print(f"{'Bootstrap PF EM':<28} {t_bpf_total:>9.1f}s {hist_bpf['A_err'][-1]:>14.4f} "
      f"{hist_bpf['sq_err'][-1]:>14.4f}")
print(f"{'Twisted PF EM (seq)':<28} {t_seq_total:>9.1f}s {hist_tw_seq['A_err'][-1]:>14.4f} "
      f"{hist_tw_seq['sq_err'][-1]:>14.4f}")
print(f"{'Twisted PF EM (par)':<28} {t_par_total:>9.1f}s {hist_tw_par['A_err'][-1]:>14.4f} "
      f"{hist_tw_par['sq_err'][-1]:>14.4f}")
print(f"\nSpeedup (seq/par): {t_seq_total/t_par_total:.1f}x")
print(f"Speedup (bpf/par): {t_bpf_total/t_par_total:.1f}x")

print(f"\n{'':>15} {'True':>8} {'BPF':>8} {'Tw-seq':>8} {'Tw-par':>8}")
print("-" * 50)
print(f"{'sigma_q':>15} {SIGMA_Q_TRUE:>8.4f} {float(sq_bpf):>8.4f} "
      f"{float(sq_tw_seq):>8.4f} {float(sq_tw_par):>8.4f}")
print(f"{'||A||_F':>15} {float(jnp.linalg.norm(A_TRUE, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_bpf, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_tw_seq, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_tw_par, 'fro')):>8.4f}")
print(f"{'spec_radius(A)':>15} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_TRUE)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_bpf)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_tw_seq)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_tw_par)))):>8.4f}")

# FFBSi smoother baseline at true params
# Evaluate smoother quality head-to-head: run both smoothers at the true A,
# sigma_q and compare RMSE of smoothed latent vs the simulated ground truth.
print(f"\n{'='*70}")
print(f"Smoother comparison at true params — RMSE vs simulated latent")
print(f"{'='*70}")

def plds_transition_mean(x_prev, p):
    return p['A'] @ x_prev

def plds_log_lik(y_t, x_t, p):
    return jnp.sum(y_t * x_t - jnp.exp(x_t) - jax.lax.lgamma(y_t + 1))

true_params = {'A': A_TRUE, 'sigma_q': SIGMA_Q_TRUE}

key, bpf_key, ffbsi_key, init_key = jr.split(key, 4)
x0_particles = jr.normal(init_key, (N, D)) * 0.5

run_bpf = jax.jit(bootstrap_pf, static_argnames=("transition_mean", "log_lik"))
_ = run_bpf(ys, x0_particles, plds_transition_mean, plds_log_lik,
            SIGMA_Q_TRUE, true_params, bpf_key)

t0 = time.perf_counter()
bpf_filt, bpf_parts, bpf_log_w, bpf_ess, bpf_log_ml = run_bpf(
    ys, x0_particles, plds_transition_mean, plds_log_lik,
    SIGMA_Q_TRUE, true_params, bpf_key)
bpf_filt.block_until_ready()
t_bpf_smooth = time.perf_counter() - t0
rmse_bpf_filt = float(jnp.sqrt(jnp.mean((bpf_filt - true_xs)**2)))
print(f"Bootstrap PF (filter):  time={t_bpf_smooth:.3f}s   "
      f"RMSE vs true x={rmse_bpf_filt:.4f}   "
      f"ESS mean={float(bpf_ess.mean()):.0f}/{N}")

M_BACKWARD = N
run_ffbsi = jax.jit(ffbsi_smoother, static_argnames=("M", "transition_mean"))
_ = run_ffbsi(bpf_parts, bpf_log_w, plds_transition_mean,
              SIGMA_Q_TRUE, true_params, ffbsi_key, M=M_BACKWARD)

t0 = time.perf_counter()
bps_trajs = run_ffbsi(bpf_parts, bpf_log_w, plds_transition_mean,
                       SIGMA_Q_TRUE, true_params, ffbsi_key, M=M_BACKWARD)
bps_trajs.block_until_ready()
t_ffbsi = time.perf_counter() - t0
bps_means = bps_trajs.mean(axis=0)   # (T, D)
rmse_ffbsi = float(jnp.sqrt(jnp.mean((bps_means - true_xs)**2)))
print(f"Bootstrap PS (FFBSi):   time={t_ffbsi:.3f}s   "
      f"RMSE vs true x={rmse_ffbsi:.4f}   (M={M_BACKWARD})")

# Twisted PF at true params (sequential + parallel), head-to-head
ekf_means_true, _ = ekf_smoother_poisson(ys, A_TRUE, SIGMA_Q_TRUE, x0_guess, P0)
twist_true = ebif_poisson(ys, A_TRUE, SIGMA_Q_TRUE, ekf_means_true)
tape_true = (ys, pf_noise, twist_true)

run_seq_true = jax.jit(tpf.run_sequential)
_ = run_seq_true(x0s, tape_true, true_params)
t0 = time.perf_counter()
states_seq_true = run_seq_true(x0s, tape_true, true_params).block_until_ready()
t_tpf_seq = time.perf_counter() - t0
log_ws_seq = jax.jit(tpf.compute_log_weights)(x0s, states_seq_true, tape_true, true_params)
w_seq = jax.nn.softmax(jnp.sum(log_ws_seq, axis=1))
tpf_seq_means = jnp.einsum('n,ntd->td', w_seq, states_seq_true)
rmse_tpf_seq = float(jnp.sqrt(jnp.mean((tpf_seq_means - true_xs)**2)))

key, deer_key = jr.split(key)
params_par_true = {**true_params, 'key': deer_key}
run_par_true = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
_ = run_par_true(x0s, tape_true, params_par_true, yinit_guess=ekf_means_true, max_iter=100)
t0 = time.perf_counter()
states_par_true, deer_iters = run_par_true(
    x0s, tape_true, params_par_true, yinit_guess=ekf_means_true, max_iter=100)
states_par_true.block_until_ready()
t_tpf_par = time.perf_counter() - t0
log_ws_par = jax.jit(tpf.compute_log_weights)(x0s, states_par_true, tape_true, true_params)
w_par = jax.nn.softmax(jnp.sum(log_ws_par, axis=1))
tpf_par_means = jnp.einsum('n,ntd->td', w_par, states_par_true)
rmse_tpf_par = float(jnp.sqrt(jnp.mean((tpf_par_means - true_xs)**2)))

print(f"Twisted PF (seq):       time={t_tpf_seq:.3f}s   "
      f"RMSE vs true x={rmse_tpf_seq:.4f}")
print(f"Twisted PF (parallel):  time={t_tpf_par:.3f}s   "
      f"RMSE vs true x={rmse_tpf_par:.4f}   ({int(deer_iters[0])} DEER iters)")
print(f"\nTwisted par vs FFBSi:   RMSE {rmse_tpf_par:.4f} vs {rmse_ffbsi:.4f}   "
      f"time {t_tpf_par:.3f}s vs {t_ffbsi:.3f}s   "
      f"speedup {t_ffbsi/t_tpf_par:.1f}x")

# Plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

iters_range = jnp.arange(1, N_EM + 1)

# A relative error convergence
ax = axes[0, 0]
ax.semilogy(iters_range, hist_bpf['A_err'], 'gray', lw=1.5, label='Bootstrap PF')
ax.semilogy(iters_range, hist_tw_seq['A_err'], 'b', lw=1.5, label='Twisted (seq)')
ax.semilogy(iters_range, hist_tw_par['A_err'], 'r--', lw=1.5, label='Twisted (par)')
ax.set_xlabel('EM iteration')
ax.set_ylabel(r'$\|A - A^*\|_F / \|A^*\|_F$')
ax.set_title('Transition matrix recovery')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# sigma_q relative error convergence
ax = axes[0, 1]
ax.semilogy(iters_range, hist_bpf['sq_err'], 'gray', lw=1.5, label='Bootstrap PF')
ax.semilogy(iters_range, hist_tw_seq['sq_err'], 'b', lw=1.5, label='Twisted (seq)')
ax.semilogy(iters_range, hist_tw_par['sq_err'], 'r--', lw=1.5, label='Twisted (par)')
ax.set_xlabel('EM iteration')
ax.set_ylabel(r'$|\sigma_q - \sigma_q^*| / \sigma_q^*$')
ax.set_title('Process noise recovery')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Per-iteration time
ax = axes[1, 0]
ax.plot(iters_range, hist_bpf['time'], 'gray', lw=1.5, label='Bootstrap PF')
ax.plot(iters_range, hist_tw_seq['time'], 'b', lw=1.5, label='Twisted (seq)')
ax.plot(iters_range, hist_tw_par['time'], 'r--', lw=1.5, label='Twisted (par)')
ax.set_xlabel('EM iteration')
ax.set_ylabel('Time (s)')
ax.set_title('Per-iteration wall-clock time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# A matrix heatmaps
ax = axes[1, 1]
A_err_bpf = jnp.abs(A_bpf - A_TRUE)
A_err_tw  = jnp.abs(A_tw_par - A_TRUE)
combined = jnp.concatenate([A_TRUE, A_tw_par, A_err_tw], axis=1)
im = ax.imshow(combined, aspect='auto', cmap='RdBu_r',
               vmin=-float(jnp.abs(A_TRUE).max()),
               vmax=float(jnp.abs(A_TRUE).max()))
ax.set_title(f'A: true | recovered | |error|')
ax.set_xticks([D//2, D + D//2, 2*D + D//2])
ax.set_xticklabels(['True', 'Twisted PF', '|Error|'])
plt.colorbar(im, ax=ax)

plt.suptitle(
    f'Particle EM on Poisson LDS — D={D}, T={T}, N={N}, {N_EM} EM iters\n'
    f'BPF: {t_bpf_total:.1f}s | Twisted seq: {t_seq_total:.1f}s | '
    f'Twisted par: {t_par_total:.1f}s '
    f'({t_seq_total/t_par_total:.1f}x speedup)',
    fontsize=11)
plt.tight_layout()
plt.savefig('particle_em_poisson_lds.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved plot to particle_em_poisson_lds.png")