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

Compares six E-step implementations:
  1. Bootstrap PF              (sequential, resampling)
  2. Mean-field Gaussian VI    (sequential)
  3. EBIF Twisted PF           (sequential, no resampling)
  4. EBIF Twisted PF           (parallel via DEER)
  5. Amortized Twisted PF      (sequential, no resampling)
  6. Amortized Twisted PF      (parallel via DEER)
"""

import time
import os
from functools import partial
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

from src.particle_filter import TwistedParallelParticleFilter
from src.baselines import bootstrap_pf, ffbsi_smoother

# Model configuration
D = int(os.environ.get("PLDS_D", 5))              # latent state dimension
T = int(os.environ.get("PLDS_T", 2000))           # time series length
N = int(os.environ.get("PLDS_N", 256))            # particles
N_EM = int(os.environ.get("PLDS_N_EM", 50))       # EM iterations

# Amortized twist training configuration.  The learned twist is trained
# offline to match EBIF targets across plausible PLDS parameter values, then
# reused inside both sequential and DEER-parallel E-steps.
AMORTIZED_TWIST_SAMPLES = int(os.environ.get("PLDS_AMORTIZED_TWIST_SAMPLES", 128))
AMORTIZED_TWIST_STEPS = int(os.environ.get("PLDS_AMORTIZED_TWIST_STEPS", 800))
AMORTIZED_TWIST_LR = float(os.environ.get("PLDS_AMORTIZED_TWIST_LR", 3e-4))
H_GRU = int(os.environ.get("PLDS_TWIST_H_GRU", 64))
H_MLP = int(os.environ.get("PLDS_TWIST_H_MLP", 64))

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


# 
# Amortized full-covariance twist for PLDS
# 
LOWER_TRI_ROWS = jnp.array(np.tril_indices(D, k=-1)[0])
LOWER_TRI_COLS = jnp.array(np.tril_indices(D, k=-1)[1])
N_CHOL_OFFDIAG = D * (D - 1) // 2
D_TWIST_THETA = D * D + 1
D_TWIST_OUT = 2 * D + N_CHOL_OFFDIAG


def inverse_softplus(y):
    """Numerically stable inverse of softplus on positive scalars."""
    y = jnp.maximum(y, 1e-6)
    return y + jnp.log(-jnp.expm1(-y))


def plds_params_to_twist_theta(A, sigma_q):
    """Pack PLDS parameters into the conditioning vector for the twist net."""
    return jnp.concatenate([
        A.reshape(-1),
        inverse_softplus(sigma_q)[None],
    ])


def project_stable_A(A, max_radius=0.99):
    """Clip spectral radius without changing stable matrices."""
    radius = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    scale = jnp.minimum(1.0, max_radius / jnp.maximum(radius, 1e-6))
    return jnp.real(A * scale)


def sample_plds_twist_training_params(key, n_samples, A_anchor, sigma_anchor):
    """
    Draw plausible PLDS parameters for amortized twist training.

    The distribution mixes random stable dynamics with the EM initialisation,
    then jitters both A and sigma_q.  Training targets are EBIF twists on the
    fixed observed sequence for each sampled parameter.
    """
    k_A, k_mix, k_noise, k_sigma = jr.split(key, 4)
    A_keys = jr.split(k_A, n_samples)
    stable_As = jax.vmap(
        lambda k: make_stable_A(k, D, eig_range=(0.70, 0.98))
    )(A_keys)
    mix = jr.uniform(k_mix, (n_samples, 1, 1), minval=0.0, maxval=1.0)
    A_samples = mix * stable_As + (1.0 - mix) * A_anchor[None]
    A_samples = A_samples + 0.05 * jr.normal(k_noise, (n_samples, D, D))
    A_samples = jax.vmap(project_stable_A)(A_samples)

    log_sigma = jnp.log(jnp.maximum(sigma_anchor, 1e-3))
    log_sigma = log_sigma + jr.uniform(
        k_sigma, (n_samples,), minval=-0.8, maxval=0.8,
    )
    sigma_samples = jnp.clip(jnp.exp(log_sigma), 0.10, 0.80)

    # Always include the EM starting point exactly.
    A_samples = A_samples.at[0].set(project_stable_A(A_anchor))
    sigma_samples = sigma_samples.at[0].set(sigma_anchor)
    return A_samples, sigma_samples


def _twist_to_components(twist):
    offdiag_vec = twist['chol_offdiag'][:, LOWER_TRI_ROWS, LOWER_TRI_COLS]
    return twist['mean'], twist['chol_log_diag'], offdiag_vec


def _unpack_offdiag(offdiag_vec):
    T_ = offdiag_vec.shape[0]
    offdiag = jnp.zeros((T_, D, D), dtype=offdiag_vec.dtype)
    return offdiag.at[:, LOWER_TRI_ROWS, LOWER_TRI_COLS].set(offdiag_vec)


def init_amortized_twist_params(key):
    """Initialise a backward-GRU amortized full-covariance twist network."""
    keys = jr.split(key, 8)

    def _init_linear(key, d_in, d_out, scale=1.0):
        return {
            'W': jr.normal(key, (d_in, d_out)) * (scale / np.sqrt(d_in)),
            'b': jnp.zeros(d_out),
        }

    params = {}
    params['theta_enc1'] = _init_linear(keys[0], D_TWIST_THETA, H_MLP)
    params['theta_enc2'] = _init_linear(keys[1], H_MLP, H_GRU)

    d_gru_in = D + H_GRU
    params['gru_Wz'] = _init_linear(keys[2], d_gru_in + H_GRU, H_GRU)
    params['gru_Wr'] = _init_linear(keys[3], d_gru_in + H_GRU, H_GRU)
    params['gru_Wh'] = _init_linear(keys[4], d_gru_in + H_GRU, H_GRU)

    params['readout1'] = _init_linear(keys[5], 2 * H_GRU, H_MLP)
    params['readout2'] = _init_linear(keys[6], H_MLP, D_TWIST_OUT, scale=0.1)
    return params


def apply_amortized_twist_plds(net_params, theta, observations):
    """
    Predict full-covariance Gaussian twist parameters for a PLDS.

    The returned pytree matches ebif_poisson exactly:
      mean          : (T, D)
      chol_log_diag : (T, D)
      chol_offdiag  : (T, D, D), strict lower triangle used
    """
    obs_features = jnp.log1p(jnp.clip(observations, 0.0, 1e6))
    T_ = observations.shape[0]

    def linear(p, x):
        return x @ p['W'] + p['b']

    h_theta = jax.nn.relu(linear(net_params['theta_enc1'], theta))
    h_theta = jnp.tanh(linear(net_params['theta_enc2'], h_theta))

    def gru_step(h, obs_t):
        inp = jnp.concatenate([obs_t, h_theta])
        combined = jnp.concatenate([inp, h])
        z = jax.nn.sigmoid(linear(net_params['gru_Wz'], combined))
        r = jax.nn.sigmoid(linear(net_params['gru_Wr'], combined))
        h_tilde = jnp.tanh(
            linear(net_params['gru_Wh'], jnp.concatenate([inp, r * h]))
        )
        h_new = (1.0 - z) * h + z * h_tilde
        return h_new, h_new

    h0 = jnp.zeros(H_GRU)
    _, h_rev = jax.lax.scan(gru_step, h0, obs_features[::-1])
    h_seq = h_rev[::-1]

    h_theta_seq = jnp.broadcast_to(h_theta, (T_, H_GRU))
    readout_in = jnp.concatenate([h_seq, h_theta_seq], axis=1)
    hidden = jax.nn.relu(linear(net_params['readout1'], readout_in))
    out = linear(net_params['readout2'], hidden)

    mean = out[:, :D]
    chol_log_diag = jnp.clip(out[:, D:2 * D], -7.0, 4.0)
    offdiag_vec = jnp.clip(out[:, 2 * D:], -8.0, 8.0)
    return {
        'mean':          mean,
        'chol_log_diag': chol_log_diag,
        'chol_offdiag':  _unpack_offdiag(offdiag_vec),
    }


def train_amortized_twist_plds(
    key, ys, A_anchor, sigma_anchor, n_samples=128, n_steps=800, lr=3e-4,
):
    """Train the PLDS amortized twist by matching EBIF targets."""
    key, init_key, sample_key = jr.split(key, 3)
    net_params = init_amortized_twist_params(init_key)

    A_samples, sigma_samples = sample_plds_twist_training_params(
        sample_key, n_samples, A_anchor, sigma_anchor,
    )
    theta_samples = jax.vmap(plds_params_to_twist_theta)(
        A_samples, sigma_samples,
    )

    print("  Generating EBIF targets for amortized PLDS twist...")

    @jax.jit
    def make_targets(A_batch, sigma_batch):
        def one_target(A, sigma_q):
            ekf_means, _ = ekf_smoother_poisson(ys, A, sigma_q, x0_guess, P0)
            twist = ebif_poisson(ys, A, sigma_q, ekf_means)
            return _twist_to_components(twist)
        return jax.vmap(one_target)(A_batch, sigma_batch)

    target_means, target_log_diags, target_offdiags = make_targets(
        A_samples, sigma_samples,
    )
    target_means.block_until_ready()
    print(f"  Training data ready: {n_samples} parameter samples, T={ys.shape[0]}")

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(net_params)

    @jax.jit
    def mse_loss_fn(net_params, theta_batch, tgt_mean, tgt_log_diag, tgt_offdiag):
        def single_loss(theta, mean_t, log_diag_t, offdiag_t):
            pred = apply_amortized_twist_plds(net_params, theta, ys)
            pred_mean, pred_log_diag, pred_offdiag = _twist_to_components(pred)
            loss_mean = jnp.mean((pred_mean - mean_t) ** 2)
            loss_log_diag = jnp.mean((pred_log_diag - log_diag_t) ** 2)
            loss_offdiag = jnp.mean((pred_offdiag - offdiag_t) ** 2)
            return loss_mean + loss_log_diag + 0.5 * loss_offdiag
        losses = jax.vmap(single_loss)(
            theta_batch, tgt_mean, tgt_log_diag, tgt_offdiag,
        )
        return jnp.mean(losses)

    @jax.jit
    def train_step(net_params, opt_state, theta_batch, mean_b, log_diag_b, offdiag_b):
        loss, grads = jax.value_and_grad(mse_loss_fn)(
            net_params, theta_batch, mean_b, log_diag_b, offdiag_b,
        )
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, net_params)
        net_params = optax.apply_updates(net_params, updates)
        return net_params, opt_state, loss

    batch_size = min(32, n_samples)
    for step in range(n_steps):
        key, batch_key = jr.split(key)
        idx = jr.choice(batch_key, n_samples, (batch_size,), replace=False)
        net_params, opt_state, loss = train_step(
            net_params, opt_state, theta_samples[idx],
            target_means[idx], target_log_diags[idx], target_offdiags[idx],
        )
        if step % 100 == 0 or step == n_steps - 1:
            print(f"    Step {step:>5d}/{n_steps}: twist MSE = {float(loss):.6f}")

    return net_params


@jax.jit
def amortized_twist_rmse(net_params, A, sigma_q, observations):
    """RMSE of the amortized twist against the EBIF target at one theta."""
    ekf_means, _ = ekf_smoother_poisson(observations, A, sigma_q, x0_guess, P0)
    target = ebif_poisson(observations, A, sigma_q, ekf_means)
    theta = plds_params_to_twist_theta(A, sigma_q)
    pred = apply_amortized_twist_plds(net_params, theta, observations)
    tgt_m, tgt_ld, tgt_off = _twist_to_components(target)
    pred_m, pred_ld, pred_off = _twist_to_components(pred)
    return (
        jnp.sqrt(jnp.mean((pred_m - tgt_m) ** 2)),
        jnp.sqrt(jnp.mean((pred_ld - tgt_ld) ** 2)),
        jnp.sqrt(jnp.mean((pred_off - tgt_off) ** 2)),
    )


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

def run_em_twisted(
    A_init,
    sigma_q_init,
    parallel=False,
    max_deer=100,
    twist_mode="ebif",
    twist_net_params=None,
):
    """Run particle EM using either EBIF or amortized twisted PF for E-step."""
    use_amortized = twist_mode == "amortized"
    if twist_mode not in ("ebif", "amortized"):
        raise ValueError(f"Unknown twist_mode={twist_mode}")
    if use_amortized and twist_net_params is None:
        raise ValueError("twist_net_params is required for amortized twists")

    def make_twist_and_yinit(A_curr, sq_curr):
        if use_amortized:
            theta = plds_params_to_twist_theta(A_curr, sq_curr)
            twist = apply_amortized_twist_plds(twist_net_params, theta, ys)
            return twist, twist['mean']
        ekf_means, _ = ekf_smoother_poisson(ys, A_curr, sq_curr, x0_guess, P0)
        twist = ebif_poisson(ys, A_curr, sq_curr, ekf_means)
        return twist, ekf_means

    # JIT the whole E-step + M-step once — non-jit calls inside the loop
    # cause full re-trace per EM iter (~2s overhead per call for the
    # parallel PF; see debug_pls_pf_timing.py).
    @jax.jit
    def em_step_seq(A_curr, sq_curr):
        twist, _ = make_twist_and_yinit(A_curr, sq_curr)
        tape = (ys, pf_noise, twist)
        model_params = {'A': A_curr, 'sigma_q': sq_curr}
        states = tpf.run_sequential(x0s, tape, model_params)
        log_ws = tpf.compute_log_weights(x0s, states, tape, model_params)
        S_xp, S_pp, S_xx = compute_suff_stats_twisted(x0s, states, log_ws)
        A_new, sq_new = m_step(S_xp, S_pp, S_xx, T)
        return A_new, sq_new

    @partial(jax.jit, static_argnames=("max_iter",))
    def em_step_par(A_curr, sq_curr, key, max_iter):
        twist, yinit = make_twist_and_yinit(A_curr, sq_curr)
        tape = (ys, pf_noise, twist)
        model_params = {'A': A_curr, 'sigma_q': sq_curr}
        params_par = {**model_params, 'key': key}
        states, iters = tpf.run_parallel(
            x0s, tape, params_par, yinit_guess=yinit, max_iter=max_iter,
        )
        log_ws = tpf.compute_log_weights(x0s, states, tape, model_params)
        S_xp, S_pp, S_xx = compute_suff_stats_twisted(x0s, states, log_ws)
        A_new, sq_new = m_step(S_xp, S_pp, S_xx, T)
        return A_new, sq_new, iters

    A_curr = A_init
    sq_curr = sigma_q_init
    history = {'A_err': [], 'sq_err': [], 'time': [], 'deer_iters': []}

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
        if parallel:
            history['deer_iters'].append(int(iters[0]))

        if (it + 1) % 10 == 0 or it == 0:
            prefix = "am" if use_amortized else "ebif"
            mode = f"{prefix}-{'par' if parallel else 'seq'}"
            iters_str = f"  iters={int(iters[0])}" if parallel else ""
            print(f"  [{mode}] it={it+1:>3}  "
                  f"||A-A*||/||A*||={A_err:.4f}  "
                  f"|sq-sq*|/sq*={sq_err:.4f}  "
                  f"time={dt:.3f}s{iters_str}")

        A_curr = A_new
        sq_curr = sq_new

    return A_curr, sq_curr, history


#  EM with mean-field Gaussian VI
def run_em_variational(A_init, sigma_q_init, n_inner=100, lr=0.05):
    """
    Variational EM with a mean-field Gaussian posterior.

      q(x_{1:T}) = prod_t  N(x_t ; mu_t, diag(sigma_t^2))

    For PLDS the ELBO is closed-form:

      sum_t  [ y_t . mu_t  -  exp(mu_t + 1/2 sigma_t^2) ]            # E_q[Poisson]
        - 1/(2 sigma_q^2)  sum_t  E_q[ ||x_t - A x_{t-1}||^2 ]        # E_q[transition]
        + 1/2  sum_t  log(2 pi e sigma_t^2)                           # entropy

    where, for diagonal q,
      E_q[ ||x_t - A x_{t-1}||^2 ]
        = ||mu_t - A mu_{t-1}||^2  +  sum_d sigma_{t,d}^2
          +  sum_d sigma_{t-1,d}^2  sum_i A_{i,d}^2

    Mean-field implies Cov_q(x_t, x_{t-1}) = 0, so the cross-term in the
    sufficient statistics is the bare outer product of means -- this is
    the structural bias that costs M-step accuracy on coupled dynamics.
    """

    def neg_elbo(vparams, A, sigma_q):
        mu = vparams['mu']                     # (T, D)
        log_sigma = vparams['log_sigma']       # (T, D)
        sigma2 = jnp.exp(2.0 * log_sigma)

        # Expected Poisson log-likelihood (drop log(y!) constant)
        log_lik = jnp.sum(
            ys * mu - jnp.exp(jnp.clip(mu + 0.5 * sigma2, -10.0, 10.0))
        )

        # Expected Gaussian transition log-likelihood
        prec_q = 1.0 / sigma_q**2
        col_sq = jnp.sum(A**2, axis=0)         # (D,)  sum_i A_{i,d}^2

        diff_1 = mu[0] - A @ x0_guess
        sq_1 = jnp.sum(diff_1**2) + jnp.sum(sigma2[0])

        diff = mu[1:] - mu[:-1] @ A.T          # (T-1, D)
        sq_means = jnp.sum(diff**2)
        var_curr = jnp.sum(sigma2[1:])
        var_prev = jnp.sum(sigma2[:-1] * col_sq[None, :])

        log_trans = -0.5 * prec_q * (sq_1 + sq_means + var_curr + var_prev)
        log_trans -= 0.5 * D * T * jnp.log(2.0 * jnp.pi * sigma_q**2)

        # Entropy (per-step + sum)
        entropy = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * sigma2)) + 0.5 * T * D

        return -(log_lik + log_trans + entropy)

    optimizer = optax.adam(lr)

    @jax.jit
    def vi_inner_loop(vparams, opt_state, A, sigma_q):
        def step(carry, _):
            vp, ost = carry
            loss, grads = jax.value_and_grad(neg_elbo)(vp, A, sigma_q)
            updates, ost = optimizer.update(grads, ost, vp)
            vp = optax.apply_updates(vp, updates)
            return (vp, ost), loss
        (vparams, opt_state), losses = jax.lax.scan(
            step, (vparams, opt_state), None, length=n_inner
        )
        return vparams, opt_state, losses

    @jax.jit
    def vi_suff_stats(vparams):
        mu = vparams['mu']                                 # (T, D)
        sigma2 = jnp.exp(2.0 * vparams['log_sigma'])       # (T, D)
        full_mu = jnp.concatenate([x0_guess[None], mu], axis=0)
        full_s2 = jnp.concatenate([jnp.zeros((1, D)), sigma2], axis=0)
        x_prev_mu, x_curr_mu = full_mu[:-1], full_mu[1:]
        x_prev_s2, x_curr_s2 = full_s2[:-1], full_s2[1:]
        # Cov_q(x_t, x_{t-1}) = 0 under mean-field
        S_xp = x_curr_mu.T @ x_prev_mu
        S_pp = x_prev_mu.T @ x_prev_mu + jnp.diag(jnp.sum(x_prev_s2, axis=0))
        S_xx = x_curr_mu.T @ x_curr_mu + jnp.diag(jnp.sum(x_curr_s2, axis=0))
        return S_xp, S_pp, S_xx

    @jax.jit
    def em_step_vi(A_curr, sq_curr, vparams, opt_state):
        vparams, opt_state, losses = vi_inner_loop(vparams, opt_state, A_curr, sq_curr)
        S_xp, S_pp, S_xx = vi_suff_stats(vparams)
        A_new, sq_new = m_step(S_xp, S_pp, S_xx, T)
        return A_new, sq_new, vparams, opt_state, losses

    A_curr = A_init
    sq_curr = sigma_q_init
    history = {'A_err': [], 'sq_err': [], 'time': [], 'final_neg_elbo': []}

    # One-time warm-start: EKF mean + diagonal as q init
    ekf_m_init, ekf_P_init = ekf_smoother_poisson(ys, A_curr, sq_curr, x0_guess, P0)
    init_mu = ekf_m_init
    diag_var = jnp.diagonal(ekf_P_init, axis1=1, axis2=2)
    init_log_sigma = 0.5 * jnp.log(jnp.clip(diag_var, 1e-6, None))
    vparams = {'mu': init_mu, 'log_sigma': init_log_sigma}
    opt_state = optimizer.init(vparams)

    # Warmup compile (timing excludes this)
    _ = em_step_vi(A_curr, sq_curr, vparams, opt_state)
    jax.block_until_ready(_[0])

    for it in range(N_EM):
        t0 = time.perf_counter()
        A_new, sq_new, vparams, opt_state, losses = em_step_vi(
            A_curr, sq_curr, vparams, opt_state
        )
        jax.block_until_ready(A_new)
        dt = time.perf_counter() - t0

        A_err = float(jnp.linalg.norm(A_new - A_TRUE, 'fro') /
                      jnp.linalg.norm(A_TRUE, 'fro'))
        sq_err = float(jnp.abs(sq_new - SIGMA_Q_TRUE) / SIGMA_Q_TRUE)
        history['A_err'].append(A_err)
        history['sq_err'].append(sq_err)
        history['time'].append(dt)
        history['final_neg_elbo'].append(float(losses[-1]))

        if (it + 1) % 10 == 0 or it == 0:
            print(f"  [vi]  it={it+1:>3}  "
                  f"||A-A*||/||A*||={A_err:.4f}  "
                  f"|sq-sq*|/sq*={sq_err:.4f}  "
                  f"time={dt:.3f}s  neg-ELBO={float(losses[-1]):.1f}")

        A_curr = A_new
        sq_curr = sq_new

    return A_curr, sq_curr, history, vparams


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


#  Train amortized twist, then run all six EM variants
print("\n--- Training amortized PLDS twist network ---")
key, amortized_key = jr.split(key)
t0_twist_train = time.perf_counter()
trained_twist_params = train_amortized_twist_plds(
    amortized_key,
    ys,
    A_init,
    sigma_q_init,
    n_samples=AMORTIZED_TWIST_SAMPLES,
    n_steps=AMORTIZED_TWIST_STEPS,
    lr=AMORTIZED_TWIST_LR,
)
t_twist_train = time.perf_counter() - t0_twist_train
rmse_m_true, rmse_ld_true, rmse_off_true = amortized_twist_rmse(
    trained_twist_params, A_TRUE, SIGMA_Q_TRUE, ys,
)
rmse_m_init, rmse_ld_init, rmse_off_init = amortized_twist_rmse(
    trained_twist_params, A_init, sigma_q_init, ys,
)
print(f"  Training time: {t_twist_train:.1f}s")
print("  Twist verification vs EBIF:")
print(f"    init theta: RMSE(mean)={float(rmse_m_init):.4f}, "
      f"RMSE(logdiag)={float(rmse_ld_init):.4f}, "
      f"RMSE(offdiag)={float(rmse_off_init):.4f}")
print(f"    true theta: RMSE(mean)={float(rmse_m_true):.4f}, "
      f"RMSE(logdiag)={float(rmse_ld_true):.4f}, "
      f"RMSE(offdiag)={float(rmse_off_true):.4f}")

print("\n--- Bootstrap PF EM ---")
A_bpf, sq_bpf, hist_bpf = run_em_bootstrap(A_init, sigma_q_init)

print("\n--- Mean-field VI EM ---")
A_vi, sq_vi, hist_vi, vparams_vi = run_em_variational(A_init, sigma_q_init)

print("\n--- Twisted PF EM (sequential) ---")
A_tw_seq, sq_tw_seq, hist_tw_seq = run_em_twisted(A_init, sigma_q_init, parallel=False)

print("\n--- Twisted PF EM (parallel / DEER) ---")
A_tw_par, sq_tw_par, hist_tw_par = run_em_twisted(A_init, sigma_q_init, parallel=True)

print("\n--- Amortized Twisted PF EM (sequential) ---")
A_am_seq, sq_am_seq, hist_am_seq = run_em_twisted(
    A_init,
    sigma_q_init,
    parallel=False,
    twist_mode="amortized",
    twist_net_params=trained_twist_params,
)

print("\n--- Amortized Twisted PF EM (parallel / DEER) ---")
A_am_par, sq_am_par, hist_am_par = run_em_twisted(
    A_init,
    sigma_q_init,
    parallel=True,
    twist_mode="amortized",
    twist_net_params=trained_twist_params,
)


# Summary
print(f"\n{'='*70}")
print(f"Summary — Poisson LDS Particle EM, D={D}, T={T}, N={N}, {N_EM} iters")
print(f"{'='*70}")

t_bpf_total = sum(hist_bpf['time'])
t_vi_total  = sum(hist_vi['time'])
t_seq_total = sum(hist_tw_seq['time'])
t_par_total = sum(hist_tw_par['time'])
t_am_seq_total = sum(hist_am_seq['time'])
t_am_par_total = sum(hist_am_par['time'])

print(f"\n{'Method':<28} {'Total time':>10} {'Final ||A-A*||':>14} {'Final |sq-sq*|':>14}")
print("-" * 70)
print(f"{'Bootstrap PF EM':<28} {t_bpf_total:>9.1f}s {hist_bpf['A_err'][-1]:>14.4f} "
      f"{hist_bpf['sq_err'][-1]:>14.4f}")
print(f"{'Mean-field VI EM':<28} {t_vi_total:>9.1f}s {hist_vi['A_err'][-1]:>14.4f} "
      f"{hist_vi['sq_err'][-1]:>14.4f}")
print(f"{'Twisted PF EM (seq)':<28} {t_seq_total:>9.1f}s {hist_tw_seq['A_err'][-1]:>14.4f} "
      f"{hist_tw_seq['sq_err'][-1]:>14.4f}")
print(f"{'Twisted PF EM (par)':<28} {t_par_total:>9.1f}s {hist_tw_par['A_err'][-1]:>14.4f} "
      f"{hist_tw_par['sq_err'][-1]:>14.4f}")
print(f"{'Amortized Twisted (seq)':<28} {t_am_seq_total:>9.1f}s {hist_am_seq['A_err'][-1]:>14.4f} "
      f"{hist_am_seq['sq_err'][-1]:>14.4f}")
print(f"{'Amortized Twisted (par)':<28} {t_am_par_total:>9.1f}s {hist_am_par['A_err'][-1]:>14.4f} "
      f"{hist_am_par['sq_err'][-1]:>14.4f}")
print(f"\nSpeedup (seq/par): {t_seq_total/t_par_total:.1f}x")
print(f"Speedup (am seq/am par): {t_am_seq_total/t_am_par_total:.1f}x")
print(f"Speedup (bpf/par): {t_bpf_total/t_par_total:.1f}x")
print(f"Speedup (vi/par):  {t_vi_total/t_par_total:.1f}x")

print(f"\n{'':>15} {'True':>8} {'BPF':>8} {'VI':>8} {'Tw-seq':>8} {'Tw-par':>8}"
      f" {'Am-seq':>8} {'Am-par':>8}")
print("-" * 78)
print(f"{'sigma_q':>15} {SIGMA_Q_TRUE:>8.4f} {float(sq_bpf):>8.4f} "
      f"{float(sq_vi):>8.4f} "
      f"{float(sq_tw_seq):>8.4f} {float(sq_tw_par):>8.4f} "
      f"{float(sq_am_seq):>8.4f} {float(sq_am_par):>8.4f}")
print(f"{'||A||_F':>15} {float(jnp.linalg.norm(A_TRUE, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_bpf, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_vi, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_tw_seq, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_tw_par, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_am_seq, 'fro')):>8.4f} "
      f"{float(jnp.linalg.norm(A_am_par, 'fro')):>8.4f}")
print(f"{'spec_radius(A)':>15} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_TRUE)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_bpf)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_vi)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_tw_seq)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_tw_par)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_am_seq)))):>8.4f} "
      f"{float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_am_par)))):>8.4f}")

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

# Amortized twist at true params, using the same sequential and DEER paths.
theta_true_twist = plds_params_to_twist_theta(A_TRUE, SIGMA_Q_TRUE)
twist_am_true = apply_amortized_twist_plds(trained_twist_params, theta_true_twist, ys)
tape_am_true = (ys, pf_noise, twist_am_true)

_ = run_seq_true(x0s, tape_am_true, true_params)
t0 = time.perf_counter()
states_am_seq_true = run_seq_true(x0s, tape_am_true, true_params).block_until_ready()
t_am_tpf_seq = time.perf_counter() - t0
log_ws_am_seq = jax.jit(tpf.compute_log_weights)(
    x0s, states_am_seq_true, tape_am_true, true_params,
)
w_am_seq = jax.nn.softmax(jnp.sum(log_ws_am_seq, axis=1))
am_seq_means = jnp.einsum('n,ntd->td', w_am_seq, states_am_seq_true)
rmse_am_tpf_seq = float(jnp.sqrt(jnp.mean((am_seq_means - true_xs)**2)))

key, deer_key = jr.split(key)
params_par_true = {**true_params, 'key': deer_key}
_ = run_par_true(
    x0s,
    tape_am_true,
    params_par_true,
    yinit_guess=twist_am_true['mean'],
    max_iter=100,
)
t0 = time.perf_counter()
states_am_par_true, deer_iters_am = run_par_true(
    x0s,
    tape_am_true,
    params_par_true,
    yinit_guess=twist_am_true['mean'],
    max_iter=100,
)
states_am_par_true.block_until_ready()
t_am_tpf_par = time.perf_counter() - t0
log_ws_am_par = jax.jit(tpf.compute_log_weights)(
    x0s, states_am_par_true, tape_am_true, true_params,
)
w_am_par = jax.nn.softmax(jnp.sum(log_ws_am_par, axis=1))
am_par_means = jnp.einsum('n,ntd->td', w_am_par, states_am_par_true)
rmse_am_tpf_par = float(jnp.sqrt(jnp.mean((am_par_means - true_xs)**2)))

print(f"Amortized PF (seq):     time={t_am_tpf_seq:.3f}s   "
      f"RMSE vs true x={rmse_am_tpf_seq:.4f}")
print(f"Amortized PF (parallel): time={t_am_tpf_par:.3f}s   "
      f"RMSE vs true x={rmse_am_tpf_par:.4f}   "
      f"({int(deer_iters_am[0])} DEER iters)")

# Mean-field VI smoother at true params: optimize ELBO from EKF init for
# the same n_inner steps used in EM, then read off mu_t.
def vi_smooth_rmse(A, sigma_q, n_inner=100, lr=0.05):
    optimizer = optax.adam(lr)

    def neg_elbo(vp):
        mu = vp['mu']
        s2 = jnp.exp(2.0 * vp['log_sigma'])
        log_lik = jnp.sum(ys * mu - jnp.exp(jnp.clip(mu + 0.5 * s2, -10.0, 10.0)))
        prec_q = 1.0 / sigma_q**2
        col_sq = jnp.sum(A**2, axis=0)
        diff_1 = mu[0] - A @ x0_guess
        sq_1 = jnp.sum(diff_1**2) + jnp.sum(s2[0])
        diff = mu[1:] - mu[:-1] @ A.T
        sq_means = jnp.sum(diff**2)
        var_curr = jnp.sum(s2[1:])
        var_prev = jnp.sum(s2[:-1] * col_sq[None, :])
        log_trans = -0.5 * prec_q * (sq_1 + sq_means + var_curr + var_prev)
        log_trans -= 0.5 * D * T * jnp.log(2.0 * jnp.pi * sigma_q**2)
        entropy = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * s2)) + 0.5 * T * D
        return -(log_lik + log_trans + entropy)

    em_init, ekf_P = ekf_smoother_poisson(ys, A, sigma_q, x0_guess, P0)
    diag_var = jnp.diagonal(ekf_P, axis1=1, axis2=2)
    vp = {'mu': em_init,
          'log_sigma': 0.5 * jnp.log(jnp.clip(diag_var, 1e-6, None))}
    ost = optimizer.init(vp)

    @jax.jit
    def loop(vp, ost):
        def step(c, _):
            vp, ost = c
            loss, g = jax.value_and_grad(neg_elbo)(vp)
            upd, ost = optimizer.update(g, ost, vp)
            vp = optax.apply_updates(vp, upd)
            return (vp, ost), loss
        (vp, ost), losses = jax.lax.scan(step, (vp, ost), None, length=n_inner)
        return vp, losses

    _ = loop(vp, ost)                         # warmup
    jax.block_until_ready(_[0]['mu'])
    t0 = time.perf_counter()
    vp_final, losses = loop(vp, ost)
    jax.block_until_ready(vp_final['mu'])
    dt = time.perf_counter() - t0
    rmse = float(jnp.sqrt(jnp.mean((vp_final['mu'] - true_xs)**2)))
    return dt, rmse, float(losses[-1])

t_vi_smooth, rmse_vi_smooth, neg_elbo_final = vi_smooth_rmse(A_TRUE, SIGMA_Q_TRUE)
print(f"Mean-field VI:          time={t_vi_smooth:.3f}s   "
      f"RMSE vs true x={rmse_vi_smooth:.4f}   neg-ELBO={neg_elbo_final:.1f}")

print(f"\nTwisted par vs FFBSi:   RMSE {rmse_tpf_par:.4f} vs {rmse_ffbsi:.4f}   "
      f"time {t_tpf_par:.3f}s vs {t_ffbsi:.3f}s   "
      f"speedup {t_ffbsi/t_tpf_par:.1f}x")
print(f"Amortized par vs FFBSi: RMSE {rmse_am_tpf_par:.4f} vs {rmse_ffbsi:.4f}   "
      f"time {t_am_tpf_par:.3f}s vs {t_ffbsi:.3f}s   "
      f"speedup {t_ffbsi/t_am_tpf_par:.1f}x")
print(f"Twisted par vs VI:      RMSE {rmse_tpf_par:.4f} vs {rmse_vi_smooth:.4f}   "
      f"time {t_tpf_par:.3f}s vs {t_vi_smooth:.3f}s")
print(f"Amortized par vs VI:    RMSE {rmse_am_tpf_par:.4f} vs {rmse_vi_smooth:.4f}   "
      f"time {t_am_tpf_par:.3f}s vs {t_vi_smooth:.3f}s")

# Plot
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "figures"))
import paper_style as ps  # noqa: E402
import numpy as _np
ps.apply()

iters_range = _np.arange(1, N_EM + 1)

# Cache numerical results so the figure can be regenerated without re-running
# the whole EM sweep.  Suffix tracks T so T=2000 and T=8000 don't collide.
_cache_path = f"examples/figures/_plds_cache_T{T}.npz"
_np.savez(
    _cache_path,
    iters=iters_range,
    bpf_A=_np.asarray(hist_bpf['A_err']),    bpf_sq=_np.asarray(hist_bpf['sq_err']),    bpf_t=_np.asarray(hist_bpf['time']),
    vi_A=_np.asarray(hist_vi['A_err']),      vi_sq=_np.asarray(hist_vi['sq_err']),      vi_t=_np.asarray(hist_vi['time']),
    tw_seq_A=_np.asarray(hist_tw_seq['A_err']), tw_seq_sq=_np.asarray(hist_tw_seq['sq_err']), tw_seq_t=_np.asarray(hist_tw_seq['time']),
    tw_par_A=_np.asarray(hist_tw_par['A_err']), tw_par_sq=_np.asarray(hist_tw_par['sq_err']), tw_par_t=_np.asarray(hist_tw_par['time']),
    am_seq_A=_np.asarray(hist_am_seq['A_err']), am_seq_sq=_np.asarray(hist_am_seq['sq_err']), am_seq_t=_np.asarray(hist_am_seq['time']),
    am_par_A=_np.asarray(hist_am_par['A_err']), am_par_sq=_np.asarray(hist_am_par['sq_err']), am_par_t=_np.asarray(hist_am_par['time']),
    totals=_np.array([t_bpf_total, t_vi_total, t_seq_total, t_par_total,
                      t_am_seq_total, t_am_par_total]),
    T=T, D=D, N=N, N_EM=N_EM,
)
print(f"  cached EM histories to {_cache_path}")

# ---------- Figure ----------
# After the run, the figure is also produced offline by
#   examples/figures/replot_plds.py
# from the cached npz; that script is the canonical path for paper builds.
fig, axes = plt.subplots(2, 2, figsize=(ps.COL_FULL, 5.3),
                         constrained_layout=True)

# Color/style assignments — VI gets a distinctive treatment because it is the
# headline ("the fast Gaussian baseline that fails to converge").
STYLE = dict(
    bpf    = dict(color=ps.PALETTE['bpf'],         ls='-',  lw=1.1, label='Bootstrap PF'),
    vi     = dict(color=ps.PALETTE['ffbsi'],       ls='-',  lw=1.6, label='Mean-field VI'),
    tw_seq = dict(color=ps.PALETTE['twisted_seq'], ls='-',  lw=1.1, label='Twisted (seq)'),
    tw_par = dict(color=ps.PALETTE['par'],         ls='--', lw=1.3, label='Twisted (par)'),
    am_seq = dict(color=ps.PALETTE['par_alt'],     ls='-',  lw=1.1, label='Amortized (seq)'),
    am_par = dict(color=ps.PALETTE['par_alt'],     ls='--', lw=1.3, label='Amortized (par)'),
)

def _plot(ax, key, ydata):
    ax.plot(iters_range, _np.asarray(ydata), **STYLE[key])

# (a) Transition-matrix recovery — VI is plotted last so it sits on top
ax = axes[0, 0]
_plot(ax, 'bpf',    hist_bpf['A_err'])
_plot(ax, 'tw_seq', hist_tw_seq['A_err'])
_plot(ax, 'tw_par', hist_tw_par['A_err'])
_plot(ax, 'am_seq', hist_am_seq['A_err'])
_plot(ax, 'am_par', hist_am_par['A_err'])
_plot(ax, 'vi',     hist_vi['A_err'])
ax.set_yscale('log')
ax.set_xlabel('EM iteration')
ax.set_ylabel(r'$\|A - A^\star\|_F / \|A^\star\|_F$')
ax.set_title(r'Transition matrix recovery')
# Annotate the VI plateau (the headline failure of mean-field on this model)
vi_final = float(_np.asarray(hist_vi['A_err'])[-1])
ax.annotate(
    'Mean-field VI\nplateaus',
    xy=(N_EM * 0.85, vi_final),
    xytext=(0.50, 0.62), textcoords='axes fraction',
    fontsize=7, color=ps.PALETTE['ffbsi'],
    arrowprops=dict(arrowstyle='->', color=ps.PALETTE['ffbsi'],
                    lw=0.7, alpha=0.85, shrinkA=2, shrinkB=2),
    ha='left', va='bottom',
    bbox=dict(boxstyle='round,pad=0.25', fc='white',
              ec=ps.PALETTE['ffbsi'], lw=0.5, alpha=0.92),
)

# (b) Process-noise recovery + the legend (panel a is annotated, this is clean)
ax = axes[0, 1]
_plot(ax, 'bpf',    hist_bpf['sq_err'])
_plot(ax, 'tw_seq', hist_tw_seq['sq_err'])
_plot(ax, 'tw_par', hist_tw_par['sq_err'])
_plot(ax, 'am_seq', hist_am_seq['sq_err'])
_plot(ax, 'am_par', hist_am_par['sq_err'])
_plot(ax, 'vi',     hist_vi['sq_err'])
ax.set_yscale('log')
ax.set_xlabel('EM iteration')
ax.set_ylabel(r'$|\sigma_q - \sigma_q^\star| / \sigma_q^\star$')
ax.set_title(r'Process-noise recovery')
ax.legend(loc='lower left', fontsize=6.5, ncol=2,
          columnspacing=0.8, handlelength=1.5, framealpha=0.9)

# (c) Cumulative wall-clock — much more legible than per-iter
ax = axes[1, 0]
def _cumtime(h, key):
    return _np.cumsum(_np.asarray(h['time']))
_styles = [(hist_bpf, 'bpf'), (hist_vi, 'vi'),
           (hist_tw_seq, 'tw_seq'), (hist_tw_par, 'tw_par'),
           (hist_am_seq, 'am_seq'), (hist_am_par, 'am_par')]
finals = []
for h, key in _styles:
    cum = _cumtime(h, key)
    ax.plot(iters_range, cum, **STYLE[key])
    finals.append((STYLE[key]['label'], float(cum[-1]), STYLE[key]['color']))
ax.set_xlabel('EM iteration')
ax.set_ylabel('Cumulative wall-clock (s)')
ax.set_title('Total wall-clock vs. EM iteration')
ax.set_yscale('log')

# (d) End-to-end time bars + speedup callout
ax = axes[1, 1]
labels = ['BPF', 'VI', 'Tw\nseq', 'Tw\npar', 'Am\nseq', 'Am\npar']
totals = [t_bpf_total, t_vi_total, t_seq_total, t_par_total,
          t_am_seq_total, t_am_par_total]
colors = [STYLE[k]['color'] for k in
          ['bpf', 'vi', 'tw_seq', 'tw_par', 'am_seq', 'am_par']]
xpos = _np.arange(len(labels))
ax.bar(xpos, totals, color=colors, edgecolor='k', lw=0.5,
       alpha=0.88, width=0.65)
ax.set_yscale('log')
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('Total EM time (s)')
ax.set_title('End-to-end EM cost ($\\downarrow$ better)')
ymax = max(totals) * 6.0
ax.set_ylim(min(totals) * 0.55, ymax)
for xi, v in zip(xpos, totals):
    ax.text(xi, v * 1.18, f'{v:.1f} s', ha='center', va='bottom',
            fontsize=6.5, color='0.15')

# Annotate the headline speedup: amortized par vs twisted seq
sp_am_par_vs_seq = t_seq_total / t_am_par_total if t_am_par_total > 0 else 0.0
ax.annotate(
    f'Amortized par:\n$\\mathbf{{{sp_am_par_vs_seq:.1f}\\times}}$ vs. Tw seq',
    xy=(5, t_am_par_total),
    xytext=(0.04, 0.97), textcoords='axes fraction',
    fontsize=7, color=ps.PALETTE['par_alt'],
    arrowprops=dict(arrowstyle='->', color=ps.PALETTE['par_alt'],
                    lw=0.7, alpha=0.85, shrinkA=2, shrinkB=2),
    ha='left', va='top',
    bbox=dict(boxstyle='round,pad=0.22', fc='white',
              ec=ps.PALETTE['par_alt'], lw=0.5, alpha=0.92),
)

fig.suptitle(
    f'Particle EM on Poisson LDS  '
    f'($D{{=}}{D}$, $T{{=}}{T}$, $N{{=}}{N}$, {N_EM} EM iters)',
    fontsize=10, y=1.03,
)

plt.savefig(f'particle_em_poisson_lds{"" if T == 2000 else f"_T{T}"}.png')
plt.close(fig)
print(f"\nSaved plot to particle_em_poisson_lds{'' if T == 2000 else f'_T{T}'}.png")
