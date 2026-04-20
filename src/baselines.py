"""
baselines.py

Classical particle-smoothing baselines — Bootstrap PF and FFBSi.

Gaussian-transition models with arbitrary observation likelihoods.
Assumes x_{t+1} | x_t ~ N(f(x_t), diag(sigma_q^2)) with user-supplied
deterministic mean f and log-likelihood log p(y_t | x_t).
"""
from typing import Callable, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jr


def _systematic_resample(key: jnp.ndarray,
                         log_w: jnp.ndarray,
                         particles: jnp.ndarray) -> jnp.ndarray:
    w = jax.nn.softmax(log_w)
    n = w.shape[0]
    u = (jnp.arange(n) + jr.uniform(key)) / n
    cumw = jnp.cumsum(w)
    idx = jnp.clip(jnp.searchsorted(cumw, u), 0, n - 1)
    return particles[idx]


def bootstrap_pf(
    ys: jnp.ndarray,
    x0_particles: jnp.ndarray,
    transition_mean: Callable[[jnp.ndarray, Any], jnp.ndarray],
    log_lik: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    sigma_q: Any,
    params: Any,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Bootstrap particle filter with Gaussian transitions.

    Arguments
    ---------
    ys              : (T, dy)      observations
    x0_particles    : (N, D)       initial particles (already spread/drawn)
    transition_mean : fn(x_prev (D,), params) -> (D,)  deterministic mean
    log_lik         : fn(y_t (dy,), x_t (D,), params) -> scalar log p(y|x)
    sigma_q         : scalar or (D,)                   transition std
    params          : pytree
    key             : PRNGKey

    Returns
    -------
    filt_means : (T, D)   weighted filter means
    all_parts  : (T, N, D) pre-resampling particles
    all_log_w  : (T, N)    normalised pre-resampling log-weights
    ess        : (T,)      effective sample size per step
    log_ml     : scalar    log marginal likelihood estimate
    """
    sigma_q_b = jnp.asarray(sigma_q)

    v_trans = jax.vmap(transition_mean, in_axes=(0, None))
    v_lik   = jax.vmap(log_lik,         in_axes=(None, 0, None))

    N = x0_particles.shape[0]

    def step(carry, yt):
        particles, key, log_ml_acc = carry
        key, trans_key, resamp_key = jr.split(key, 3)

        preds = v_trans(particles, params)                    
        noise = jr.normal(trans_key, particles.shape)
        particles_new = preds + sigma_q_b * noise

        log_w = v_lik(yt, particles_new, params)             
        log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)
        w_soft = jnp.exp(log_w_norm)
        filt_mean = jnp.sum(w_soft[:, None] * particles_new, axis=0)
        ess = 1.0 / jnp.sum(w_soft**2)
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))

        particles_resamp = _systematic_resample(resamp_key, log_w, particles_new)

        return ((particles_resamp, key, log_ml_acc + log_ml_inc),
                (particles_new, log_w_norm, filt_mean, ess))

    (_, _, log_ml), (all_parts, all_log_w, filt_means, ess) = jax.lax.scan(
        step, (x0_particles, key, 0.0), ys)
    return filt_means, all_parts, all_log_w, ess, log_ml


def ffbsi_smoother(
    all_parts: jnp.ndarray,
    all_log_w: jnp.ndarray,
    transition_mean: Callable[[jnp.ndarray, Any], jnp.ndarray],
    sigma_q: Any,
    params: Any,
    key: jnp.ndarray,
    M: int,
) -> jnp.ndarray:
    """
    Particle smoother baseline.

    For each of M backward trajectories, walk from t=T to t=0 sampling
    ancestors with probability ∝ w^(f)_i · p(x_{t+1} | x_i) where the
    transition is Gaussian with mean f(x_i) and std sigma_q.

    Returns
    -------
    (M, T, D) array of smoothed trajectories.
    """
    T_, N, D = all_parts.shape
    sigma_q_b = jnp.asarray(sigma_q)
    v_trans = jax.vmap(transition_mean, in_axes=(0, None))

    def backward_one(m_key):
        m_key, init_key = jr.split(m_key)
        idx_T = jr.categorical(init_key, all_log_w[-1])
        x_T = all_parts[-1, idx_T]

        def bwd_step(carry, inputs):
            x_next, key = carry
            parts_t, log_w_t, step_key = inputs
            preds = v_trans(parts_t, params)                
            diff = (x_next - preds) / sigma_q_b
            log_trans = -0.5 * jnp.sum(diff**2, axis=-1)
            log_smooth = log_w_t + log_trans
            idx = jr.categorical(step_key, log_smooth)
            x_t = parts_t[idx]
            key, _ = jr.split(key)
            return (x_t, key), x_t

        step_keys = jr.split(m_key, T_ - 1)
        (_, _), path_rev = jax.lax.scan(
            bwd_step, (x_T, m_key),
            (all_parts[:-1][::-1], all_log_w[:-1][::-1], step_keys),
        )
        path = jnp.concatenate([path_rev[::-1], x_T[None]], axis=0)
        return path

    keys = jr.split(key, M)
    return jax.vmap(backward_one)(keys)
