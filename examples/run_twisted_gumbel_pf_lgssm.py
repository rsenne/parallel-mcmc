"""
run_twisted_gumbel_pf_lgssm.py

Prototype: twisted PF + windowed Gumbel-STE resampling, parallelised
per-window via per-particle DEER.

Hypothesis (from logs/gumbel_pf_scaling.log):
  Full per-step Gumbel-STE PF gives correct forward law but iters scale
  linearly with T because ancestor flips serialise. Restricting
  resampling to window boundaries every K steps should keep each
  intra-window chain smooth (DEER-friendly) and confine the dense
  cross-particle coupling to a small number of checkpoint events.

This script:
  1. Simulates a 1-D LGSSM and computes the RTS-optimal Gaussian twist.
  2. Runs sequential and DEER-parallelised twisted Gumbel PF on the
     SAME tape (transition noise + Gumbel noise).
  3. Reports filter-mean RMSE vs Kalman, log-Z agreement,
     max |par - seq|, and wall-clock.
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src.gumbel_particle_filter import TwistedGumbelParticleFilter


PHI, SIGMA_Q, SIGMA_R = 0.9, 0.5, 1.0
D = 1
N = 32
T = 1024
K = 32                  # window size; T must be divisible by K
TAU = 1.0
MAX_ITER = 200


def kalman_filter(ys, phi, sigma_q, sigma_r):
    def step(carry, y):
        m, P = carry
        mp = phi * m
        Pp = phi**2 * P + sigma_q**2
        S  = Pp + sigma_r**2
        Kg = Pp / S
        mf = mp + Kg * (y - mp)
        Pf = (1 - Kg) * Pp
        log_lik = -0.5 * (jnp.log(2 * jnp.pi * S) + (y - mp)**2 / S)
        return (mf, Pf), (mf, Pf, log_lik)
    _, (means, variances, log_liks) = jax.lax.scan(step, (0.0, 1.0), ys)
    return means, variances, jnp.sum(log_liks)


def backward_information_filter_1d(ys, phi, sigma_q, sigma_r):
    """Compute RTS-optimal twist (Gaussian) for 1-D LGSSM. Copied from
    examples/run_pf_comparison.py — see there for derivation."""
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2

    def step(carry, y):
        J, h = carry
        denom = 1.0 + sq * J
        J_pred = phi**2 * J / denom
        h_pred = phi * h / denom
        J_new = J_pred + prec_obs
        h_new = h_pred + y * prec_obs
        return (J_new, h_new), (J_new, h_new)

    J_T = prec_obs
    h_T = ys[-1] * prec_obs
    _, (Js_rev, hs_rev) = jax.lax.scan(step, (J_T, h_T), ys[-2::-1])
    Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_T])])
    hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_T])])
    return hs / Js, 1.0 / Js


def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi = params["phi"]; sq = params["sigma_q"]
    mu = twist["mean"]; tau = jnp.exp(twist["log_scale"])
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    phi = params["phi"]; sq = params["sigma_q"]; sr = params["sigma_r"]
    mu = twist["mean"]; tau = jnp.exp(twist["log_scale"])
    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sr**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sr**2))
    log_trans = (-0.5 * jnp.sum((x_next - phi * x_prev)**2) / sq**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sq**2))
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    log_prop  = (-0.5 * jnp.sum((x_next - mp)**2 / vp)
                 - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * vp)))
    return log_lik + log_trans - log_prop


def main():
    print(f"Device: {jax.devices()}")
    print(f"T={T}  N={N}  D={D}  K={K}  num_windows={T//K}  tau={TAU}")

    key = jr.PRNGKey(0)
    key, k1, k2, k3 = jr.split(key, 4)

    x0 = jr.normal(k1)

    def sim_step(x, qr):
        xn = PHI * x + SIGMA_Q * qr[0]
        return xn, (xn, xn + SIGMA_R * qr[1])
    _, (true_xs, ys) = jax.lax.scan(
        sim_step, x0, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
    )

    kf_means, kf_vars, kf_log_ml = kalman_filter(ys, PHI, SIGMA_Q, SIGMA_R)

    ys_2d = ys[:, None]
    twist_means, twist_vars = backward_information_filter_1d(
        ys, PHI, SIGMA_Q, SIGMA_R
    )
    twist_params = {
        "mean":      twist_means[:, None],
        "log_scale": 0.5 * jnp.log(twist_vars)[:, None],
    }
    model_params = {"phi": PHI, "sigma_q": SIGMA_Q, "sigma_r": SIGMA_R}

    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=D, tau=TAU)

    key, tape_key, x0_key, deer_key = jr.split(key, 4)
    tape = tgpf.make_tape(ys_2d, twist_params, tape_key, N=N, K=K)
    x0s = jr.normal(x0_key, (N, D))

    # Sequential
    run_seq = jax.jit(tgpf.run_sequential)
    _ = run_seq(x0s, tape, model_params)              # warmup
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_seq, _, logZ_seq, ess_seq = run_seq(x0s, tape, model_params)
    jax.block_until_ready(states_seq)
    t_seq = time.perf_counter() - t0

    # Parallel
    params_par = {**model_params, "key": deer_key, "tau": TAU}
    run_par = jax.jit(tgpf.run_parallel, static_argnames=("max_iter",))
    _ = run_par(x0s, tape, params_par, max_iter=MAX_ITER)
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_par, iters_per_win, logZ_par, ess_par = run_par(
        x0s, tape, params_par, max_iter=MAX_ITER
    )
    jax.block_until_ready(states_par)
    t_par = time.perf_counter() - t0

    # Metrics
    max_diff = float(jnp.max(jnp.abs(states_par - states_seq)))
    rmse_seq = float(jnp.sqrt(jnp.mean(
        (states_seq.mean(axis=0)[:, 0] - kf_means)**2
    )))
    rmse_par = float(jnp.sqrt(jnp.mean(
        (states_par.mean(axis=0)[:, 0] - kf_means)**2
    )))

    # iters_per_win has shape (num_windows, N). Report max.
    max_iters = int(jnp.max(iters_per_win))
    mean_iters = float(jnp.mean(iters_per_win))

    print("=" * 72)
    print(f"Twisted-Gumbel PF on LGSSM     T={T}  N={N}  K={K}")
    print("=" * 72)
    print(f"Sequential : time = {t_seq:.4f}s    log Z = {float(logZ_seq):+.4f}    "
          f"RMSE vs Kalman = {rmse_seq:.4f}")
    print(f"Parallel   : time = {t_par:.4f}s    log Z = {float(logZ_par):+.4f}    "
          f"RMSE vs Kalman = {rmse_par:.4f}")
    print(f"Speedup    : {t_seq / t_par:.2f}x")
    print(f"Kalman     :                       log Z = {float(kf_log_ml):+.4f}")
    print("-" * 72)
    print(f"max |state_par - state_seq|  = {max_diff:.3e}")
    print(f"DEER iters per window  max   = {max_iters}")
    print(f"                       mean  = {mean_iters:.2f}")
    print(f"ESS / N  (at resample)  seq  min={float(jnp.min(ess_seq))/N:.3f}  "
          f"mean={float(jnp.mean(ess_seq))/N:.3f}")
    print(f"                        par  min={float(jnp.min(ess_par))/N:.3f}  "
          f"mean={float(jnp.mean(ess_par))/N:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
