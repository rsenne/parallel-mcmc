"""
run_twisted_gumbel_pf_scaling.py

Wall-clock scaling sweep for the twisted-Gumbel parallel particle
filter (windowed DEER + Gumbel-STE checkpoint resampling).

Sweeps T while keeping N and the window size K fixed. For each T:
  - sequential twisted-Gumbel PF (jax.lax.scan over windows),
  - parallel  twisted-Gumbel PF (per-particle DEER over windows),
both on the SAME tape, and reports seq time, par time, speedup,
DEER iter stats, and max |par - seq|.

Hypothesis: per-window DEER iters stay ~O(1) (the twist makes each
intra-window chain easy), so par cost grows like
    num_windows * (deer_iters * log K + N^2)
vs the sequential O(T) cost. Speedup should emerge once T is large
enough that the per-window parallel work dominates the outer scan.
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src.gumbel_particle_filter import TwistedGumbelParticleFilter


PHI, SIGMA_Q, SIGMA_R = 0.9, 0.5, 1.0
D = 1
N = 64
K = 32
TAU = 1.0
MAX_ITER = 100


def backward_information_filter_1d(ys, phi, sigma_q, sigma_r):
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2

    def step(carry, y):
        J, h = carry
        denom = 1.0 + sq * J
        J_pred = phi**2 * J / denom
        h_pred = phi * h / denom
        return (J_pred + prec_obs, h_pred + y * prec_obs), None

    J_T = prec_obs
    h_T = ys[-1] * prec_obs
    (_, _), _ = jax.lax.scan(step, (J_T, h_T), ys[-2::-1])

    # Re-scan keeping history (the carry-only version is for tightness)
    def step_full(carry, y):
        J, h = carry
        denom = 1.0 + sq * J
        J_pred = phi**2 * J / denom
        h_pred = phi * h / denom
        J_new = J_pred + prec_obs
        h_new = h_pred + y * prec_obs
        return (J_new, h_new), (J_new, h_new)
    _, (Js_rev, hs_rev) = jax.lax.scan(step_full, (J_T, h_T), ys[-2::-1])
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


def time_min(fn, *args, repeat=3):
    """Run fn(*args) once for warmup, then min over `repeat` real timings."""
    out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    return min(times), out


def run_one_T(T, seed=0):
    key = jr.PRNGKey(seed)
    key, k1, k2, k3, k4, k5, k6 = jr.split(key, 7)
    x0 = jr.normal(k1)

    def sim_step(x, qr):
        xn = PHI * x + SIGMA_Q * qr[0]
        return xn, (xn, xn + SIGMA_R * qr[1])
    _, (_, ys) = jax.lax.scan(
        sim_step, x0, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
    )

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
    tape = tgpf.make_tape(ys[:, None], twist_params, k4, N=N, K=K)
    x0s = jr.normal(k5, (N, D))

    run_seq = jax.jit(tgpf.run_sequential)
    run_par = jax.jit(tgpf.run_parallel, static_argnames=("max_iter",))

    params_par = {**model_params, "key": k6, "tau": TAU}

    t_seq, (states_seq, _, _) = time_min(run_seq, x0s, tape, model_params)
    t_par, (states_par, iters_per_win, _) = time_min(
        lambda x0_, t_, p_: run_par(x0_, t_, p_, max_iter=MAX_ITER),
        x0s, tape, params_par,
    )

    err = float(jnp.max(jnp.abs(states_par - states_seq)))
    max_iters = int(jnp.max(iters_per_win))
    mean_iters = float(jnp.mean(iters_per_win))
    return t_seq, t_par, max_iters, mean_iters, err


def main():
    print(f"Device: {jax.devices()}    N={N}  D={D}  K={K}  tau={TAU}")
    print("=" * 96)
    print(f"{'T':>7}  {'t_seq (s)':>11}  {'t_par (s)':>11}  {'speedup':>8}  "
          f"{'iters max':>10}  {'iters mean':>10}  {'max |par-seq|':>14}")
    print("-" * 96)

    Ts = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    for T in Ts:
        try:
            t_seq, t_par, max_iters, mean_iters, err = run_one_T(T)
            speedup = t_seq / t_par
            print(f"{T:>7}  {t_seq:>11.4f}  {t_par:>11.4f}  "
                  f"{speedup:>7.2f}x  {max_iters:>10}  {mean_iters:>10.2f}  "
                  f"{err:>14.2e}")
        except Exception as e:
            print(f"{T:>7}  failed: {type(e).__name__}: {e}")
            break


if __name__ == "__main__":
    main()
