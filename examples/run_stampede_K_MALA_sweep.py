"""
run_stampede_K_MALA_sweep.py

Sweep MALA chain length K_MALA at fixed PF size to find the crossover
where parallel-in-MALA-steps becomes faster than sequential.

Hypothesis: at fixed PF cost C_PF, sequential MALA costs K_MALA * C_PF.
Parallel MALA costs (DEER_iters * vmap_of_K_MALA_PFs). On a wide GPU
the vmap-of-N PFs costs roughly the same as one PF up to memory, so
parallel cost is ~DEER_iters * C_PF and the speedup is K_MALA / DEER_iters.
Crossover is K_MALA >= DEER_iters (~24 in our T=1024 setup).
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src import samplers
from src.gumbel_particle_filter import TwistedGumbelParticleFilter


PHI_TRUE = 0.9
SIGMA_Q  = 0.5
SIGMA_R  = 1.0

T   = 1024
N   = 32
K_W = 64
TAU = 1.0
PF_MAX_ITER = 100

D_PARAM       = 1
STEP_SIZE     = 5e-3
MALA_MAX_ITER = 1000
MALA_K_BLOCK  = 50          # windowed-MALA block size (memory: ~50 sweet spot)


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
                 - 0.5 * jnp.log(2 * jnp.pi * sr**2))
    log_trans = (-0.5 * jnp.sum((x_next - phi * x_prev)**2) / sq**2
                 - 0.5 * jnp.log(2 * jnp.pi * sq**2))
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    log_prop  = (-0.5 * jnp.sum((x_next - mp)**2 / vp)
                 - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * vp)))
    return log_lik + log_trans - log_prop


def parallel_bif_1d(ys, phi, sigma_q, sigma_r):
    """Parallel BIF for the 1D LGSSM via closed-form stationary J* and
    an affine associative scan for h. O(log T) parallel depth."""
    T = ys.shape[0]
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2
    b = 1.0 - phi**2 - sq * prec_obs
    J_star = (-b + jnp.sqrt(b**2 + 4.0 * sq * prec_obs)) / (2.0 * sq)
    alpha  = phi / (1.0 + sq * J_star)

    alphas    = jnp.full(T, alpha)
    betas_rev = ys[::-1] * prec_obs

    def compose(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_r * a_l, a_r * b_l + b_r)

    _, cum_betas = jax.lax.associative_scan(compose, (alphas, betas_rev))
    hs = cum_betas[::-1]
    twist_means = hs / J_star
    twist_vars  = jnp.full(T, 1.0 / J_star)
    return twist_means, twist_vars


def time_min(fn, *args, repeat=3):
    out = fn(*args); jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    return min(times), out


def run_K_MALA(K_MALA, ys, x0s, tape, deer_key, mala_key):
    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=1, tau=TAU)

    target_params = {
        "tape":     tape,
        "x0s":      x0s,
        "deer_key": deer_key,
    }

    def log_prob(theta, tp):
        phi = jnp.tanh(theta[0])
        model_params = {
            "phi": phi, "sigma_q": SIGMA_Q, "sigma_r": SIGMA_R,
            "tau": TAU, "key": tp["deer_key"],
        }
        _, _, log_Z, _ = tgpf.run_parallel(
            tp["x0s"], tp["tape"], model_params, max_iter=PF_MAX_ITER
        )
        log_jac = jnp.log(1.0 - phi**2 + 1e-12)
        return log_Z + log_jac

    # Window size cannot exceed chain length.
    k_block_eff = min(MALA_K_BLOCK, K_MALA)
    sampler = samplers.ParallelMALA(
        log_prob, D_PARAM, K_MALA, MALA_MAX_ITER,
        autodiff_mode="forward",
        window_size=k_block_eff,
    )

    initial_state = jnp.array([0.0])
    mala_params = {
        "step_size":     STEP_SIZE,
        "target_params": target_params,
        "grad_max_norm": 50.0,
        "key":           jr.PRNGKey(123),
    }
    yinit_guess = jnp.tile(initial_state[None, :], (K_MALA, 1))

    run_seq = jax.jit(sampler.run_sequential_mala)
    run_par = jax.jit(sampler.run_parallel_mala_window)

    t_seq, states_seq = time_min(run_seq, mala_key, initial_state, mala_params)
    t_par, (states_par, iters) = time_min(
        run_par, mala_key, initial_state, yinit_guess, mala_params
    )

    err = float(jnp.max(jnp.abs(states_seq - states_par)))
    phi_final_seq = float(jnp.tanh(states_seq[-1, 0]))
    phi_final_par = float(jnp.tanh(states_par[-1, 0]))
    # iters is shape (num_windows,) for windowed MALA
    iters_max  = int(jnp.max(iters)) if iters.ndim > 0 else int(iters)
    iters_mean = float(jnp.mean(iters)) if iters.ndim > 0 else float(iters)
    return t_seq, t_par, iters_max, iters_mean, err, phi_final_seq, phi_final_par


def main():
    print(f"Device: {jax.devices()}")
    print(f"STAMPEDE K_MALA sweep   T={T}  N={N}  K_W={K_W}  step={STEP_SIZE}")

    key = jr.PRNGKey(0)
    key, k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 8)
    x0 = jr.normal(k1)
    def sim(x, qr):
        xn = PHI_TRUE * x + SIGMA_Q * qr[0]
        return xn, (xn, xn + SIGMA_R * qr[1])
    _, (_, ys) = jax.lax.scan(
        sim, x0, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
    )

    # Precompute the BIF twist ONCE at phi=0 (non-cheating anchor) via the
    # PARALLEL BIF (closed-form stationary J* + associative-scan affine
    # recurrence for h). The PF is unbiased for any valid twist; anchoring
    # at the wrong phi only inflates PF variance, not bias.
    twist_means, twist_vars = parallel_bif_1d(ys, 0.0, SIGMA_Q, SIGMA_R)
    twist_params = {
        "mean":      twist_means[:, None],
        "log_scale": 0.5 * jnp.log(twist_vars)[:, None],
    }
    tgpf_for_tape = TwistedGumbelParticleFilter(
        twisted_step_fn, log_weight_fn, D=1, tau=TAU
    )
    tape = tgpf_for_tape.make_tape(ys[:, None], twist_params, k4, N=N, K=K_W)
    x0s = jr.normal(k5, (N, 1))

    print("=" * 92)
    print(f"{'K_MALA':>7}  {'t_seq (s)':>11}  {'t_par (s)':>11}  {'speedup':>8}  "
          f"{'iters':>6}  {'max|seq-par|':>13}  {'phi_seq':>9}  {'phi_par':>9}")
    print("-" * 92)

    Ks = [16, 32, 64, 128, 256, 512]
    for K_MALA in Ks:
        try:
            (t_seq, t_par, iters_max, iters_mean,
             err, pf_seq, pf_par) = run_K_MALA(
                K_MALA, ys, x0s, tape, k6, k7
            )
            print(f"{K_MALA:>7}  {t_seq:>11.4f}  {t_par:>11.4f}  "
                  f"{t_seq/t_par:>7.2f}x  {iters_max:>6}  {err:>13.2e}  "
                  f"{pf_seq:>+9.4f}  {pf_par:>+9.4f}")
        except Exception as e:
            print(f"{K_MALA:>7}  failed: {type(e).__name__}: {e}")
            break


if __name__ == "__main__":
    main()
