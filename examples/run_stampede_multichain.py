"""
run_stampede_multichain.py

Multi-chain STAMPEDE on the 1-D LGSSM.

Architecture: each chain is a *sequential* MALA whose log_prob calls the
*parallel* twisted-Gumbel PF for log p(y | phi). Chains are independent
and run together via jax.vmap (or jax.pmap across multiple devices).

Why this beats single-chain parallel-MALA:
  - sequential MALA's per-step cost is one parallel PF (~few ms at T=1024
    on L40S) -- already cheap because the inner is parallel-in-time.
  - parallel-MALA-via-DEER adds outer DEER iterations that cost more than
    they save (see logs/stampede_T_sweep.log).
  - vmap'd independent chains add ZERO depth: M chains run in time of one,
    until the GPU saturates. Posterior samples scale linearly with M.

Reports:
  - wall-clock for 1 chain, M chains via vmap, and (optionally) per-chain
    serial loop.
  - across-chain posterior mean / std at the final MALA step.
  - per-chain final phi to show independent runs land in the same region.
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

D_PARAM   = 1
K_MALA    = 512
STEP_SIZE = 5e-3

CHAINS_M_VALUES = [1, 2, 4, 8, 16, 32]


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
    T = ys.shape[0]
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2
    b = 1.0 - phi**2 - sq * prec_obs
    J_star = (-b + jnp.sqrt(b**2 + 4.0 * sq * prec_obs)) / (2.0 * sq)
    alpha = phi / (1.0 + sq * J_star)
    alphas    = jnp.full(T, alpha)
    betas_rev = ys[::-1] * prec_obs
    def compose(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_r * a_l, a_r * b_l + b_r)
    _, cum_betas = jax.lax.associative_scan(compose, (alphas, betas_rev))
    hs = cum_betas[::-1]
    return hs / J_star, jnp.full(T, 1.0 / J_star)


def time_min(fn, *args, repeat=2):
    out = fn(*args); jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    return min(times), out


def main():
    print(f"Device: {jax.devices()}")
    print(f"STAMPEDE multi-chain   T={T}  N={N}  K_W={K_W}  K_MALA={K_MALA}  "
          f"step={STEP_SIZE}")

    key = jr.PRNGKey(0)
    key, k1, k2, k3, k4, k5 = jr.split(key, 6)

    # --- simulate data once ---
    x0 = jr.normal(k1)
    def sim(x, qr):
        xn = PHI_TRUE * x + SIGMA_Q * qr[0]
        return xn, (xn, xn + SIGMA_R * qr[1])
    _, (_, ys) = jax.lax.scan(
        sim, x0, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
    )

    twist_means, twist_vars = parallel_bif_1d(ys, 0.0, SIGMA_Q, SIGMA_R)
    twist_params = {
        "mean":      twist_means[:, None],
        "log_scale": 0.5 * jnp.log(twist_vars)[:, None],
    }
    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=1, tau=TAU)
    tape = tgpf.make_tape(ys[:, None], twist_params, k4, N=N, K=K_W)
    x0s_pf = jr.normal(k5, (N, 1))

    # --- log_prob: depends only on theta and target_params (PF tape baked in) ---
    target_params = {"tape": tape, "x0s": x0s_pf, "deer_key": jr.PRNGKey(1)}

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

    sampler = samplers.ParallelMALA(
        log_prob, D_PARAM, K_MALA, max_iter=1,   # max_iter unused by seq
        autodiff_mode="forward",
    )

    # --- one-chain function: SEQUENTIAL MALA (uses parallel PF inside) ---
    def run_one_chain(chain_key, init_state):
        mala_params = {
            "step_size":     STEP_SIZE,
            "target_params": target_params,
            "grad_max_norm": 50.0,
        }
        return sampler.run_sequential_mala(chain_key, init_state, mala_params)

    # vmap'd version
    run_multi = jax.jit(jax.vmap(run_one_chain, in_axes=(0, 0)))

    # --- sweep M = number of chains ---
    print("=" * 92)
    print(f"{'M chains':>9}  {'t_total (s)':>13}  {'t/chain (s)':>13}  "
          f"{'speedup vs M=1':>15}  {'phi (mean ± std)':>22}  {'min phi':>9}  {'max phi':>9}")
    print("-" * 92)

    t_one_chain = None
    for M in CHAINS_M_VALUES:
        key, key_chain_seeds, key_chain_inits = jr.split(key, 3)
        chain_keys = jr.split(key_chain_seeds, M)
        chain_inits = 0.1 * jr.normal(key_chain_inits, (M, D_PARAM))

        t, states = time_min(run_multi, chain_keys, chain_inits)
        states.block_until_ready()
        # states: (M, K_MALA, D)
        final_phis = jnp.tanh(states[:, -1, 0])
        phi_mean = float(jnp.mean(final_phis))
        phi_std  = float(jnp.std(final_phis))
        phi_min  = float(jnp.min(final_phis))
        phi_max  = float(jnp.max(final_phis))
        if t_one_chain is None and M == 1:
            t_one_chain = t
        speedup = (t_one_chain * M / t) if t_one_chain else 1.0
        print(f"{M:>9}  {t:>13.4f}  {t/M:>13.4f}  {speedup:>14.2f}x  "
              f"{phi_mean:>+11.4f} ± {phi_std:.4f}   "
              f"{phi_min:>+9.4f}  {phi_max:>+9.4f}")


if __name__ == "__main__":
    main()
