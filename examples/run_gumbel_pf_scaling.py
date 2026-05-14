"""
run_gumbel_pf_scaling.py

Wall-clock scaling sweep for the Gumbel-STE parallel particle filter on GPU.

For each T in a geometric sweep we run the same tape through both
solvers and report:
    - seq PF wall-clock,
    - DEER-PF wall-clock + iteration count,
    - speedup,
    - max |state_par - state_seq| (numerical match diagnostic).

DEER's per-iteration cost is one associative scan: O(log T) parallel
depth on accelerator vs O(T) for the sequential scan. The crossover
should occur at moderate T and grow with hardware parallelism.
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src.gumbel_particle_filter import GumbelParallelParticleFilter


PHI, SIGMA_Q, SIGMA_R = 0.9, 0.5, 1.0
D, N = 1, 32
TAU = 1.0
DAMP = 0.5
MAX_ITER = 1000


def transition_fn(X, noise, params):
    return params["phi"] * X + params["sigma_q"] * noise


def log_lik_fn(X_pred, obs, params):
    sr = params["sigma_r"]
    residual = obs[:, None] - X_pred
    log_norm = -0.5 * jnp.log(2 * jnp.pi * sr**2) * X_pred.shape[0]
    return log_norm - 0.5 * jnp.sum(residual**2, axis=0) / sr**2


def time_call(fn, *args, repeat=3):
    """Run fn(*args) repeat times after one warmup. Return min time."""
    out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.time()
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        times.append(time.time() - t0)
    return min(times), out


def run_one_T(T, seed=0):
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 6)
    x0 = jr.normal(keys[0])
    qn = jr.normal(keys[1], (T,))
    rn = jr.normal(keys[2], (T,))

    def sim(x, no):
        q, r = no
        x = PHI * x + SIGMA_Q * q
        return x, (x, x + SIGMA_R * r)

    _, (_, ys) = jax.lax.scan(sim, x0, (qn, rn))
    pf = GumbelParallelParticleFilter(transition_fn, log_lik_fn, D=D, N=N, tau=TAU)
    tape = pf.make_tape(ys[:, None], keys[3])
    s0 = jr.normal(keys[4], (D * N,))
    params = {"phi": PHI, "sigma_q": SIGMA_Q, "sigma_r": SIGMA_R}
    params_par = {**params, "key": keys[5], "tau": TAU}

    run_seq = jax.jit(pf.run_sequential)
    run_par = jax.jit(pf.run_parallel, static_argnames=("max_iter", "solver"))

    t_seq, seq = time_call(run_seq, s0, tape, params)

    def par_wrap(s0_, tape_, p_):
        return run_par(s0_, tape_, p_, max_iter=MAX_ITER,
                       solver="full", damp_factor=DAMP)

    t_par, (par, n_iters) = time_call(par_wrap, s0, tape, params_par)
    err = float(jnp.max(jnp.abs(par - seq)))
    return t_seq, t_par, int(n_iters), err


def main():
    print(f"Device: {jax.devices()}    N={N}  D={D}  tau={TAU}  damp={DAMP}")
    print("=" * 84)
    print(f"{'T':>6}  {'t_seq (s)':>11}  {'t_par (s)':>11}  {'speedup':>8}  "
          f"{'iters':>6}  {'max |par-seq|':>14}")
    print("-" * 84)

    Ts = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for T in Ts:
        try:
            t_seq, t_par, n_iters, err = run_one_T(T)
            speedup = t_seq / t_par
            print(f"{T:>6}  {t_seq:>11.4f}  {t_par:>11.4f}  "
                  f"{speedup:>7.2f}x  {n_iters:>6}  {err:>14.2e}")
        except Exception as e:
            print(f"{T:>6}  failed: {e}")
            break


if __name__ == "__main__":
    main()
