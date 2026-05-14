"""
run_stampede_lgssm.py

Smallest end-to-end demo of the revived "STAMPEDE" idea:
  parallel-in-time MALA  +  parallel-in-time twisted-Gumbel PF.

Each MALA step accepts/rejects a proposed parameter theta using a
log-likelihood log p(y | theta) computed by the windowed Gumbel-STE
particle filter built in this branch. The MALA chain is solved in
parallel-in-time via DEER (src.samplers.ParallelMALA), and inside
each DEER iteration the K_MALA log-likelihood evaluations are vmap'd
into K_MALA simultaneous PF runs -- each itself parallel-in-time.

The compositional parallel depth is therefore
    O( <const> * <log T> )
because the K_MALA evaluations all live on the vmap axis (width),
and each PF is internally parallel-in-time (depth log T-ish).

Model
-----
1D LGSSM with a SINGLE unknown parameter phi (state-transition coeff):
    x_t = phi * x_{t-1}  +  sigma_q * eps_t
    y_t = x_t            +  sigma_r * eta_t
sigma_q and sigma_r are fixed at their true values for simplicity --
this isolates the wiring question (does parallel MALA + parallel PF
compose?) from any tuning of multi-parameter MALA.

Sanity check
------------
Sequential MALA and parallel MALA are run on the SAME PRNG drivers
and SAME PF tape. With DEER converged the two chains must agree
up to numerical tolerance. Reporting:
    - final-state agreement,
    - max trajectory difference,
    - DEER iters on MALA,
    - wall-clock.
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src import samplers
from src.gumbel_particle_filter import TwistedGumbelParticleFilter


# --- model truth ---
PHI_TRUE = 0.9
SIGMA_Q  = 0.5
SIGMA_R  = 1.0

# --- PF (inner) ---
T   = 1024
N   = 32
K_W = 64      # PF window size
TAU = 1.0
PF_MAX_ITER = 100

# --- MALA (outer) ---
D_PARAM      = 1
CHAIN_LEN    = 32
STEP_SIZE    = 5e-3
MALA_MAX_ITER = 200


# ---- twisted-PF step / log-weight (LGSSM, diagonal twist) ----

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


# ---- log-posterior used by MALA ----
# theta = [phi]. Prior on phi: N(0, 1). Likelihood from the parallel-in-time PF.
# The PF tape and initial particles are passed via target_params and held
# fixed across MALA steps -> log_prob is a deterministic function of theta,
# which makes the sequential-vs-parallel MALA comparison crisp.

def build_log_prob(tgpf):
    def log_prob(theta, tp):
        phi = theta[0]
        model_params = {
            "phi":     phi,
            "sigma_q": SIGMA_Q,
            "sigma_r": SIGMA_R,
            "tau":     TAU,
            "key":     tp["deer_key"],
        }
        # tgpf.run_parallel -> parallel-in-time PF; log_Z = log p(y | phi)
        _, _, log_Z, _ = tgpf.run_parallel(
            tp["x0s"], tp["tape"], model_params, max_iter=PF_MAX_ITER
        )
        log_prior = -0.5 * phi**2          # standard normal prior on phi
        return log_Z + log_prior
    return log_prob


def main():
    print(f"Device: {jax.devices()}")
    print(f"STAMPEDE LGSSM sanity check  "
          f"T={T}  N={N}  K_W={K_W}  CHAIN_LEN={CHAIN_LEN}  STEP={STEP_SIZE}")

    key = jr.PRNGKey(0)
    key, k_data1, k_data2, k_data3 = jr.split(key, 4)

    # --- simulate observations under PHI_TRUE ---
    x0 = jr.normal(k_data1)
    def sim_step(x, qr):
        xn = PHI_TRUE * x + SIGMA_Q * qr[0]
        return xn, (xn, xn + SIGMA_R * qr[1])
    _, (true_xs, ys) = jax.lax.scan(
        sim_step, x0, (jr.normal(k_data2, (T,)), jr.normal(k_data3, (T,)))
    )

    # --- build a flat twist (~bootstrap) so log_prob doesn't depend on phi
    # through the twist init.  All phi-dependence flows through the PF
    # transition itself.
    twist_params = {
        "mean":      jnp.zeros((T, 1)),
        "log_scale": jnp.full((T, 1), 0.5 * jnp.log(100.0)),  # var=100
    }

    # --- build TwistedGumbelParticleFilter and a fixed tape ---
    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=1, tau=TAU)
    key, tape_key, x0s_key, deer_key = jr.split(key, 4)
    tape = tgpf.make_tape(ys[:, None], twist_params, tape_key, N=N, K=K_W)
    x0s = jr.normal(x0s_key, (N, 1))

    target_params = {
        "tape":     tape,
        "x0s":      x0s,
        "deer_key": deer_key,
    }
    log_prob = build_log_prob(tgpf)

    # --- one-off correctness probe of log_prob ---
    lp0 = log_prob(jnp.array([0.5]), target_params)
    lp1 = log_prob(jnp.array([PHI_TRUE]), target_params)
    print(f"  log p(phi=0.5)       = {float(lp0):+.4f}")
    print(f"  log p(phi=phi_true)  = {float(lp1):+.4f}   "
          f"(should be the higher one)")

    # --- MALA setup ---
    # autodiff_mode='forward' is REQUIRED because log_prob contains
    # jax.lax.while_loop (inside qdeer.seq1d) which reverse-mode autodiff
    # cannot handle.
    sampler = samplers.ParallelMALA(
        log_prob, D_PARAM, CHAIN_LEN, MALA_MAX_ITER,
        autodiff_mode="forward",
    )

    key, mala_key, mala_deer_key = jr.split(key, 3)
    initial_state = jnp.array([0.0])                       # start at prior mean
    mala_params = {
        "step_size":     STEP_SIZE,
        "target_params": target_params,
        "grad_max_norm": 50.0,
        "key":           mala_deer_key,                    # for outer DEER Hutchinson
    }

    # --- sequential MALA ---
    run_seq = jax.jit(sampler.run_sequential_mala)
    _ = run_seq(mala_key, initial_state, mala_params)              # warmup
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_seq = run_seq(mala_key, initial_state, mala_params)
    states_seq.block_until_ready()
    t_seq = time.perf_counter() - t0

    # --- parallel MALA ---
    yinit_guess = jnp.tile(initial_state[None, :], (CHAIN_LEN, 1))
    run_par = jax.jit(sampler.run_parallel_mala)
    _ = run_par(mala_key, initial_state, yinit_guess, mala_params)  # warmup
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_par, iters = run_par(mala_key, initial_state, yinit_guess, mala_params)
    states_par.block_until_ready()
    t_par = time.perf_counter() - t0

    max_diff = float(jnp.max(jnp.abs(states_seq - states_par)))
    final_seq = float(states_seq[-1, 0])
    final_par = float(states_par[-1, 0])

    print("=" * 72)
    print(f"Sequential MALA : time = {t_seq:.4f}s   final phi = {final_seq:+.4f}")
    print(f"Parallel   MALA : time = {t_par:.4f}s   final phi = {final_par:+.4f}   "
          f"DEER iters = {int(iters)}")
    print(f"Speedup         : {t_seq / t_par:.2f}x")
    print(f"max |seq - par| = {max_diff:.3e}")
    print(f"true phi        = {PHI_TRUE:+.4f}")
    print(f"chain trajectory (seq):  {[round(float(s[0]), 4) for s in states_seq]}")
    print(f"chain trajectory (par):  {[round(float(s[0]), 4) for s in states_par]}")


if __name__ == "__main__":
    main()
