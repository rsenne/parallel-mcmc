"""
run_gumbel_pf_linear_gaussian.py

Validation experiment for the hard-forward / soft-Jacobian Gumbel-Softmax
parallel particle filter on a 1-D linear-Gaussian SSM.

Model:
    x_0  ~ N(0, 1)
    x_t  = phi * x_{t-1} + sigma_q * eps_t,   eps_t ~ N(0, 1)
    y_t  = x_t           + sigma_r * eta_t,   eta_t ~ N(0, 1)

The central claim of the construction is:
    DEER-PF with a given (transition-noise, Gumbel) tape must converge
    to the *same* trajectory as a sequential bootstrap PF with the same
    tape, because the forward step is an exact Gumbel-max categorical
    draw — only the Jacobian is softened.

This script verifies that by:
  (a) running both on identical tapes,
  (b) reporting max |state_par - state_seq|,
  (c) reporting ancestor-index agreement,
  (d) reporting log-Z agreement and filtering-mean RMSE vs Kalman,
  (e) reporting DEER iterations and wall-clock time.

Note on wall-clock: on CPU the parallel solver is slower than the
sequential PF because each DEER iteration costs an associative scan
over T. The speedup is on GPU/TPU where the scan parallelises, with
runtime scaling like O(iters * log T) vs O(T). What this script
demonstrates is *correctness*: that DEER converges to the exact
sequential trajectory under shared randomness.
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.gumbel_particle_filter import GumbelParallelParticleFilter


PHI     = 0.9
SIGMA_Q = 0.5
SIGMA_R = 1.0

T   = 512
D   = 1
N   = 32
TAU = 1.0            # Gumbel-Softmax temperature for the DEER Jacobian
SOLVER = "full"      # 'full' | 'quasi' | 'rank1'
DAMP = 0.5
MAX_ITER = 600


def kalman_filter(ys, phi, sigma_q, sigma_r):
    def step(carry, y):
        m, P = carry
        mp = phi * m
        Pp = phi**2 * P + sigma_q**2
        S  = Pp + sigma_r**2
        K  = Pp / S
        mf = mp + K * (y - mp)
        Pf = (1 - K) * Pp
        log_lik = -0.5 * (jnp.log(2 * jnp.pi * S) + (y - mp)**2 / S)
        return (mf, Pf), (mf, Pf, log_lik)
    _, (means, variances, log_liks) = jax.lax.scan(step, (0.0, 1.0), ys)
    return means, variances, jnp.sum(log_liks)


def transition_fn(X, noise, params):
    return params["phi"] * X + params["sigma_q"] * noise


def log_lik_fn(X_pred, obs, params):
    # Gaussian log-likelihood including normalising constant — matters for log-Z
    sr = params["sigma_r"]
    residual = obs[:, None] - X_pred                              # (D, N)
    log_norm = -0.5 * jnp.log(2 * jnp.pi * sr**2) * X_pred.shape[0]
    return log_norm - 0.5 * jnp.sum(residual**2, axis=0) / sr**2


def ancestors_along_trajectory(pf, prev_states, tape, params):
    """Vectorised: for each t compute argmax_j (log_w_t + g_t,i,j) given
    the post-resample state at t-1 = prev_states[t]."""
    observations, noise, gumbel = tape

    def one_step(prev_state, obs_t, noise_t, gumbel_t):
        X = prev_state.reshape(pf.D, pf.N)
        X_pred = pf.transition_fn(X, noise_t, params)
        log_w = pf.log_lik_fn(X_pred, obs_t, params)
        log_a = log_w - jax.scipy.special.logsumexp(log_w)
        logits = log_a[None, :] + gumbel_t
        return jnp.argmax(logits, axis=-1)                                # (N,)

    return jax.vmap(one_step)(prev_states, observations, noise, gumbel)   # (T, N)


def main():
    master_key = jr.PRNGKey(0)
    key, k1, k2, k3 = jr.split(master_key, 4)

    x0 = jr.normal(k1) * 1.0

    def sim_step(x, noise_obs):
        q, r = noise_obs
        x_new = PHI * x + SIGMA_Q * q
        y     = x_new + SIGMA_R * r
        return x_new, (x_new, y)

    q_noise = jr.normal(k2, (T,))
    r_noise = jr.normal(k3, (T,))
    _, (true_xs, ys) = jax.lax.scan(sim_step, x0, (q_noise, r_noise))
    ys_2d = ys[:, None]

    params = {"phi": PHI, "sigma_q": SIGMA_Q, "sigma_r": SIGMA_R}

    pf = GumbelParallelParticleFilter(transition_fn, log_lik_fn, D=D, N=N, tau=TAU)

    key, tape_key = jr.split(key)
    tape = pf.make_tape(ys_2d, tape_key)

    # Initial particles: prior N(0,1). Re-used for both sequential and parallel.
    key, s0_key = jr.split(key)
    s0 = jr.normal(s0_key, (D * N,))

    # Sequential PF
    run_seq = jax.jit(pf.run_sequential)
    _ = run_seq(s0, tape, params).block_until_ready()                     # warmup

    t0 = time.time()
    states_seq = run_seq(s0, tape, params).block_until_ready()
    t_seq = time.time() - t0

    # Parallel DEER-PF on the SAME tape
    key, deer_key = jr.split(key)
    params_par = {**params, "key": deer_key, "tau": TAU}
    run_par = jax.jit(pf.run_parallel, static_argnames=("max_iter", "solver"))

    _ = run_par(s0, tape, params_par, max_iter=MAX_ITER,
                solver=SOLVER, damp_factor=DAMP)                          # warmup

    t0 = time.time()
    states_par, n_iters = run_par(
        s0, tape, params_par, max_iter=MAX_ITER,
        solver=SOLVER, damp_factor=DAMP,
    )
    states_par.block_until_ready()
    t_par = time.time() - t0

    # Diagnostics
    max_state_err = jnp.max(jnp.abs(states_par - states_seq))

    # Ancestor agreement: compute argmax ancestors implied by each trajectory
    # using its own prev-states. If forward law is preserved, these match.
    prev_states_seq = jnp.concatenate([s0[None, :], states_seq[:-1]], axis=0)
    prev_states_par = jnp.concatenate([s0[None, :], states_par[:-1]], axis=0)
    anc_seq = ancestors_along_trajectory(pf, prev_states_seq, tape, params)
    anc_par = ancestors_along_trajectory(pf, prev_states_par, tape, params)
    anc_match_fraction = jnp.mean((anc_seq == anc_par).astype(jnp.float32))

    # Log-marginal-likelihood estimates
    logZ_seq = pf.log_marginal_likelihood(s0, states_seq, tape, params)
    logZ_par = pf.log_marginal_likelihood(s0, states_par, tape, params)

    # Kalman ground truth
    kf_means, kf_vars, kf_logZ = kalman_filter(ys, PHI, SIGMA_Q, SIGMA_R)

    seq_means = states_seq.reshape(T, D, N).mean(axis=-1)[:, 0]
    par_means = states_par.reshape(T, D, N).mean(axis=-1)[:, 0]

    rmse_seq = jnp.sqrt(jnp.mean((seq_means - kf_means)**2))
    rmse_par = jnp.sqrt(jnp.mean((par_means - kf_means)**2))

    print("=" * 72)
    print(f"Gumbel-STE Parallel PF on LGSSM   T={T}  N={N}  D={D}  "
          f"tau={TAU}  solver={SOLVER}")
    print("=" * 72)
    print(f"Sequential PF :  time={t_seq:.3f}s   RMSE vs Kalman = {rmse_seq:.4f}")
    print(f"Parallel   PF :  time={t_par:.3f}s   RMSE vs Kalman = {rmse_par:.4f}"
          f"   DEER iters = {int(n_iters)}")
    print(f"Speedup       :  {t_seq / t_par:.2f}x")
    print("-" * 72)
    print(f"max |state_par - state_seq|        = {float(max_state_err):.3e}")
    print(f"ancestor-index agreement fraction  = {float(anc_match_fraction):.4f}")
    print(f"log Z   Kalman        = {float(kf_logZ):.4f}")
    print(f"        sequential PF = {float(logZ_seq):.4f}  "
          f"(err {float(logZ_seq - kf_logZ):+.4f})")
    print(f"        parallel   PF = {float(logZ_par):.4f}  "
          f"(err {float(logZ_par - kf_logZ):+.4f})")
    print("=" * 72)

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t_idx = jnp.arange(T)

    ax = axes[0]
    ax.plot(t_idx, true_xs, "k-", lw=1.0, alpha=0.6, label="true state")
    ax.plot(t_idx, ys, "k.", ms=2, alpha=0.25, label="observations")
    ax.plot(t_idx, kf_means, "g-", lw=1.4, label="Kalman mean")
    ax.plot(t_idx, seq_means, "b--", lw=1.2,
            label=f"sequential Gumbel-PF (RMSE={float(rmse_seq):.3f})")
    ax.legend(fontsize=9)
    ax.set_title("Sequential Gumbel-PF vs Kalman")

    ax = axes[1]
    ax.plot(t_idx, true_xs, "k-", lw=1.0, alpha=0.6)
    ax.plot(t_idx, kf_means, "g-", lw=1.4, label="Kalman mean")
    ax.plot(t_idx, par_means, "r--", lw=1.2,
            label=f"DEER Gumbel-PF (iters={int(n_iters)}, "
                  f"RMSE={float(rmse_par):.3f})")
    ax.legend(fontsize=9)
    ax.set_title("Parallel DEER Gumbel-PF vs Kalman")
    ax.set_xlabel("timestep")

    plt.tight_layout()
    plt.savefig("gumbel_pf_linear_gaussian.png", dpi=150)


if __name__ == "__main__":
    main()
