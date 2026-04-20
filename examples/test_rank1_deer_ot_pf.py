"""
test_rank1_deer_ot_pf.py

Minimal test of rank-1 Hutchinson DEER on the OT-resampling particle filter.

The OT-PF has a *dense* per-step Jacobian because Sinkhorn resampling
couples all particles.  We compare convergence of:

  1. Sequential OT-PF    (ground truth, jax.lax.scan)
  2. Diagonal DEER        (current qdeer, expected to stall)
  3. Rank-1 DEER          (new)

on a small D=2, N=32, T=64 LGSSM problem.  Metrics: final L2 error vs
sequential PF, iterations to converge, wall time.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src.particle_filter import ParallelParticleFilter
from src import qdeer


# Model setup: small LGSSM 
D = 2
N = 32
T = 64

A = jnp.array([[0.95, 0.05], [-0.05, 0.95]])
Q = 0.3
H = jnp.eye(D)
R = 0.5


def transition_fn(X, noise, params):
    """x_t = A x_{t-1} + Q * noise."""
    return params['A'] @ X + params['Q'] * noise


def log_lik_fn(X_pred, obs, params):
    """y_t ~ N(H x_t, R^2 I)."""
    diff = obs[:, None] - params['H'] @ X_pred        # (D, N)
    return -0.5 * jnp.sum(diff ** 2, axis=0) / params['R'] ** 2


# Simulate data 
key = jr.PRNGKey(0)
key, k_sim, k_init, k_noise, k_tape = jr.split(key, 5)

def sim_step(x, noise_obs):
    q, r = noise_obs
    xn = A @ x + Q * q
    yn = H @ xn + R * r
    return xn, (xn, yn)

x0_true = jr.normal(k_sim, (D,))
_, (true_xs, ys) = jax.lax.scan(
    sim_step, x0_true, (jr.normal(k_sim, (T, D)), jr.normal(k_sim, (T, D)))
)

# Instantiate OT-PF 
pf = ParallelParticleFilter(
    transition_fn=transition_fn,
    log_lik_fn=log_lik_fn,
    D=D,
    N=N,
    ot_eps=0.1,
    ot_iters=50,
)

model_params = {'A': A, 'Q': Q, 'H': H, 'R': R}

# Initial particles + tape of pre-drawn noise
X0 = jr.normal(k_init, (D, N))
s0 = X0.ravel()
noise_tape = jr.normal(k_noise, (T, D, N))
tape = (ys, noise_tape)

print(f"LGSSM: D={D}, N={N}, T={T}  (state dim DN = {D*N})")
print(f"ys range: [{float(ys.min()):.2f}, {float(ys.max()):.2f}]")

#  1. Sequential OT-PF
print("\n--- [1] Sequential OT-PF ---")
run_seq = jax.jit(pf.run_sequential)
states_seq = run_seq(s0, tape, model_params)
states_seq.block_until_ready()

t0 = time.perf_counter()
states_seq = run_seq(s0, tape, model_params)
states_seq.block_until_ready()
t_seq = time.perf_counter() - t0
print(f"  Time: {t_seq*1000:.1f} ms | Output shape: {states_seq.shape}")

# Compute filter means for visualisation
def filter_means(states):
    return states.reshape(T, D, N).mean(axis=-1)   # (T, D)
means_seq = filter_means(states_seq)
print(f"  Filter mean[0] = {means_seq[0]}  true x[0] = {true_xs[0]}")

#  2. Diagonal DEER
print("\n--- [2] Diagonal DEER ---")
key, deer_key = jr.split(key)
params_deer = {**model_params, 'key': deer_key}

# Warm-start from zeros
yinit = jnp.zeros((T, D * N))

run_diag = jax.jit(pf.run_parallel, static_argnames=("max_iter",))
states_diag, iters_diag = run_diag(
    s0, tape, params_deer, yinit_guess=yinit, max_iter=200)
states_diag.block_until_ready()

t0 = time.perf_counter()
states_diag, iters_diag = run_diag(
    s0, tape, params_deer, yinit_guess=yinit, max_iter=200)
states_diag.block_until_ready()
t_diag = time.perf_counter() - t0

err_diag = float(jnp.sqrt(jnp.mean((states_diag - states_seq) ** 2)))
print(f"  Time: {t_diag*1000:.1f} ms | Iters: {int(iters_diag)} | "
      f"RMSE vs seq: {err_diag:.4e}")

#  3. Rank-1 DEER
print("\n--- [3] Rank-1 DEER ---")
key, r1_key = jr.split(key)
params_r1 = {**model_params, 'key': r1_key}


def run_rank1(s0, tape, params, yinit_guess, max_iter):
    return qdeer.seq1d_rank1(
        pf.pf_step, s0, tape, params,
        yinit_guess=yinit_guess, max_iter=max_iter,
    )

run_r1_jit = jax.jit(run_rank1, static_argnames=("max_iter",))
states_r1, iters_r1 = run_r1_jit(s0, tape, params_r1, yinit, 200)
states_r1.block_until_ready()

t0 = time.perf_counter()
states_r1, iters_r1 = run_r1_jit(s0, tape, params_r1, yinit, 200)
states_r1.block_until_ready()
t_r1 = time.perf_counter() - t0

err_r1 = float(jnp.sqrt(jnp.mean((states_r1 - states_seq) ** 2)))
print(f"  Time: {t_r1*1000:.1f} ms | Iters: {int(iters_r1)} | "
      f"RMSE vs seq: {err_r1:.4e}")

# Summary 
print("\n" + "=" * 64)
print(f"{'Method':<20} {'Time (ms)':>10} {'Iters':>8} {'RMSE vs seq':>14}")
print("-" * 64)
print(f"{'Sequential':<20} {t_seq*1000:>10.1f} {'--':>8} {'0.0':>14}")
print(f"{'Diagonal DEER':<20} {t_diag*1000:>10.1f} {int(iters_diag):>8} "
      f"{err_diag:>14.4e}")
print(f"{'Rank-1 DEER':<20} {t_r1*1000:>10.1f} {int(iters_r1):>8} "
      f"{err_r1:>14.4e}")
