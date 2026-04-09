"""
run_pf_linear_gaussian.py

Demonstrates the parallel particle filter on a 1-D linear-Gaussian SSM
where the Kalman filter provides the exact posterior mean as ground truth.

Model
-----
  x_0   ~ N(0, 1)
  x_t   = phi * x_{t-1}  +  sigma_q * eps_t,   eps_t ~ N(0,1)
  y_t   = x_t            +  sigma_r * eta_t,   eta_t ~ N(0,1)

With phi=0.9, sigma_q=0.5, sigma_r=1.0.

What this shows
---------------
  1. Sequential PF  (jax.lax.scan)   — baseline
  2. Parallel PF    (quasi-DEER)     — parallelised across T
  Both are compared to the Kalman filter mean.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import ParallelParticleFilter


PHI     = 0.9      # state transition coefficient
SIGMA_Q = 0.5      # process noise std
SIGMA_R = 1.0      # observation noise std

T  = 512           # number of timesteps
D  = 1             # latent dimension
N  = 256           # number of particles


def kalman_filter(ys, phi, sigma_q, sigma_r):
    """Returns filtered means and variances for the 1-D linear-Gaussian SSM."""
    T = len(ys)
    means = jnp.zeros(T)
    variances = jnp.zeros(T)

    # prior
    m, P = 0.0, 1.0

    def step(carry, y):
        m, P = carry
        # predict
        m_pred = phi * m
        P_pred = phi**2 * P + sigma_q**2
        # update
        S = P_pred + sigma_r**2
        K = P_pred / S
        m_new = m_pred + K * (y - m_pred)
        P_new = (1.0 - K) * P_pred
        return (m_new, P_new), (m_new, P_new)

    _, (means, variances) = jax.lax.scan(step, (m, P), ys)
    return means, variances

def transition_fn(X, noise, params):
    """X: (D, N), noise: (D, N) -> X_pred: (D, N)"""
    return params["phi"] * X + params["sigma_q"] * noise


def log_lik_fn(X_pred, obs, params):
    """X_pred: (D, N), obs: (D,) -> log_w: (N,)"""
    # obs is (D,) = (1,) here; broadcast over particles
    residual = obs[None, :] - X_pred   # (D, N) — wrong shape, fix:
    # X_pred shape: (D, N),  obs shape: (D,)
    residual = obs[:, None] - X_pred   # (D, N)
    return -0.5 * jnp.sum(residual**2, axis=0) / params["sigma_r"]**2

master_key = jr.PRNGKey(0)
key, k1, k2, k3 = jr.split(master_key, 4)

x0    = jr.normal(k1) * 1.0
xs    = jnp.zeros(T + 1)
xs    = xs.at[0].set(x0)

def sim_step(x, noise_obs):
    q_noise, r_noise = noise_obs
    x_new = PHI * x + SIGMA_Q * q_noise
    y     = x_new + SIGMA_R * r_noise
    return x_new, (x_new, y)

q_noise = jr.normal(k2, (T,))
r_noise = jr.normal(k3, (T,))
_, (true_xs, ys) = jax.lax.scan(sim_step, x0, (q_noise, r_noise))

# observations shape: (T, D) = (T, 1)
ys_2d = ys[:, None]   # (T, 1)

params = {
    "phi":     PHI,
    "sigma_q": SIGMA_Q,
    "sigma_r": SIGMA_R,
}

pf = ParallelParticleFilter(transition_fn, log_lik_fn, D=D, N=N,
                             ot_eps=0.05, ot_iters=50)

key, tape_key = jr.split(key)
tape = pf.make_tape(ys_2d, tape_key)           # (obs (T,1), noise (T,D,N))

# initial particles: sample from prior N(0, 1)
key, s0_key = jr.split(key)
s0 = jr.normal(s0_key, (D * N,))

key, deer_key = jr.split(key)
params["key"] = deer_key                        # required by quasi-DEER


run_seq = jax.jit(pf.run_sequential)
_ = run_seq(s0, tape, params)  # warmup

t0 = time.time()
states_seq = run_seq(s0, tape, params).block_until_ready()
t_seq = time.time() - t0

# particle means at each step
seq_means = states_seq.reshape(T, D, N).mean(axis=-1)[:, 0]   # (T,)


run_par = jax.jit(pf.run_parallel)
key, deer_key2 = jr.split(key)
params["key"] = deer_key2

_ = run_par(s0, tape, params, max_iter=100)  # warmup

t0 = time.time()
states_par, iters = run_par(s0, tape, params, max_iter=100)
states_par.block_until_ready()
t_par = time.time() - t0

par_means = states_par.reshape(T, D, N).mean(axis=-1)[:, 0]   # (T,)

kf_means, kf_vars = kalman_filter(ys, PHI, SIGMA_Q, SIGMA_R)

rmse_seq = jnp.sqrt(jnp.mean((seq_means - kf_means)**2))
rmse_par = jnp.sqrt(jnp.mean((par_means - kf_means)**2))

print(f"T={T}, N={N} particles")
print(f"Sequential PF:  time={t_seq:.3f}s   RMSE vs Kalman={rmse_seq:.4f}")
print(f"Parallel   PF:  time={t_par:.3f}s   RMSE vs Kalman={rmse_par:.4f}  ({iters} DEER iters)")


plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

t_idx = jnp.arange(T)

ax = axes[0]
ax.plot(t_idx, true_xs, "k-", lw=1,   label="true state", alpha=0.7)
ax.plot(t_idx, ys,      "k.", ms=2,   label="observations", alpha=0.3)
ax.plot(t_idx, kf_means, "g-", lw=1.5, label="Kalman mean")
ax.plot(t_idx, seq_means, "b--", lw=1.5, label=f"seq PF mean (RMSE={rmse_seq:.3f})")
ax.legend(fontsize=9)
ax.set_title("Sequential particle filter vs Kalman")

ax = axes[1]
ax.plot(t_idx, true_xs,  "k-", lw=1, alpha=0.7)
ax.plot(t_idx, kf_means, "g-", lw=1.5, label="Kalman mean")
ax.plot(t_idx, par_means, "r--", lw=1.5,
        label=f"parallel PF mean ({iters} iters, RMSE={rmse_par:.3f})")
ax.legend(fontsize=9)
ax.set_title("Parallel particle filter (quasi-DEER) vs Kalman")
ax.set_xlabel("timestep")

plt.tight_layout()
plt.savefig("pf_linear_gaussian.png", dpi=150)
plt.show()
