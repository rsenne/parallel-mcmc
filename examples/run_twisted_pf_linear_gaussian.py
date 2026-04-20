"""
run_twisted_pf_linear_gaussian.py

Demonstrates the *twisted* parallel particle filter on a 1-D linear-Gaussian
SSM.  Instead of OT-based resampling (which couples all particles and makes
DEER's diagonal Jacobian approximation poor), we use twisted proposals that
incorporate look-ahead information to keep weights balanced WITHOUT
resampling.

Each particle trajectory is then an independent D-dimensional recurrence.
DEER solves each one in O(log T) parallel depth, and N particles run in
parallel on GPU.  For D=1 the diagonal Jacobian is exact (it's a scalar),
so DEER converges in very few iterations.

Model
-----
  x_0   ~ N(0, 1)
  x_t   = phi * x_{t-1}  +  sigma_q * eps_t,   eps_t ~ N(0,1)
  y_t   = x_t            +  sigma_r * eta_t,   eta_t ~ N(0,1)

Twist
-----
  Gaussian twist psi_t(x) = N(x; mu_t, tau_t^2), initialised from the
  Kalman filter posterior.  The twisted proposal is the product of two
  Gaussians (transition prior x twist), which is available in closed form.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter
from src.baselines import bootstrap_pf, ffbsi_smoother

PHI     = 0.9
SIGMA_Q = 0.5
SIGMA_R = 1.0

T  = 10000
D  = 1
N  = 256

def kalman_filter(ys, phi, sigma_q, sigma_r):
    def step(carry, y):
        m, P = carry
        m_pred = phi * m
        P_pred = phi**2 * P + sigma_q**2
        S = P_pred + sigma_r**2
        K = P_pred / S
        m_new = m_pred + K * (y - m_pred)
        P_new = (1.0 - K) * P_pred
        return (m_new, P_new), (m_new, P_new)
    _, (means, variances) = jax.lax.scan(step, (0.0, 1.0), ys)
    return means, variances

def twisted_step_fn(x_prev, driver, params):
    """
    Twisted transition for a single particle (D-dimensional).

    The twisted proposal combines the transition prior N(phi*x, sigma_q^2)
    with a Gaussian twist N(mu_twist, tau_twist^2), giving a Gaussian
    posterior in closed form.
    """
    obs, noise, twist = driver
    phi     = params['phi']
    sigma_q = params['sigma_q']

    mu_twist  = twist['mean']                    # (D,)
    tau_twist = jnp.exp(twist['log_scale'])      # (D,)

    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)

    return mean_post + jnp.sqrt(var_post) * noise

def log_weight_fn(x_prev, x_next, obs, twist, params):
    """
    Incremental log importance weight:
        log w_t = log p(y_t|x_t) + log p(x_t|x_{t-1}) - log q_twist(x_t|x_{t-1})
    """
    phi     = params['phi']
    sigma_q = params['sigma_q']
    sigma_r = params['sigma_r']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])

    # log p(y_t | x_t)
    log_lik = (-0.5 * jnp.sum((obs - x_next)**2) / sigma_r**2
               - 0.5 * D * jnp.log(2 * jnp.pi * sigma_r**2))

    # log p(x_t | x_{t-1})
    log_trans = (-0.5 * jnp.sum((x_next - phi * x_prev)**2) / sigma_q**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_q**2))

    # log q_twist(x_t | x_{t-1})
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    log_prop   = (-0.5 * jnp.sum((x_next - mean_post)**2 / var_post)
                  - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * var_post)))

    return log_lik + log_trans - log_prop

master_key = jr.PRNGKey(0)
key, k1, k2, k3 = jr.split(master_key, 4)

x0 = jr.normal(k1) * 1.0

def sim_step(x, noise_obs):
    q_noise, r_noise = noise_obs
    x_new = PHI * x + SIGMA_Q * q_noise
    y     = x_new + SIGMA_R * r_noise
    return x_new, (x_new, y)

q_noise = jr.normal(k2, (T,))
r_noise = jr.normal(k3, (T,))
_, (true_xs, ys) = jax.lax.scan(sim_step, x0, (q_noise, r_noise))
ys_2d = ys[:, None]  # (T, 1)

kf_means, kf_vars = kalman_filter(ys, PHI, SIGMA_Q, SIGMA_R)

twist_params = {
    'mean':      kf_means[:, None],                  # (T, D)
    'log_scale': 0.5 * jnp.log(kf_vars)[:, None],   # (T, D)
}

params = {
    'phi':     PHI,
    'sigma_q': SIGMA_Q,
    'sigma_r': SIGMA_R,
}

tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)

key, tape_key = jr.split(key)
tape = tpf.make_tape(ys_2d, twist_params, tape_key, N)

key, s0_key = jr.split(key)
x0s = jr.normal(s0_key, (N, D))

run_seq = jax.jit(tpf.run_sequential)
_ = run_seq(x0s, tape, params)  # warmup

t0 = time.time()
states_seq = run_seq(x0s, tape, params).block_until_ready()
t_seq = time.time() - t0

seq_means = states_seq.mean(axis=0)[:, 0]  # (T,)

key, deer_key = jr.split(key)
params['key'] = deer_key

run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
_ = run_par(x0s, tape, params, max_iter=100)  # warmup

t0 = time.time()
states_par, iters = run_par(x0s, tape, params, max_iter=100)
states_par.block_until_ready()
t_par = time.time() - t0

par_means = states_par.mean(axis=0)[:, 0]  # (T,)
deer_iters = int(iters[0])

log_ws = jax.jit(tpf.compute_log_weights)(x0s, states_par, tape, params)
cum_log_ws = jnp.cumsum(log_ws, axis=1)                          # (N, T)
log_norm   = jax.scipy.special.logsumexp(cum_log_ws, axis=0)     # (T,)
norm_ws    = jnp.exp(cum_log_ws - log_norm[None, :])             # (N, T)
weighted_par_means = jnp.sum(
    norm_ws[:, :, None] * states_par, axis=0
)[:, 0]  # (T,)

rmse_seq = jnp.sqrt(jnp.mean((seq_means - kf_means)**2))
rmse_par = jnp.sqrt(jnp.mean((par_means - kf_means)**2))
rmse_par_w = jnp.sqrt(jnp.mean((weighted_par_means - kf_means)**2))

print(f"T={T}, N={N} particles, D={D}")
print(f"Sequential twisted PF:  time={t_seq:.3f}s   RMSE vs Kalman={rmse_seq:.4f}")
print(f"Parallel   twisted PF:  time={t_par:.3f}s   RMSE vs Kalman={rmse_par:.4f}"
      f"  ({deer_iters} DEER iters)")
print(f"  (weighted means):                         RMSE vs Kalman={rmse_par_w:.4f}")
print(f"Speedup: {t_seq / t_par:.2f}x")

# Baselines: Bootstrap PF + FFBSi smoother
def lgssm_transition_mean(x_prev, p):
    return p['phi'] * x_prev

def lgssm_log_lik(y_t, x_t, p):
    return -0.5 * jnp.sum((y_t - x_t)**2) / p['sigma_r']**2

key, bpf_key, ffbsi_key, init_key = jr.split(key, 4)
x0_particles = jr.normal(init_key, (N, D)) * 1.0

bpf_params = {'phi': PHI, 'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R}

run_bpf = jax.jit(bootstrap_pf, static_argnames=("transition_mean", "log_lik"))
_ = run_bpf(ys_2d, x0_particles, lgssm_transition_mean, lgssm_log_lik,
            SIGMA_Q, bpf_params, bpf_key)

t0 = time.time()
bpf_filt, bpf_parts, bpf_log_w, bpf_ess, bpf_log_ml = run_bpf(
    ys_2d, x0_particles, lgssm_transition_mean, lgssm_log_lik,
    SIGMA_Q, bpf_params, bpf_key)
bpf_filt.block_until_ready()
t_bpf = time.time() - t0
rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_filt[:, 0] - kf_means)**2)))
print(f"\nBootstrap PF (filter):  time={t_bpf:.3f}s   RMSE vs Kalman={rmse_bpf:.4f}"
      f"  ESS mean={float(bpf_ess.mean()):.0f}/{N}")

M_BACKWARD = N
run_ffbsi = jax.jit(ffbsi_smoother, static_argnames=("M", "transition_mean"))
_ = run_ffbsi(bpf_parts, bpf_log_w, lgssm_transition_mean,
              SIGMA_Q, bpf_params, ffbsi_key, M=M_BACKWARD)

t0 = time.time()
bps_trajs = run_ffbsi(bpf_parts, bpf_log_w, lgssm_transition_mean,
                       SIGMA_Q, bpf_params, ffbsi_key, M=M_BACKWARD)
bps_trajs.block_until_ready()
t_bps = time.time() - t0
bps_means = bps_trajs.mean(axis=0)[:, 0]
rmse_bps = float(jnp.sqrt(jnp.mean((bps_means - kf_means)**2)))
print(f"Bootstrap PS (FFBSi):   time={t_bps:.3f}s   RMSE vs Kalman={rmse_bps:.4f}"
      f"  (M={M_BACKWARD})")
print(f"Twisted PF vs FFBSi:    RMSE  {float(rmse_par):.4f} vs {rmse_bps:.4f}"
      f"   time  {t_par:.3f}s vs {t_bps:.3f}s   speedup {t_bps/t_par:.1f}x")

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

t_idx = jnp.arange(T)

ax = axes[0]
ax.plot(t_idx, true_xs, "k-", lw=1,   label="true state", alpha=0.7)
ax.plot(t_idx, ys,      "k.", ms=2,   label="observations", alpha=0.3)
ax.plot(t_idx, kf_means, "g-", lw=1.5, label="Kalman mean")
ax.plot(t_idx, seq_means, "b--", lw=1.5,
        label=f"seq twisted PF (RMSE={rmse_seq:.3f})")
ax.legend(fontsize=9)
ax.set_title("Sequential twisted particle filter vs Kalman")

ax = axes[1]
ax.plot(t_idx, true_xs,  "k-", lw=1, alpha=0.7)
ax.plot(t_idx, kf_means, "g-", lw=1.5, label="Kalman mean")
ax.plot(t_idx, par_means, "r--", lw=1.5,
        label=f"parallel twisted PF ({deer_iters} iters, RMSE={rmse_par:.3f})")
ax.legend(fontsize=9)
ax.set_title(f"Parallel twisted PF (per-particle DEER) vs Kalman — {t_seq/t_par:.1f}x speedup")
ax.set_xlabel("timestep")

plt.tight_layout()
plt.savefig("twisted_pf_linear_gaussian.png", dpi=150)
plt.show()
