"""
run_twisted_pf_stochvol.py

Stochastic volatility model — a nonlinear SSM where the Kalman filter
cannot be applied.  Demonstrates:

  1. Learning twist (backward) parameters via gradient descent through
     the differentiable sequential particle smoother.
  2. Deploying the learned twists with the parallel DEER solver for
     fast inference (O(log T) depth).

Model
-----
  x_t = phi * x_{t-1} + sigma_q * eps_t          (log-volatility)
  y_t = beta * exp(x_t / 2) * eta_t              (observations)

The twisted proposal modifies the transition prior with a per-timestep
Gaussian "twist" that approximates the backward smoothing message.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter

PHI     = 0.95
SIGMA_Q = 0.2
BETA    = 1.0

T  = 1000
D  = 1
N  = 128

def twisted_step_fn(x_prev, driver, params):
    """Twisted transition (same form for any model with linear-Gaussian transition)."""
    obs, noise, twist = driver
    phi     = params['phi']
    sigma_q = params['sigma_q']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    return mean_post + jnp.sqrt(var_post) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    """Log importance weight — SV observation model."""
    phi     = params['phi']
    sigma_q = params['sigma_q']
    beta    = params['beta']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])

    # log p(y_t | x_t) — stochastic volatility likelihood
    log_lik = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(beta)
               - x_next[0] / 2.0
               - obs[0]**2 / (2.0 * beta**2 * jnp.exp(x_next[0])))

    # log p(x_t | x_{t-1})
    log_trans = (-0.5 * (x_next[0] - phi * x_prev[0])**2 / sigma_q**2
                 - 0.5 * jnp.log(2.0 * jnp.pi * sigma_q**2))

    # log q_twist(x_t | x_{t-1})
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    log_prop = (-0.5 * (x_next[0] - mean_post[0])**2 / var_post[0]
                - 0.5 * jnp.log(2.0 * jnp.pi * var_post[0]))

    return log_lik + log_trans - log_prop

def approx_backward_info_filter_sv(ys, phi, sigma_q, beta, n_iter=3):
    """
    Approximate backward information filter for the stochastic volatility model.

    Iteratively linearises the observation log-likelihood
        log p(y|x) = -0.5*log(2pi) - log(beta) - x/2 - y^2/(2*beta^2*exp(x))
    around the current backward message mean.

    Returns twist_means (T,) and twist_vars (T,).
    """
    T = ys.shape[0]
    sq2 = sigma_q**2

    def obs_info(y, x0):
        """Gaussian approx of observation info at linearisation point x0."""
        # Clamp x0 to avoid overflow in exp(-x0) for large negative x0
        x0_c = jnp.clip(x0, -20.0, 20.0)
        r = y**2 / (2.0 * beta**2 * jnp.exp(x0_c))
        r = jnp.clip(r, 1e-8, 100.0)             # clip once for consistency
        J_obs = r                                  # -d^2/dx^2 log p(y|x)
        g = -0.5 + r                              # d/dx log p(y|x) at x0
        h_obs = g + J_obs * x0_c
        return J_obs, h_obs

    # Initialise twist means with stationary mean (zero)
    twist_means = jnp.zeros(T)

    for _ in range(n_iter):
        def step(carry, inputs):
            J, h = carry
            y_t, lin_pt = inputs
            # Predict backward through linear transition x_{t+1} = phi*x_t + noise
            denom = 1.0 + sq2 * J
            J_pred = phi**2 * J / denom
            h_pred = phi * h / denom
            # Add linearised observation
            J_obs, h_obs = obs_info(y_t, lin_pt)
            J_obs = jnp.maximum(J_obs, 1e-6)   # ensure positive
            return (J_pred + J_obs, h_pred + h_obs), (J_pred + J_obs, h_pred + h_obs)

        # Last timestep
        J_obs_T, h_obs_T = obs_info(ys[-1], twist_means[-1])
        J_obs_T = jnp.maximum(J_obs_T, 1e-6)
        J_T = J_obs_T
        h_T = h_obs_T

        _, (Js_rev, hs_rev) = jax.lax.scan(
            step, (J_T, h_T),
            (ys[-2::-1], twist_means[-2::-1]),
        )

        Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_T])])
        hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_T])])
        twist_means = hs / Js

    twist_vars = 1.0 / Js
    return twist_means, twist_vars


key = jr.PRNGKey(0)
key, k1, k2, k3 = jr.split(key, 4)
x0_true = jr.normal(k1) * 0.5

def sim_sv_step(x, noise_obs):
    q, r = noise_obs
    xn = PHI * x + SIGMA_Q * q
    y  = BETA * jnp.exp(xn / 2.0) * r
    return xn, (xn, y)

_, (true_xs, ys) = jax.lax.scan(
    sim_sv_step, x0_true, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
)
ys_2d = ys[:, None]

tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)
params = {'phi': PHI, 'sigma_q': SIGMA_Q, 'beta': BETA}

# Fixed particles and noise
key, s0_key, noise_key = jr.split(key, 3)
x0s   = jr.normal(s0_key, (N, D)) * 0.5
noise = jr.normal(noise_key, (N, T, D))

# Initialise twists from approximate backward info filter
print("Running approximate backward information filter ...")
bif_means, bif_vars = jax.jit(approx_backward_info_filter_sv, static_argnames=('n_iter',))(ys, PHI, SIGMA_Q, BETA, n_iter=10)
bif_rmse = float(jnp.sqrt(jnp.mean((bif_means - true_xs)**2)))
print(f"  BIF twist RMSE: {bif_rmse:.4f}")

twist_mean      = bif_means[:, None]              # (T, 1)
twist_log_scale = 0.5 * jnp.log(bif_vars)[:, None]  # (T, 1)


def loss_fn(twist_mean, twist_log_scale):
    twist_params = {'mean': twist_mean, 'log_scale': twist_log_scale}
    tape   = (ys_2d, noise, twist_params)
    states = tpf.run_sequential(x0s, tape, params)                 # (N, T, D)
    log_ws = tpf.compute_log_weights(x0s, states, tape, params)    # (N, T)
    total  = jnp.sum(log_ws, axis=1)                               # (N,)
    elbo   = jax.scipy.special.logsumexp(total) - jnp.log(float(N))
    return -elbo

grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

def adam_init(p1, p2):
    return (jnp.zeros_like(p1), jnp.zeros_like(p1),
            jnp.zeros_like(p2), jnp.zeros_like(p2), 0)

def adam_step(g1, g2, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m1, v1, m2, v2, t = state
    t  = t + 1
    m1 = b1 * m1 + (1 - b1) * g1
    v1 = b2 * v1 + (1 - b2) * g1**2
    m2 = b1 * m2 + (1 - b1) * g2
    v2 = b2 * v2 + (1 - b2) * g2**2
    u1 = -lr * (m1 / (1 - b1**t)) / (jnp.sqrt(v1 / (1 - b2**t)) + eps)
    u2 = -lr * (m2 / (1 - b2**t)) / (jnp.sqrt(v2 / (1 - b2**t)) + eps)
    return u1, u2, (m1, v1, m2, v2, t)

N_STEPS   = 500
opt_state = adam_init(twist_mean, twist_log_scale)
losses    = []

print(f"Learning twist parameters (T={T}, N={N}) ...")
t_learn_start = time.perf_counter()

for i in range(N_STEPS):
    loss, (g_mean, g_scale) = grad_fn(twist_mean, twist_log_scale)
    u_mean, u_scale, opt_state = adam_step(g_mean, g_scale, opt_state, lr=1e-2)
    twist_mean      = twist_mean + u_mean
    twist_log_scale = twist_log_scale + u_scale
    losses.append(float(loss))
    if (i + 1) % 25 == 0:
        print(f"  step {i+1:>4}/{N_STEPS}  neg-ELBO = {loss:.2f}")

t_learn = time.perf_counter() - t_learn_start
print(f"Learning complete in {t_learn:.1f}s\n")

twist_learned = {'mean': twist_mean, 'log_scale': twist_log_scale}
tape_learned  = (ys_2d, noise, twist_learned)

# Sequential
run_seq = jax.jit(tpf.run_sequential)
run_seq(x0s, tape_learned, params).block_until_ready()

t0 = time.perf_counter()
states_seq = run_seq(x0s, tape_learned, params).block_until_ready()
t_seq = time.perf_counter() - t0

# Parallel
key, dk = jr.split(key)
params['key'] = dk
run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
s, i_ = run_par(x0s, tape_learned, params, max_iter=100)
s.block_until_ready()

t0 = time.perf_counter()
states_par, iters = run_par(x0s, tape_learned, params, max_iter=100)
states_par.block_until_ready()
t_par = time.perf_counter() - t0
deer_iters = int(iters[0])

seq_means = states_seq.mean(axis=0)[:, 0]
par_means = states_par.mean(axis=0)[:, 0]

rmse_seq = float(jnp.sqrt(jnp.mean((seq_means - true_xs)**2)))
rmse_par = float(jnp.sqrt(jnp.mean((par_means - true_xs)**2)))

# Also run with NO twist (prior proposal) for reference
twist_none = {'mean': jnp.zeros((T, D)), 'log_scale': 3.0 * jnp.ones((T, D))}
tape_none  = (ys_2d, noise, twist_none)
states_none = run_seq(x0s, tape_none, params).block_until_ready()
none_means  = states_none.mean(axis=0)[:, 0]
rmse_none   = float(jnp.sqrt(jnp.mean((none_means - true_xs)**2)))

print(f"T={T}, N={N}")
print(f"No twist (prior):     RMSE vs true = {rmse_none:.4f}")
print(f"Learned twist (seq):  time={t_seq:.4f}s   RMSE vs true = {rmse_seq:.4f}")
print(f"Learned twist (par):  time={t_par:.4f}s   RMSE vs true = {rmse_par:.4f}"
      f"  ({deer_iters} DEER iters)")
print(f"Speedup: {t_seq / t_par:.1f}x")

# Baselines: Bootstrap PF + FFBSi smoother
from src.baselines import bootstrap_pf, ffbsi_smoother

def sv_transition_mean(x_prev, p):
    return p['phi'] * x_prev

def sv_log_lik(y_t, x_t, p):
    # y | x ~ N(0, beta^2 * exp(x))
    var = p['beta']**2 * jnp.exp(x_t)
    return -0.5 * jnp.sum(y_t**2 / var + jnp.log(2 * jnp.pi * var))

key, bpf_key, ffbsi_key, init_key = jr.split(key, 4)
x0_particles = jr.normal(init_key, (N, D)) * 0.5
bpf_params = {'phi': PHI, 'sigma_q': SIGMA_Q, 'beta': BETA}

run_bpf = jax.jit(bootstrap_pf,
                  static_argnames=("transition_mean", "log_lik"))
_ = run_bpf(ys_2d, x0_particles, sv_transition_mean, sv_log_lik,
            SIGMA_Q, bpf_params, bpf_key)

t0 = time.perf_counter()
bpf_filt, bpf_parts, bpf_log_w, bpf_ess, bpf_log_ml = run_bpf(
    ys_2d, x0_particles, sv_transition_mean, sv_log_lik,
    SIGMA_Q, bpf_params, bpf_key)
bpf_filt.block_until_ready()
t_bpf = time.perf_counter() - t0
rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_filt[:, 0] - true_xs)**2)))
print(f"\nBootstrap PF (filter):  time={t_bpf:.3f}s   RMSE vs true = {rmse_bpf:.4f}"
      f"  ESS mean={float(bpf_ess.mean()):.0f}/{N}")

M_BACKWARD = N
run_ffbsi = jax.jit(ffbsi_smoother,
                    static_argnames=("M", "transition_mean"))
_ = run_ffbsi(bpf_parts, bpf_log_w, sv_transition_mean,
              SIGMA_Q, bpf_params, ffbsi_key, M=M_BACKWARD)

t0 = time.perf_counter()
bps_trajs = run_ffbsi(bpf_parts, bpf_log_w, sv_transition_mean,
                       SIGMA_Q, bpf_params, ffbsi_key, M=M_BACKWARD)
bps_trajs.block_until_ready()
t_bps = time.perf_counter() - t0
bps_means = bps_trajs.mean(axis=0)[:, 0]
rmse_bps = float(jnp.sqrt(jnp.mean((bps_means - true_xs)**2)))
print(f"Bootstrap PS (FFBSi):   time={t_bps:.3f}s   RMSE vs true = {rmse_bps:.4f}"
      f"  (M={M_BACKWARD})")
print(f"Twisted PF vs FFBSi:    RMSE  {rmse_par:.4f} vs {rmse_bps:.4f}"
      f"   time  {t_par:.4f}s vs {t_bps:.3f}s   speedup {t_bps/t_par:.1f}x")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

t_idx = jnp.arange(T)

ax = axes[0, 0]
ax.plot(losses, 'b-', lw=1)
ax.set_xlabel('Optimisation step')
ax.set_ylabel('Negative ELBO')
ax.set_title('Twist learning')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(t_idx, true_xs, 'k-', lw=1, alpha=0.7, label='true $x_t$')
ax.plot(t_idx, twist_mean[:, 0], 'r-', lw=1, alpha=0.7, label='learned twist mean')
ax.fill_between(
    t_idx,
    twist_mean[:, 0] - jnp.exp(twist_log_scale[:, 0]),
    twist_mean[:, 0] + jnp.exp(twist_log_scale[:, 0]),
    color='r', alpha=0.15, label='twist $\\pm$ scale',
)
ax.set_xlabel('timestep')
ax.set_title('Learned backward messages (twists)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(t_idx, true_xs, 'k-', lw=1, alpha=0.7, label='true $x_t$')
ax.plot(t_idx, none_means, color='gray', lw=1, alpha=0.5,
        label=f'no twist (RMSE={rmse_none:.3f})')
ax.plot(t_idx, seq_means, 'b-', lw=1, alpha=0.8,
        label=f'learned twist seq (RMSE={rmse_seq:.3f})')
ax.plot(t_idx, par_means, 'r--', lw=1, alpha=0.8,
        label=f'learned twist par (RMSE={rmse_par:.3f})')
ax.set_xlabel('timestep')
ax.set_title('Smoothing estimates')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(t_idx, ys, 'k-', lw=0.5, alpha=0.7)
ax.set_xlabel('timestep')
ax.set_ylabel('$y_t$')
ax.set_title('Observations (stochastic volatility)')
ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Stochastic volatility — T={T}, N={N}, '
    f'{deer_iters} DEER iters, {t_seq/t_par:.0f}x speedup',
    fontsize=13, y=1.02,
)
plt.tight_layout()
plt.savefig('twisted_pf_stochvol.png', dpi=150, bbox_inches='tight')
plt.show()
