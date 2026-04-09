"""
run_twisted_pf_lorenz63.py

Parallel particle smoother on the Lorenz-63 system — a chaotic nonlinear
SSM where no analytic filter exists.

Model
-----
  Lorenz-63 (Euler-discretised):
    x_t = f(x_{t-1}) + sigma_q * eps_t,    eps_t ~ N(0, I_3)
    y_t = x_t + sigma_r * eta_t,           eta_t ~ N(0, I_3)

  f(x) = x + dt * [sigma*(x2-x1), x1*(rho-x3)-x2, x1*x2-beta*x3]

Pipeline
--------
  1. Initialise twists from an EKF smoother (cheap, approximate).
  2. Refine twists via gradient descent on the ELBO (differentiable
     sequential PF).
  3. Deploy the learned twists with the parallel DEER solver.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter

SIGMA_L = 10.0
RHO     = 28.0
BETA_L  = 8.0 / 3.0
DT      = 0.01

SIGMA_Q = 0.5     # process noise std (per dimension)
SIGMA_R = 2.0     # observation noise std (per dimension)

T  = 500
D  = 3
N  = 256

def lorenz_deriv(x):
    return jnp.array([
        SIGMA_L * (x[1] - x[0]),
        x[0] * (RHO - x[2]) - x[1],
        x[0] * x[1] - BETA_L * x[2],
    ])

def lorenz_step(x):
    """Euler step."""
    return x + DT * lorenz_deriv(x)

def lorenz_jacobian(x):
    """Jacobian of the Euler-discretised Lorenz step."""
    J = jnp.array([
        [-SIGMA_L,  SIGMA_L,  0.0],
        [RHO - x[2], -1.0,   -x[0]],
        [x[1],        x[0],  -BETA_L],
    ])
    return jnp.eye(D) + DT * J


def ekf_smoother(ys, x0, sigma_q, sigma_r):
    """Extended Kalman smoother for Lorenz-63."""
    Q = sigma_q**2 * jnp.eye(D)
    R = sigma_r**2 * jnp.eye(D)

    # ── forward EKF ──
    def fwd_step(carry, y):
        m, P = carry
        m_pred = lorenz_step(m)
        J = lorenz_jacobian(m)
        P_pred = J @ P @ J.T + Q
        S = P_pred + R
        K = jnp.linalg.solve(S.T, P_pred.T).T      # P_pred @ S^{-1}
        m_new = m_pred + K @ (y - m_pred)
        P_new = (jnp.eye(D) - K) @ P_pred
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new)

    P0 = jnp.eye(D)
    _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(
        fwd_step, (x0, P0), ys
    )

    def bwd_step(carry, t):
        m_s, P_s = carry
        J = lorenz_jacobian(m_filts[t])
        G = jnp.linalg.solve(P_preds[t + 1].T, (P_filts[t] @ J.T).T).T
        m_s_new = m_filts[t] + G @ (m_s - m_preds[t + 1])
        P_s_new = P_filts[t] + G @ (P_s - P_preds[t + 1]) @ G.T
        return (m_s_new, P_s_new), (m_s_new, P_s_new)

    ts = jnp.arange(T - 2, -1, -1)
    _, (ms_rev, Ps_rev) = jax.lax.scan(
        bwd_step, (m_filts[-1], P_filts[-1]), ts
    )

    smooth_means = jnp.concatenate([ms_rev[::-1], m_filts[-1:]])
    smooth_covs  = jnp.concatenate([Ps_rev[::-1], P_filts[-1:]])
    return smooth_means, smooth_covs

def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    sigma_q   = params['sigma_q']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])

    f_prev     = lorenz_step(x_prev)
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * f_prev + prec_twist * mu_twist)
    return mean_post + jnp.sqrt(var_post) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    sigma_q   = params['sigma_q']
    sigma_r   = params['sigma_r']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])

    f_prev = lorenz_step(x_prev)

    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sigma_r**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_r**2))
    log_trans = (-0.5 * jnp.sum((x_next - f_prev)**2) / sigma_q**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_q**2))

    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * f_prev + prec_twist * mu_twist)
    log_prop   = (-0.5 * jnp.sum((x_next - mean_post)**2 / var_post)
                  - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * var_post)))
    return log_lik + log_trans - log_prop


key = jr.PRNGKey(0)
key, k1, k2, k3 = jr.split(key, 4)

# start on the attractor
x0_true = jnp.array([1.0, 1.0, 1.0])
# burn-in
def burn_step(x, _):
    return lorenz_step(x), None
x0_true, _ = jax.lax.scan(burn_step, x0_true, None, length=5000)

def sim_step(x, noise_obs):
    q, r = noise_obs
    xn = lorenz_step(x) + SIGMA_Q * q
    y  = xn + SIGMA_R * r
    return xn, (xn, y)

_, (true_xs, ys) = jax.lax.scan(
    sim_step, x0_true,
    (jr.normal(k2, (T, D)), jr.normal(k3, (T, D)))
)

print("Running EKF smoother for twist initialisation ...")
ekf_means, ekf_covs = jax.jit(ekf_smoother)(ys, x0_true, SIGMA_Q, SIGMA_R)
# diagonal of smoother covariances as twist scales
ekf_scales = jnp.sqrt(jax.vmap(jnp.diag)(ekf_covs))          # (T, D)

twist_mean      = ekf_means.copy()                             # (T, D)
twist_log_scale = jnp.log(ekf_scales + 1e-6)                   # (T, D)

ekf_rmse = float(jnp.sqrt(jnp.mean((ekf_means - true_xs)**2)))
print(f"  EKF smoother RMSE vs true: {ekf_rmse:.4f}")

tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)
params = {'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R}

key, s0_key, noise_key = jr.split(key, 3)
x0s   = x0_true[None, :] + 0.5 * jr.normal(s0_key, (N, D))
noise = jr.normal(noise_key, (N, T, D))

def loss_fn(twist_mean, twist_log_scale):
    twist_params = {'mean': twist_mean, 'log_scale': twist_log_scale}
    tape   = (ys, noise, twist_params)
    states = tpf.run_sequential(x0s, tape, params)
    log_ws = tpf.compute_log_weights(x0s, states, tape, params)
    total  = jnp.sum(log_ws, axis=1)
    elbo   = jax.scipy.special.logsumexp(total) - jnp.log(float(N))
    return -elbo

grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

def adam_init(p1, p2):
    return (jnp.zeros_like(p1), jnp.zeros_like(p1),
            jnp.zeros_like(p2), jnp.zeros_like(p2), 0)

def adam_step(g1, g2, state, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
    m1, v1, m2, v2, t = state
    t  = t + 1
    m1 = b1 * m1 + (1 - b1) * g1;  v1 = b2 * v1 + (1 - b2) * g1**2
    m2 = b1 * m2 + (1 - b1) * g2;  v2 = b2 * v2 + (1 - b2) * g2**2
    u1 = -lr * (m1 / (1 - b1**t)) / (jnp.sqrt(v1 / (1 - b2**t)) + eps)
    u2 = -lr * (m2 / (1 - b2**t)) / (jnp.sqrt(v2 / (1 - b2**t)) + eps)
    return u1, u2, (m1, v1, m2, v2, t)

N_STEPS   = 150
opt_state = adam_init(twist_mean, twist_log_scale)
losses    = []

print(f"\nLearning twist parameters (T={T}, D={D}, N={N}) ...")
t_learn_start = time.perf_counter()

for i in range(N_STEPS):
    loss, (g_mean, g_scale) = grad_fn(twist_mean, twist_log_scale)
    u_mean, u_scale, opt_state = adam_step(g_mean, g_scale, opt_state)
    twist_mean      = twist_mean + u_mean
    twist_log_scale = twist_log_scale + u_scale
    losses.append(float(loss))
    if (i + 1) % 25 == 0:
        print(f"  step {i+1:>4}/{N_STEPS}  neg-ELBO = {loss:.2f}")

t_learn = time.perf_counter() - t_learn_start
print(f"Learning complete in {t_learn:.1f}s\n")

twist_learned = {'mean': twist_mean, 'log_scale': twist_log_scale}
tape_learned  = (ys, noise, twist_learned)

run_seq = jax.jit(tpf.run_sequential)
run_seq(x0s, tape_learned, params).block_until_ready()
t0 = time.perf_counter()
states_seq = run_seq(x0s, tape_learned, params).block_until_ready()
t_seq = time.perf_counter() - t0

key, dk = jr.split(key)
params['key'] = dk
run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
s, i_ = run_par(x0s, tape_learned, params, max_iter=100, num_samples=5)
s.block_until_ready()
t0 = time.perf_counter()
states_par, iters = run_par(x0s, tape_learned, params, max_iter=100, num_samples=5)
states_par.block_until_ready()
t_par = time.perf_counter() - t0
deer_iters = int(iters[0])

seq_means = states_seq.mean(axis=0)       # (T, D)
par_means = states_par.mean(axis=0)       # (T, D)
rmse_seq  = float(jnp.sqrt(jnp.mean((seq_means - true_xs)**2)))
rmse_par  = float(jnp.sqrt(jnp.mean((par_means - true_xs)**2)))

# ESS from cumulative weights
log_ws     = jax.jit(tpf.compute_log_weights)(x0s, states_par, tape_learned, params)
cum_log_ws = jnp.cumsum(log_ws, axis=1)
log_norm   = jax.scipy.special.logsumexp(cum_log_ws, axis=0)
norm_ws    = jnp.exp(cum_log_ws - log_norm[None, :])
ess        = 1.0 / jnp.sum(norm_ws**2, axis=0)   # (T,)
mean_ess   = float(jnp.mean(ess))

print(f"T={T}, D={D}, N={N}")
print(f"EKF smoother:         RMSE vs true = {ekf_rmse:.4f}")
print(f"Learned twist (seq):  time={t_seq:.4f}s   RMSE vs true = {rmse_seq:.4f}")
print(f"Learned twist (par):  time={t_par:.4f}s   RMSE vs true = {rmse_par:.4f}"
      f"  ({deer_iters} DEER iters)")
print(f"Mean ESS: {mean_ess:.1f} / {N}")
print(f"Speedup: {t_seq / t_par:.1f}x")

fig = plt.figure(figsize=(16, 10))

t_idx = jnp.arange(T)
dim_names = ['x', 'y', 'z']

# Row 1: three state dimensions
for d in range(3):
    ax = fig.add_subplot(3, 3, d + 1)
    ax.plot(t_idx, true_xs[:, d], 'k-', lw=1, alpha=0.7, label='true')
    ax.plot(t_idx, ekf_means[:, d], 'g-', lw=0.8, alpha=0.6, label='EKF')
    ax.plot(t_idx, par_means[:, d], 'r-', lw=0.8, alpha=0.8, label='twisted PF')
    ax.set_ylabel(f'${dim_names[d]}$')
    if d == 0:
        ax.legend(fontsize=8)
    ax.set_title(f'State ${dim_names[d]}$')
    ax.grid(True, alpha=0.3)

# Row 2: loss curve, ESS, 3D attractor
ax = fig.add_subplot(3, 3, 4)
ax.plot(losses, 'b-', lw=1)
ax.set_xlabel('Step')
ax.set_ylabel('Neg-ELBO')
ax.set_title('Twist learning')
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 5)
ax.plot(t_idx, ess, 'b-', lw=0.8)
ax.axhline(N, color='k', ls='--', lw=0.5, label=f'N={N}')
ax.set_xlabel('Timestep')
ax.set_ylabel('ESS')
ax.set_title(f'Effective sample size (mean={mean_ess:.0f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(3, 3, 6, projection='3d')
ax.plot(true_xs[:, 0], true_xs[:, 1], true_xs[:, 2],
        'k-', lw=0.5, alpha=0.5, label='true')
ax.plot(par_means[:, 0], par_means[:, 1], par_means[:, 2],
        'r-', lw=0.5, alpha=0.7, label='twisted PF')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Lorenz attractor')
ax.legend(fontsize=8)

# Row 3: twist means vs true for each dimension
for d in range(3):
    ax = fig.add_subplot(3, 3, 7 + d)
    ax.plot(t_idx, true_xs[:, d], 'k-', lw=1, alpha=0.7, label='true')
    ax.plot(t_idx, twist_mean[:, d], 'r-', lw=0.8, alpha=0.7, label='twist mean')
    scale = jnp.exp(twist_log_scale[:, d])
    ax.fill_between(t_idx, twist_mean[:, d] - scale,
                    twist_mean[:, d] + scale, color='r', alpha=0.1)
    ax.set_xlabel('Timestep')
    ax.set_title(f'Learned twist ${dim_names[d]}$')
    if d == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Lorenz-63 parallel particle smoother — T={T}, D={D}, N={N}, '
    f'{deer_iters} DEER iters, {t_seq/t_par:.0f}x speedup',
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig('twisted_pf_lorenz63.png', dpi=150, bbox_inches='tight')
plt.show()
