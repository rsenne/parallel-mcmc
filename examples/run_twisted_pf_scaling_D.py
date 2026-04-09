"""
run_twisted_pf_scaling_D.py

Dimension scaling study for the twisted parallel particle smoother.

Tests two regimes:
  (a) Independent dimensions — diagonal Jacobian is EXACT; isolates
      computational overhead scaling with D.
  (b) Coupled dimensions — tridiagonal transition matrix; diagonal
      Jacobian is approximate; shows accuracy/speed tradeoff.

Model:  x_t = A x_{t-1} + sigma_q eps_t,  y_t = x_t + sigma_r eta_t
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter

PHI     = 0.9
ALPHA   = 0.05    # off-diagonal coupling strength (coupled case)
SIGMA_Q = 0.5
SIGMA_R = 1.0
T       = 4096
N       = 256

D_VALUES = [1, 2, 4, 8, 16, 32]

def kalman_smoother(ys, A, Q, R, m0, P0):
    D = A.shape[0]
    I = jnp.eye(D)

    def fwd(carry, y):
        m, P = carry
        m_pred = A @ m
        P_pred = A @ P @ A.T + Q
        S = P_pred + R
        K = jnp.linalg.solve(S.T, P_pred.T).T
        m_new = m_pred + K @ (y - m_pred)
        P_new = (I - K) @ P_pred
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new)

    _, (mp, Pp, mf, Pf) = jax.lax.scan(fwd, (m0, P0), ys)

    def bwd(carry, t):
        ms, Ps = carry
        G = jnp.linalg.solve(Pp[t + 1].T, (Pf[t] @ A.T).T).T
        ms_new = mf[t] + G @ (ms - mp[t + 1])
        Ps_new = Pf[t] + G @ (Ps - Pp[t + 1]) @ G.T
        return (ms_new, Ps_new), (ms_new, Ps_new)

    ts = jnp.arange(ys.shape[0] - 2, -1, -1)
    _, (ms_rev, Ps_rev) = jax.lax.scan(bwd, (mf[-1], Pf[-1]), ts)

    sm = jnp.concatenate([ms_rev[::-1], mf[-1:]])
    sP = jnp.concatenate([Ps_rev[::-1], Pf[-1:]])
    return sm, sP

def make_twisted_step(A_mat):
    """Build a twisted step function that closes over A."""
    def twisted_step_fn(x_prev, driver, params):
        obs, noise, twist = driver
        sigma_q   = params['sigma_q']
        mu_twist  = twist['mean']
        tau_twist = jnp.exp(twist['log_scale'])

        f_prev     = A_mat @ x_prev
        prec_q     = 1.0 / sigma_q**2
        prec_twist = 1.0 / tau_twist**2
        var_post   = 1.0 / (prec_q + prec_twist)
        mean_post  = var_post * (prec_q * f_prev + prec_twist * mu_twist)
        return mean_post + jnp.sqrt(var_post) * noise
    return twisted_step_fn


def log_weight_fn(x_prev, x_next, obs, twist, params):
    sigma_q   = params['sigma_q']
    sigma_r   = params['sigma_r']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])

    # For log weights, we need A @ x_prev.  We store it in params.
    f_prev = params['_A'] @ x_prev

    log_lik   = -0.5 * jnp.sum((obs - x_next)**2) / sigma_r**2
    log_trans = -0.5 * jnp.sum((x_next - f_prev)**2) / sigma_q**2

    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * f_prev + prec_twist * mu_twist)
    log_prop   = -0.5 * jnp.sum((x_next - mean_post)**2 / var_post)
    log_norm   = 0.5 * jnp.sum(jnp.log(var_post) - 2 * jnp.log(sigma_q))
    return log_lik + log_trans - log_prop + log_norm

def make_independent_A(D):
    return PHI * jnp.eye(D)

def make_coupled_A(D):
    A = PHI * jnp.eye(D)
    idx = jnp.arange(D - 1)
    A = A.at[idx, idx + 1].set(ALPHA)
    A = A.at[idx + 1, idx].set(ALPHA)
    return A

def run_one(D, A, label, key):
    Q = SIGMA_Q**2 * jnp.eye(D)
    R = SIGMA_R**2 * jnp.eye(D)

    # simulate
    key, k1, k2 = jr.split(key, 3)

    def sim_step(x, noise_obs):
        q, r = noise_obs
        xn = A @ x + SIGMA_Q * q
        return xn, (xn, xn + SIGMA_R * r)

    x0_true = jnp.zeros(D)
    _, (true_xs, ys) = jax.lax.scan(
        sim_step, x0_true,
        (jr.normal(k1, (T, D)), jr.normal(k2, (T, D)))
    )

    # Kalman smoother
    sm, sP = jax.jit(kalman_smoother)(ys, A, Q, R, jnp.zeros(D), jnp.eye(D))
    sm_scales = jnp.sqrt(jax.vmap(jnp.diag)(sP))

    twist_params = {
        'mean':      sm,
        'log_scale': jnp.log(sm_scales + 1e-8),
    }

    # filter
    step_fn = make_twisted_step(A)
    tpf = TwistedParallelParticleFilter(step_fn, log_weight_fn, D=D)
    p = {'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R, '_A': A}

    key, tk, sk = jr.split(key, 3)
    tape = tpf.make_tape(ys, twist_params, tk, N)
    x0s  = jr.normal(sk, (N, D))

    # sequential
    run_seq = jax.jit(tpf.run_sequential)
    run_seq(x0s, tape, p).block_until_ready()
    times_s = []
    for _ in range(5):
        t0 = time.perf_counter()
        ss = run_seq(x0s, tape, p).block_until_ready()
        times_s.append(time.perf_counter() - t0)
    t_seq = sorted(times_s)[2]
    rmse_seq = float(jnp.sqrt(jnp.mean((ss.mean(0) - sm)**2)))

    # parallel
    key, dk = jr.split(key)
    p['key'] = dk
    ns = min(D, 10)   # num_samples scales with D
    run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
    sp_, it_ = run_par(x0s, tape, p, max_iter=100, num_samples=ns)
    sp_.block_until_ready()
    times_p = []
    for _ in range(5):
        t0 = time.perf_counter()
        sp_, it_ = run_par(x0s, tape, p, max_iter=100, num_samples=ns)
        sp_.block_until_ready()
        times_p.append(time.perf_counter() - t0)
    t_par = sorted(times_p)[2]
    rmse_par   = float(jnp.sqrt(jnp.mean((sp_.mean(0) - sm)**2)))
    deer_iters = int(it_[0])
    speedup    = t_seq / t_par if t_par > 0 else float('inf')

    print(f"  D={D:>3}  {label:>12}  seq={t_seq:.4f}s  par={t_par:.4f}s"
          f"  speedup={speedup:>6.1f}x  iters={deer_iters:>3}"
          f"  RMSE_seq={rmse_seq:.4f}  RMSE_par={rmse_par:.4f}")

    return dict(D=D, label=label, t_seq=t_seq, t_par=t_par,
                speedup=speedup, deer_iters=deer_iters,
                rmse_seq=rmse_seq, rmse_par=rmse_par)


key = jr.PRNGKey(123)
results_ind = []
results_cpl = []

print(f"T={T}, N={N}\n")

for D in D_VALUES:
    key, k1, k2 = jr.split(key, 3)
    results_ind.append(run_one(D, make_independent_A(D), "independent", k1))
    results_cpl.append(run_one(D, make_coupled_A(D), "coupled", k2))

print("\n" + "=" * 90)
print(f"{'D':>4} {'type':>12} {'t_seq':>10} {'t_par':>10} {'speedup':>10}"
      f" {'iters':>6} {'RMSE_seq':>10} {'RMSE_par':>10}")
print("-" * 90)
for r in results_ind + results_cpl:
    print(f"{r['D']:>4} {r['label']:>12} {r['t_seq']:>10.4f}"
          f" {r['t_par']:>10.4f} {r['speedup']:>9.1f}x"
          f" {r['deer_iters']:>6} {r['rmse_seq']:>10.4f}"
          f" {r['rmse_par']:>10.4f}")
print("=" * 90)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

Ds_i = [r['D'] for r in results_ind]
Ds_c = [r['D'] for r in results_cpl]

ax = axes[0]
ax.plot(Ds_i, [r['t_seq'] for r in results_ind], 'b^-', label='seq (indep)')
ax.plot(Ds_i, [r['t_par'] for r in results_ind], 'ro-', label='par (indep)')
ax.plot(Ds_c, [r['t_par'] for r in results_cpl], 'rs--', label='par (coupled)')
ax.set_xlabel('State dimension D')
ax.set_ylabel('Time (s)')
ax.set_title('Wall-clock time vs D')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(Ds_i, [r['speedup'] for r in results_ind], 'go-', lw=2, label='independent')
ax.plot(Ds_c, [r['speedup'] for r in results_cpl], 'gs--', lw=2, label='coupled')
ax.set_xlabel('State dimension D')
ax.set_ylabel('Speedup')
ax.set_title('Parallel speedup vs D')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(Ds_i, [r['deer_iters'] for r in results_ind], 'mo-', lw=2, label='independent')
ax.plot(Ds_c, [r['deer_iters'] for r in results_cpl], 'ms--', lw=2, label='coupled')
ax.set_xlabel('State dimension D')
ax.set_ylabel('DEER iterations')
ax.set_title('DEER iterations vs D')
ax.legend(fontsize=8)
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Dimension scaling — T={T}, N={N}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('twisted_pf_scaling_D.png', dpi=150, bbox_inches='tight')
plt.show()
