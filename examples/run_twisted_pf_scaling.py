"""
run_twisted_pf_scaling.py

Scaling study for the twisted parallel particle smoother on a 1-D LGSSM.

Uses Kalman smoother (RTS) backward pass to derive optimal twist functions,
then benchmarks sequential vs parallel (per-particle DEER) execution across
a range of sequence lengths T.

Produces:
  - Table of timing results
  - Log-log plot of time vs T
  - Speedup vs T
  - DEER iterations vs T
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter

PHI     = 0.9
SIGMA_Q = 0.5
SIGMA_R = 1.0
D       = 1
N       = 256
N_RUNS  = 5   # timing runs per T (take median)

T_VALUES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

def kalman_filter(ys, phi, sigma_q, sigma_r):
    def step(carry, y):
        m, P = carry
        m_pred = phi * m
        P_pred = phi**2 * P + sigma_q**2
        S = P_pred + sigma_r**2
        K = P_pred / S
        m_new = m_pred + K * (y - m_pred)
        P_new = (1.0 - K) * P_pred
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new)

    _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(
        step, (0.0, 1.0), ys
    )
    return m_filts, P_filts, m_preds, P_preds


def rts_smoother(m_filts, P_filts, m_preds, P_preds, phi):
    """Rauch-Tung-Striebel backward pass."""
    T = m_filts.shape[0]

    def step(carry, t):
        m_s, P_s = carry
        G = phi * P_filts[t] / P_preds[t + 1]   # smoother gain
        m_s_new = m_filts[t] + G * (m_s - m_preds[t + 1])
        P_s_new = P_filts[t] + G**2 * (P_s - P_preds[t + 1])
        return (m_s_new, P_s_new), (m_s_new, P_s_new)

    # backward from t = T-2 to 0 (last filtered = last smoothed)
    init = (m_filts[-1], P_filts[-1])
    ts = jnp.arange(T - 2, -1, -1)
    _, (ms_rev, Ps_rev) = jax.lax.scan(step, init, ts)

    # reverse and append last
    smooth_means = jnp.concatenate([ms_rev[::-1], m_filts[-1:]])
    smooth_vars  = jnp.concatenate([Ps_rev[::-1], P_filts[-1:]])
    return smooth_means, smooth_vars


def backward_information_filter_1d(ys, phi, sigma_q, sigma_r):
    """Optimal twist via backward information filter: psi_t(x_t) = p(y_{t:T}|x_t)."""
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2
    def step(carry, y):
        J, h = carry
        denom = 1.0 + sq * J
        J_pred = phi**2 * J / denom
        h_pred = phi * h / denom
        J_new = J_pred + prec_obs
        h_new = h_pred + y * prec_obs
        return (J_new, h_new), (J_new, h_new)
    J_T = prec_obs
    h_T = ys[-1] * prec_obs
    _, (Js_rev, hs_rev) = jax.lax.scan(step, (J_T, h_T), ys[-2::-1])
    Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_T])])
    hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_T])])
    return hs / Js, 1.0 / Js


def twisted_step_fn(x_prev, driver, params):
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
    phi     = params['phi']
    sigma_q = params['sigma_q']
    sigma_r = params['sigma_r']
    mu_twist  = twist['mean']
    tau_twist = jnp.exp(twist['log_scale'])
    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sigma_r**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_r**2))
    log_trans = (-0.5 * jnp.sum((x_next - phi * x_prev)**2) / sigma_q**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_q**2))
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    log_prop  = (-0.5 * jnp.sum((x_next - mean_post)**2 / var_post)
                 - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * var_post)))
    return log_lik + log_trans - log_prop


tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)
params_base = {'phi': PHI, 'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R}
results = []
key = jr.PRNGKey(42)

for T in T_VALUES:
    print(f"\n--- T={T} ---")

    # simulate
    key, k1, k2, k3 = jr.split(key, 4)
    x0_true = jr.normal(k1)

    def sim_step(x, noise_obs):
        q, r = noise_obs
        xn = PHI * x + SIGMA_Q * q
        return xn, (xn, xn + SIGMA_R * r)

    _, (true_xs, ys) = jax.lax.scan(
        sim_step, x0_true, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
    )
    ys_2d = ys[:, None]

    # Kalman smoother for RMSE baseline
    m_f, P_f, m_p, P_p = kalman_filter(ys, PHI, SIGMA_Q, SIGMA_R)
    sm_means, sm_vars = rts_smoother(m_f, P_f, m_p, P_p, PHI)

    # Optimal twist from backward information filter
    twist_means, twist_vars = backward_information_filter_1d(ys, PHI, SIGMA_Q, SIGMA_R)

    twist_params = {
        'mean':      twist_means[:, None],
        'log_scale': 0.5 * jnp.log(twist_vars)[:, None],
    }

    # tape & initial particles
    key, tk, sk = jr.split(key, 3)
    tape = tpf.make_tape(ys_2d, twist_params, tk, N)
    x0s  = jr.normal(sk, (N, D))

    params = {**params_base}
    run_seq = jax.jit(tpf.run_sequential)
    run_seq(x0s, tape, params).block_until_ready()        # compile

    seq_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        states_seq = run_seq(x0s, tape, params).block_until_ready()
        seq_times.append(time.perf_counter() - t0)
    t_seq = sorted(seq_times)[N_RUNS // 2]

    seq_means = states_seq.mean(axis=0)[:, 0]
    rmse_seq  = float(jnp.sqrt(jnp.mean((seq_means - sm_means)**2)))

    key, dk = jr.split(key)
    params['key'] = dk
    run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
    s, i = run_par(x0s, tape, params, max_iter=100)     # compile
    s.block_until_ready()

    par_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        states_par, iters = run_par(x0s, tape, params, max_iter=100)
        states_par.block_until_ready()
        par_times.append(time.perf_counter() - t0)
    t_par = sorted(par_times)[N_RUNS // 2]

    par_means  = states_par.mean(axis=0)[:, 0]
    rmse_par   = float(jnp.sqrt(jnp.mean((par_means - sm_means)**2)))
    deer_iters = int(iters[0])
    speedup    = t_seq / t_par if t_par > 0 else float('inf')

    print(f"  Sequential: {t_seq:.4f}s  RMSE={rmse_seq:.4f}")
    print(f"  Parallel:   {t_par:.4f}s  RMSE={rmse_par:.4f}  ({deer_iters} iters)")
    print(f"  Speedup:    {speedup:.1f}x")

    results.append(dict(
        T=T, t_seq=t_seq, t_par=t_par, rmse_seq=rmse_seq,
        rmse_par=rmse_par, deer_iters=deer_iters, speedup=speedup,
    ))

print("\n" + "=" * 85)
print(f"{'T':>8} {'t_seq (s)':>10} {'t_par (s)':>10} {'speedup':>10}"
      f" {'iters':>6} {'RMSE_seq':>10} {'RMSE_par':>10}")
print("-" * 85)
for r in results:
    print(f"{r['T']:>8} {r['t_seq']:>10.4f} {r['t_par']:>10.4f}"
          f" {r['speedup']:>9.1f}x {r['deer_iters']:>6}"
          f" {r['rmse_seq']:>10.4f} {r['rmse_par']:>10.4f}")
print("=" * 85)

Ts       = [r['T'] for r in results]
t_seqs   = [r['t_seq'] for r in results]
t_pars   = [r['t_par'] for r in results]
speedups = [r['speedup'] for r in results]
iters_l  = [r['deer_iters'] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.loglog(Ts, t_seqs, 'bo-', lw=2, label='Sequential')
ax.loglog(Ts, t_pars, 'ro-', lw=2, label='Parallel (DEER)')
ax.set_xlabel('Sequence length T')
ax.set_ylabel('Time (s)')
ax.set_title('Wall-clock time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogx(Ts, speedups, 'go-', lw=2)
ax.set_xlabel('Sequence length T')
ax.set_ylabel('Speedup')
ax.set_title('Parallel speedup')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.semilogx(Ts, iters_l, 'mo-', lw=2)
ax.set_xlabel('Sequence length T')
ax.set_ylabel('DEER iterations')
ax.set_title('DEER iterations')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Twisted parallel particle smoother — N={N} particles, D={D}',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('twisted_pf_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
