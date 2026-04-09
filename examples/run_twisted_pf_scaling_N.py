"""
run_twisted_pf_scaling_N.py

Particle count (N) scaling study for the twisted parallel particle smoother.
Sweeps N at fixed T=4096, D=1 (LGSSM with RTS smoother twists).
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
T       = 4096
D       = 1
N_RUNS  = 5

N_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]

def kalman_smoother_1d(ys, phi, sigma_q, sigma_r):
    def fwd(carry, y):
        m, P = carry
        mp = phi * m
        Pp = phi**2 * P + sigma_q**2
        S  = Pp + sigma_r**2
        K  = Pp / S
        mf = mp + K * (y - mp)
        Pf = (1 - K) * Pp
        return (mf, Pf), (mp, Pp, mf, Pf)

    _, (mp, Pp, mf, Pf) = jax.lax.scan(fwd, (0.0, 1.0), ys)

    def bwd(carry, t):
        ms, Ps = carry
        G  = phi * Pf[t] / Pp[t + 1]
        ms_new = mf[t] + G * (ms - mp[t + 1])
        Ps_new = Pf[t] + G**2 * (Ps - Pp[t + 1])
        return (ms_new, Ps_new), (ms_new, Ps_new)

    ts = jnp.arange(ys.shape[0] - 2, -1, -1)
    _, (ms_r, Ps_r) = jax.lax.scan(bwd, (mf[-1], Pf[-1]), ts)
    sm = jnp.concatenate([ms_r[::-1], mf[-1:]])
    sP = jnp.concatenate([Ps_r[::-1], Pf[-1:]])
    return sm, sP

def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi     = params['phi']
    sigma_q = params['sigma_q']
    mu      = twist['mean']
    tau     = jnp.exp(twist['log_scale'])
    pq = 1.0 / sigma_q**2
    pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise

def log_weight_fn(x_prev, x_next, obs, twist, params):
    phi = params['phi']; sq = params['sigma_q']; sr = params['sigma_r']
    mu = twist['mean']; tau = jnp.exp(twist['log_scale'])
    ll = -0.5 * jnp.sum((obs - x_next)**2) / sr**2
    lt = -0.5 * jnp.sum((x_next - phi * x_prev)**2) / sq**2
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    lp = -0.5 * jnp.sum((x_next - mp)**2 / vp)
    ln = 0.5 * jnp.sum(jnp.log(vp) - 2 * jnp.log(sq))
    return ll + lt - lp + ln

key = jr.PRNGKey(7)
key, k1, k2, k3 = jr.split(key, 4)

def sim_step(x, qr):
    xn = PHI * x + SIGMA_Q * qr[0]
    return xn, (xn, xn + SIGMA_R * qr[1])

_, (true_xs, ys) = jax.lax.scan(
    sim_step, jr.normal(k1),
    (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
)
ys_2d = ys[:, None]

sm, sP = kalman_smoother_1d(ys, PHI, SIGMA_Q, SIGMA_R)
twist_params = {'mean': sm[:, None], 'log_scale': 0.5 * jnp.log(sP)[:, None]}
params_base = {'phi': PHI, 'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R}
results = []

print(f"T={T}, D={D}\n")

for N in N_VALUES:
    key, tk, sk, dk = jr.split(key, 4)
    tpf  = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)
    tape = tpf.make_tape(ys_2d, twist_params, tk, N)
    x0s  = jr.normal(sk, (N, D))

    # sequential
    params = {**params_base}
    run_s = jax.jit(tpf.run_sequential)
    run_s(x0s, tape, params).block_until_ready()
    ts = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        ss = run_s(x0s, tape, params).block_until_ready()
        ts.append(time.perf_counter() - t0)
    t_seq = sorted(ts)[N_RUNS // 2]

    # parallel
    params['key'] = dk
    run_p = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
    sp, it = run_p(x0s, tape, params, max_iter=100)
    sp.block_until_ready()
    tp = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sp, it = run_p(x0s, tape, params, max_iter=100)
        sp.block_until_ready()
        tp.append(time.perf_counter() - t0)
    t_par = sorted(tp)[N_RUNS // 2]

    rmse_s = float(jnp.sqrt(jnp.mean((ss.mean(0)[:, 0] - sm)**2)))
    rmse_p = float(jnp.sqrt(jnp.mean((sp.mean(0)[:, 0] - sm)**2)))
    di = int(it[0])
    su = t_seq / t_par

    print(f"  N={N:>5}  seq={t_seq:.4f}s  par={t_par:.4f}s"
          f"  speedup={su:>6.1f}x  iters={di:>3}  RMSE_par={rmse_p:.4f}")
    results.append(dict(N=N, t_seq=t_seq, t_par=t_par, speedup=su,
                        deer_iters=di, rmse_seq=rmse_s, rmse_par=rmse_p))

print("\n" + "=" * 80)
print(f"{'N':>6} {'t_seq':>10} {'t_par':>10} {'speedup':>10}"
      f" {'iters':>6} {'RMSE_par':>10}")
print("-" * 80)
for r in results:
    print(f"{r['N']:>6} {r['t_seq']:>10.4f} {r['t_par']:>10.4f}"
          f" {r['speedup']:>9.1f}x {r['deer_iters']:>6}"
          f" {r['rmse_par']:>10.4f}")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

Ns = [r['N'] for r in results]

ax = axes[0]
ax.loglog(Ns, [r['t_seq'] for r in results], 'bo-', lw=2, label='Sequential')
ax.loglog(Ns, [r['t_par'] for r in results], 'ro-', lw=2, label='Parallel')
ax.set_xlabel('Number of particles N')
ax.set_ylabel('Time (s)')
ax.set_title('Wall-clock time vs N')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogx(Ns, [r['speedup'] for r in results], 'go-', lw=2)
ax.set_xlabel('Number of particles N')
ax.set_ylabel('Speedup')
ax.set_title('Parallel speedup vs N')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.semilogx(Ns, [r['rmse_par'] for r in results], 'mo-', lw=2)
ax.set_xlabel('Number of particles N')
ax.set_ylabel('RMSE vs smoother')
ax.set_title('Accuracy vs N')
ax.grid(True, alpha=0.3)

plt.suptitle(f'N scaling — T={T}, D={D}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('twisted_pf_scaling_N.png', dpi=150, bbox_inches='tight')
plt.show()
