"""
run_pf_scaling_comparison.py

T-scaling comparison: Bootstrap PF vs Twisted Parallel PF on 1-D LGSSM.

Both methods are compared against the Kalman filter as ground truth:
  - Bootstrap PF: standard sequential filter with resampling
  - Twisted parallel PF: per-particle DEER smoother with RTS twists

Key tradeoff: resampling ↔ parallelism.
The bootstrap PF maintains ESS via resampling but is inherently sequential.
The twisted PF sacrifices resampling for O(log T) parallel depth.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter

#  Model 
PHI     = 0.9
SIGMA_Q = 0.5
SIGMA_R = 1.0
D       = 1
N       = 256

T_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


# Helper functions
def kalman_filter(ys, phi, sq, sr):
    def step(carry, y):
        m, P = carry
        mp = phi * m; Pp = phi**2 * P + sq**2
        S = Pp + sr**2; K = Pp / S
        mf = mp + K * (y - mp); Pf = (1 - K) * Pp
        ll = -0.5 * (jnp.log(2 * jnp.pi * S) + (y - mp)**2 / S)
        return (mf, Pf), (mf, ll)
    _, (means, lls) = jax.lax.scan(step, (0.0, 1.0), ys)
    return means, jnp.sum(lls)

def kalman_smoother(ys, phi, sq, sr):
    def fwd(carry, y):
        m, P = carry
        mp = phi * m; Pp = phi**2 * P + sq**2
        S = Pp + sr**2; K = Pp / S
        mf = mp + K * (y - mp); Pf = (1 - K) * Pp
        return (mf, Pf), (mp, Pp, mf, Pf)
    _, (mp, Pp, mf, Pf) = jax.lax.scan(fwd, (0.0, 1.0), ys)
    def bwd(carry, t):
        ms, Ps = carry
        G = phi * Pf[t] / Pp[t + 1]
        return (mf[t] + G * (ms - mp[t+1]),
                Pf[t] + G**2 * (Ps - Pp[t+1])), \
               (mf[t] + G * (ms - mp[t+1]),
                Pf[t] + G**2 * (Ps - Pp[t+1]))
    ts = jnp.arange(ys.shape[0] - 2, -1, -1)
    _, (ms_r, Ps_r) = jax.lax.scan(bwd, (mf[-1], Pf[-1]), ts)
    return jnp.concatenate([ms_r[::-1], mf[-1:]]), \
           jnp.concatenate([Ps_r[::-1], Pf[-1:]])

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

def bootstrap_pf(ys, phi, sq, sr, key, N):
    def _resample(key, log_w, particles):
        w = jax.nn.softmax(log_w)
        n = w.shape[0]
        u = (jnp.arange(n) + jr.uniform(key)) / n
        idx = jnp.clip(jnp.searchsorted(jnp.cumsum(w), u), 0, n - 1)
        return particles[idx]
    def step(carry, yt):
        particles, key = carry
        key, tk, rk = jr.split(key, 3)
        particles = phi * particles + sq * jr.normal(tk, particles.shape)
        log_w = -0.5 * ((yt - particles) / sr)**2 - jnp.log(sr * jnp.sqrt(2*jnp.pi))
        w = jax.nn.softmax(log_w)
        filt_mean = jnp.sum(w * particles)
        ess = 1.0 / jnp.sum(w**2)
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))
        particles = _resample(rk, log_w, particles)
        return (particles, key), (filt_mean, ess, log_ml_inc)
    key, ik = jr.split(key)
    (_, _), (means, ess, log_ml_incs) = jax.lax.scan(
        step, (jr.normal(ik, (N,)), key), ys)
    return means, ess, jnp.sum(log_ml_incs)

#  Twisted PF 
def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi, sq = params['phi'], params['sigma_q']
    mu = twist['mean']; tau = jnp.exp(twist['log_scale'])
    pq = 1.0/sq**2; pt = 1.0/tau**2
    vp = 1.0/(pq+pt); mp = vp*(pq*phi*x_prev + pt*mu)
    return mp + jnp.sqrt(vp) * noise

def log_weight_fn(x_prev, x_next, obs, twist, params):
    phi, sq, sr = params['phi'], params['sigma_q'], params['sigma_r']
    mu = twist['mean']; tau = jnp.exp(twist['log_scale'])
    log_lik   = -0.5*jnp.sum((obs-x_next)**2)/sr**2 - 0.5*D*jnp.log(2*jnp.pi*sr**2)
    log_trans = -0.5*jnp.sum((x_next-phi*x_prev)**2)/sq**2 - 0.5*D*jnp.log(2*jnp.pi*sq**2)
    pq = 1.0/sq**2; pt = 1.0/tau**2; vp = 1.0/(pq+pt)
    mp = vp*(pq*phi*x_prev + pt*mu)
    log_prop = -0.5*jnp.sum((x_next-mp)**2/vp) - 0.5*jnp.sum(jnp.log(2*jnp.pi*vp))
    return log_lik + log_trans - log_prop



# Run T-scaling sweep
tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)
model_params = {'phi': PHI, 'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R}

results = []
key = jr.PRNGKey(0)

for T in T_values:
    print(f"\nT = {T:>6d} ...", end=" ", flush=True)
    key, k1, k2, k3, tape_key, x0_key, bpf_key, deer_key = jr.split(key, 8)

    # simulate
    def sim_step(x, qr):
        xn = PHI * x + SIGMA_Q * qr[0]
        return xn, (xn, xn + SIGMA_R * qr[1])
    _, (true_xs, ys) = jax.lax.scan(
        sim_step, jr.normal(k1), (jr.normal(k2, (T,)), jr.normal(k3, (T,))))

    kf_means, kf_log_ml = kalman_filter(ys, PHI, SIGMA_Q, SIGMA_R)
    ks_means, ks_vars = kalman_smoother(ys, PHI, SIGMA_Q, SIGMA_R)

    # Optimal twist from backward information filter
    twist_means, twist_vars = backward_information_filter_1d(ys, PHI, SIGMA_Q, SIGMA_R)
    ys_2d = ys[:, None]
    twist_params = {
        'mean': twist_means[:, None],
        'log_scale': 0.5 * jnp.log(twist_vars)[:, None],
    }
    tape = tpf.make_tape(ys_2d, twist_params, tape_key, N)
    x0s = jr.normal(x0_key, (N, D))

    #  Bootstrap PF 
    run_bpf = jax.jit(bootstrap_pf, static_argnames=("N",))
    bpf_out = run_bpf(ys, PHI, SIGMA_Q, SIGMA_R, bpf_key, N)
    jax.block_until_ready(bpf_out)
    t0 = time.perf_counter()
    bpf_means, bpf_ess, bpf_log_ml = run_bpf(ys, PHI, SIGMA_Q, SIGMA_R, bpf_key, N)
    jax.block_until_ready(bpf_means)
    t_bpf = time.perf_counter() - t0

    rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_means - kf_means)**2)))

    #  Twisted PF (sequential) 
    run_seq = jax.jit(tpf.run_sequential)
    _ = run_seq(x0s, tape, model_params)
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_seq = run_seq(x0s, tape, model_params)
    states_seq.block_until_ready()
    t_seq = time.perf_counter() - t0

    seq_means = states_seq.mean(axis=0)[:, 0]
    rmse_seq_f = float(jnp.sqrt(jnp.mean((seq_means - kf_means)**2)))

    #  Twisted PF (parallel) 
    mp = {**model_params, 'key': deer_key}
    run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter",))
    _ = run_par(x0s, tape, mp, max_iter=100)
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_par, iters = run_par(x0s, tape, mp, max_iter=100)
    states_par.block_until_ready()
    t_par = time.perf_counter() - t0

    par_means = states_par.mean(axis=0)[:, 0]
    rmse_par_f = float(jnp.sqrt(jnp.mean((par_means - kf_means)**2)))
    # RMSE vs smoother (the right comparison for twisted PF)
    rmse_par_s = float(jnp.sqrt(jnp.mean((par_means - ks_means)**2)))
    rmse_bpf_s = float(jnp.sqrt(jnp.mean((bpf_means - ks_means)**2)))
    deer_iters = int(iters[0])

    # Verify seq-par agreement
    max_diff = float(jnp.max(jnp.abs(states_seq - states_par)))

    res = {
        'T': T,
        't_bpf': t_bpf, 't_seq': t_seq, 't_par': t_par,
        'rmse_bpf': rmse_bpf, 'rmse_par': rmse_par_f,
        'rmse_bpf_s': rmse_bpf_s, 'rmse_par_s': rmse_par_s,
        'deer_iters': deer_iters,
        'bpf_ess_mean': float(bpf_ess.mean()),
        'bpf_log_ml': float(bpf_log_ml),
        'kf_log_ml': float(kf_log_ml),
        'max_diff': max_diff,
    }
    results.append(res)

    print(f"BPF={t_bpf:.3f}s  Seq={t_seq:.3f}s  Par={t_par:.4f}s  "
          f"({deer_iters} iters)  "
          f"RMSE: bpf_filt={rmse_bpf:.4f} tw_smth={rmse_par_s:.4f}  "
          f"diff={max_diff:.1e}")

# Summary table

print(f"\n{'T':>7} {'t_bpf':>8} {'t_seq':>8} {'t_par':>8} "
      f"{'BPF/Par':>8} {'Seq/Par':>8} {'DEER':>5} "
      f"{'BPF_filt':>9} {'TW_smth':>9} {'seq≈par':>10}")
print("-" * 95)
for r in results:
    print(f"{r['T']:>7d} {r['t_bpf']:>7.3f}s {r['t_seq']:>7.3f}s "
          f"{r['t_par']:>7.4f}s "
          f"{r['t_bpf']/max(r['t_par'],1e-6):>7.0f}x "
          f"{r['t_seq']/max(r['t_par'],1e-6):>7.0f}x "
          f"{r['deer_iters']:>5d} {r['rmse_bpf']:>9.4f} {r['rmse_par_s']:>9.4f} "
          f"{r['max_diff']:>10.1e}")

# Plot

Ts       = [r['T'] for r in results]
t_bpfs   = [r['t_bpf'] for r in results]
t_seqs   = [r['t_seq'] for r in results]
t_pars   = [r['t_par'] for r in results]
su_bpf   = [b/max(p, 1e-6) for b, p in zip(t_bpfs, t_pars)]
su_seq   = [s/max(p, 1e-6) for s, p in zip(t_seqs, t_pars)]
deer_its = [r['deer_iters'] for r in results]
r_bpfs   = [r['rmse_bpf'] for r in results]
r_pars_s = [r['rmse_par_s'] for r in results]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time vs T
ax = axes[0, 0]
ax.loglog(Ts, t_bpfs, 'o-', color='gray', lw=2, ms=6, label='Bootstrap PF (seq)')
ax.loglog(Ts, t_seqs, 's-', color='steelblue', lw=2, ms=6, label='Twisted PF (seq)')
ax.loglog(Ts, t_pars, '^-', color='indianred', lw=2, ms=6, label='Twisted PF (parallel)')
ax.set_xlabel('Sequence length T')
ax.set_ylabel('Wall-clock time (s)')
ax.set_title('Execution time vs T')
ax.legend(fontsize=9)

# Speedup vs T
ax = axes[0, 1]
ax.semilogx(Ts, su_bpf, 'o-', color='gray', lw=2, ms=6, label='vs Bootstrap PF')
ax.semilogx(Ts, su_seq, 's-', color='steelblue', lw=2, ms=6, label='vs Twisted seq')
ax.set_xlabel('Sequence length T')
ax.set_ylabel('Parallel speedup (x)')
ax.set_title('Speedup of parallel twisted PF')
ax.legend(fontsize=9)

# DEER iterations
ax = axes[1, 0]
ax.semilogx(Ts, deer_its, '^-', color='indianred', lw=2, ms=6)
ax.set_xlabel('Sequence length T')
ax.set_ylabel('DEER iterations')
ax.set_title('DEER convergence')
ax.set_ylim(bottom=0)

# RMSE vs T
ax = axes[1, 1]
ax.semilogx(Ts, r_bpfs, 'o-', color='gray', lw=2, ms=6, label='Bootstrap PF vs Kalman filter')
ax.semilogx(Ts, r_pars_s, '^-', color='indianred', lw=2, ms=6, label='Twisted parallel vs Kalman smoother')
ax.set_xlabel('Sequence length T')
ax.set_ylabel('RMSE vs Kalman filter')
ax.set_title('Estimation accuracy')
ax.legend(fontsize=9)

plt.suptitle(
    f'PF T-Scaling — LGSSM (N={N}, D={D})',
    fontsize=13)
plt.tight_layout()
plt.savefig('pf_scaling_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved plot to pf_scaling_comparison.png")
