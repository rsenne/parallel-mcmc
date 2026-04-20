"""
run_pf_comparison.py

Three-way particle filter comparison on 1-D LGSSM:

  1. Bootstrap PF     — standard sequential PF with systematic resampling
  2. Twisted PF (seq) — sequential twisted PF (no resampling, RTS smoother twists)
  3. Twisted PF (par) — parallel twisted PF via per-particle DEER

Compares: wall-clock time, filtering RMSE vs Kalman, ESS, log-ML estimate.

Model:
  x_0 ~ N(0, 1)
  x_t = phi * x_{t-1} + sigma_q * eps_t
  y_t = x_t + sigma_r * eta_t
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter

#  Model parameters 
PHI     = 0.9
SIGMA_Q = 0.5
SIGMA_R = 1.0
D       = 1
N       = 256
T       = 4096


# Ground truth: Kalman filter + RTS smoother

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

def kalman_smoother(ys, phi, sigma_q, sigma_r):
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


def backward_information_filter_1d(ys, phi, sigma_q, sigma_r):
    """
    Backward information filter: computes the optimal twisted SMC twist.

    The optimal twist is psi_t(x_t) = p(y_{t:T} | x_t), the predictive
    likelihood of current and future observations given the state.  For
    a 1-D LGSSM this is Gaussian in x_t, parameterised in information form
    by precision J_t and potential h_t.

    Returns twist means (h_t/J_t) and variances (1/J_t) for all t.
    """
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2

    def step(carry, y):
        J, h = carry
        # Predict backward through transition x_{t+1} = phi * x_t + noise
        denom = 1.0 + sq * J
        J_pred = phi**2 * J / denom
        h_pred = phi * h / denom
        # Incorporate current observation y_t
        J_new = J_pred + prec_obs
        h_new = h_pred + y * prec_obs
        return (J_new, h_new), (J_new, h_new)

    # Initialise at t = T: beta_T(x_T) = p(y_T | x_T)
    J_T = prec_obs
    h_T = ys[-1] * prec_obs

    # Backward scan from t = T-1 down to t = 0
    _, (Js_rev, hs_rev) = jax.lax.scan(step, (J_T, h_T), ys[-2::-1])

    Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_T])])
    hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_T])])

    return hs / Js, 1.0 / Js   # twist_means, twist_vars



# Method 1: Bootstrap PF (systematic resampling)

def bootstrap_pf(ys, phi, sigma_q, sigma_r, key, N):
    """
    Standard bootstrap particle filter.

    Returns: filter_means (T,), log_ml (scalar), ess_per_step (T,)
    """
    def _systematic_resample(key, log_w, particles):
        w = jax.nn.softmax(log_w)
        n = w.shape[0]
        u = (jnp.arange(n) + jr.uniform(key)) / n
        cumw = jnp.cumsum(w)
        idx = jnp.searchsorted(cumw, u)
        idx = jnp.clip(idx, 0, n - 1)
        return particles[idx]

    def step(carry, yt):
        particles, key, log_ml_acc = carry
        key, trans_key, resamp_key = jr.split(key, 3)

        # propagate: x_t ~ N(phi * x_{t-1}, sigma_q^2)
        particles = phi * particles + sigma_q * jr.normal(trans_key, particles.shape)

        # weight: log p(y_t | x_t)
        log_w = -0.5 * ((yt - particles) / sigma_r)**2 - jnp.log(
            sigma_r * jnp.sqrt(2 * jnp.pi))

        # filter mean (before resampling, using weights)
        w_norm = jax.nn.softmax(log_w)
        filt_mean = jnp.sum(w_norm * particles)

        # ESS
        ess = 1.0 / jnp.sum(w_norm**2)

        # log-ML increment
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))
        log_ml_acc = log_ml_acc + log_ml_inc

        # resample
        particles = _systematic_resample(resamp_key, log_w, particles)

        return (particles, key, log_ml_acc), (filt_mean, ess)

    key, init_key = jr.split(key)
    particles = jr.normal(init_key, (N,))
    (_, _, log_ml), (filt_means, ess) = jax.lax.scan(
        step, (particles, key, 0.0), ys)
    return filt_means, log_ml, ess

# Methods 2 & 3: Twisted PF (sequential and parallel)

def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi = params['phi']; sq = params['sigma_q']
    mu  = twist['mean']; tau = jnp.exp(twist['log_scale'])
    pq  = 1.0 / sq**2; pt = 1.0 / tau**2
    vp  = 1.0 / (pq + pt)
    mp  = vp * (pq * phi * x_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise

def log_weight_fn(x_prev, x_next, obs, twist, params):
    phi, sq, sr = params['phi'], params['sigma_q'], params['sigma_r']
    mu = twist['mean']; tau = jnp.exp(twist['log_scale'])
    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sr**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sr**2))
    log_trans = (-0.5 * jnp.sum((x_next - phi * x_prev)**2) / sq**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sq**2))
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    log_prop  = (-0.5 * jnp.sum((x_next - mp)**2 / vp)
                 - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * vp)))
    return log_lik + log_trans - log_prop



# Simulate data


key = jr.PRNGKey(0)
key, k1, k2, k3 = jr.split(key, 4)

x0 = jr.normal(k1)
def sim_step(x, qr):
    xn = PHI * x + SIGMA_Q * qr[0]
    return xn, (xn, xn + SIGMA_R * qr[1])
_, (true_xs, ys) = jax.lax.scan(sim_step, x0, (jr.normal(k2, (T,)), jr.normal(k3, (T,))))

# ground truth
kf_means, kf_vars, kf_log_ml = kalman_filter(ys, PHI, SIGMA_Q, SIGMA_R)
ks_means, ks_vars = kalman_smoother(ys, PHI, SIGMA_Q, SIGMA_R)

print(f"T={T}, N={N}, D={D}")
print(f"Kalman filter log-ML: {float(kf_log_ml):.2f}")
print("=" * 75)

#  twist params from backward information filter (optimal twist) 
ys_2d = ys[:, None]
twist_means, twist_vars = backward_information_filter_1d(ys, PHI, SIGMA_Q, SIGMA_R)
twist_params = {
    'mean':      twist_means[:, None],
    'log_scale': 0.5 * jnp.log(twist_vars)[:, None],
}
model_params = {'phi': PHI, 'sigma_q': SIGMA_Q, 'sigma_r': SIGMA_R}

tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)

key, tape_key, x0_key = jr.split(key, 3)
tape = tpf.make_tape(ys_2d, twist_params, tape_key, N)
x0s = jr.normal(x0_key, (N, D))


# Run all three methods

#  Method 1: Bootstrap PF 
key, bpf_key = jr.split(key)
run_bpf = jax.jit(bootstrap_pf, static_argnames=("N",))
_ = run_bpf(ys, PHI, SIGMA_Q, SIGMA_R, bpf_key, N)
jax.block_until_ready(_)

t0 = time.perf_counter()
bpf_means, bpf_log_ml, bpf_ess = run_bpf(ys, PHI, SIGMA_Q, SIGMA_R, bpf_key, N)
jax.block_until_ready(bpf_means)
t_bpf = time.perf_counter() - t0

#  Method 2: Twisted PF (sequential) 
run_seq = jax.jit(tpf.run_sequential)
_ = run_seq(x0s, tape, model_params)
jax.block_until_ready(_)

t0 = time.perf_counter()
states_seq = run_seq(x0s, tape, model_params)
states_seq.block_until_ready()
t_seq = time.perf_counter() - t0

#  Method 3: Twisted PF (parallel via DEER) 
key, deer_key = jr.split(key)
model_params_par = {**model_params, 'key': deer_key}

run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter",))
_ = run_par(x0s, tape, model_params_par, max_iter=100)
jax.block_until_ready(_)

t0 = time.perf_counter()
states_par, iters = run_par(x0s, tape, model_params_par, max_iter=100)
states_par.block_until_ready()
t_par = time.perf_counter() - t0
deer_iters = int(iters[0])


# Compute metrics

# Twisted PF filtering means (equal-weight since no resampling)
seq_means = states_seq.mean(axis=0)[:, 0]
par_means = states_par.mean(axis=0)[:, 0]

# Twisted PF log-ML and ESS via importance weights
log_ws_seq = jax.jit(tpf.compute_log_weights)(x0s, states_seq, tape, model_params)
log_ws_par = jax.jit(tpf.compute_log_weights)(x0s, states_par, tape, model_params)

# cumulative weights → per-timestep ESS and log-ML
def compute_ess_and_logml(log_ws):
    """Compute per-step ESS and total log-ML from (N, T) log-weights."""
    cum_log_ws = jnp.cumsum(log_ws, axis=1)  # (N, T)

    # per-step ESS
    def ess_at_t(cum_t):
        log_norm = jax.scipy.special.logsumexp(cum_t)
        w = jnp.exp(cum_t - log_norm)
        return 1.0 / jnp.sum(w**2)
    ess = jax.vmap(ess_at_t, in_axes=1)(cum_log_ws)  # (T,)

    # total log-ML
    total = jnp.sum(log_ws, axis=1)  # (N,)
    log_ml = jax.scipy.special.logsumexp(total) - jnp.log(float(log_ws.shape[0]))

    # weighted means per timestep
    def weighted_mean_at_t(cum_t, states_t):
        log_norm = jax.scipy.special.logsumexp(cum_t)
        w = jnp.exp(cum_t - log_norm)
        return jnp.sum(w[:, None] * states_t, axis=0)

    return ess, log_ml

ess_seq, logml_seq = compute_ess_and_logml(log_ws_seq)
ess_par, logml_par = compute_ess_and_logml(log_ws_par)

# RMSE vs Kalman filter
rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_means - kf_means)**2)))
rmse_seq = float(jnp.sqrt(jnp.mean((seq_means - kf_means)**2)))
rmse_par = float(jnp.sqrt(jnp.mean((par_means - kf_means)**2)))

# RMSE vs Kalman smoother
rmse_bpf_s = float(jnp.sqrt(jnp.mean((bpf_means - ks_means)**2)))
rmse_seq_s = float(jnp.sqrt(jnp.mean((seq_means - ks_means)**2)))
rmse_par_s = float(jnp.sqrt(jnp.mean((par_means - ks_means)**2)))

# seq-par agreement
max_diff = float(jnp.max(jnp.abs(states_seq - states_par)))
mean_ess_bpf = float(bpf_ess.mean())
mean_ess_seq = float(ess_seq.mean())
mean_ess_par = float(ess_par.mean())


# Print results

print(f"\n{'Method':>25} {'Time':>8} {'Speedup':>8} {'RMSE(filt)':>11} {'RMSE(smth)':>11}"
      f" {'log-ML':>10} {'mean ESS':>10}")
print("-" * 93)
print(f"{'Bootstrap PF':>25} {t_bpf:>7.3f}s {'1.0x':>8} {rmse_bpf:>11.4f} {rmse_bpf_s:>11.4f}"
      f" {float(bpf_log_ml):>10.1f} {mean_ess_bpf:>9.1f}/{N}")
print(f"{'Twisted PF (seq)':>25} {t_seq:>7.3f}s {t_bpf/t_seq:>7.1f}x {rmse_seq:>11.4f} {rmse_seq_s:>11.4f}"
      f" {float(logml_seq):>10.1f} {mean_ess_seq:>9.1f}/{N}")
print(f"{'Twisted PF (parallel)':>25} {t_par:>7.3f}s {t_bpf/t_par:>7.1f}x {rmse_par:>11.4f} {rmse_par_s:>11.4f}"
      f" {float(logml_par):>10.1f} {mean_ess_par:>9.1f}/{N}")

print(f"\nKalman exact log-ML:         {float(kf_log_ml):>10.1f}")
print(f"DEER iterations:             {deer_iters}")
print(f"Par vs seq speedup:          {t_seq/t_par:.1f}x")
print(f"Max |seq - par| state diff:  {max_diff:.6f}")

# Plot

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

t_idx = jnp.arange(T)

#  Panel 1: State tracking (zoomed) 
ax = fig.add_subplot(gs[0, :])
zoom = slice(0, min(500, T))
ax.plot(t_idx[zoom], true_xs[zoom], 'k-', lw=1, alpha=0.7, label='True state')
ax.plot(t_idx[zoom], ys[zoom], 'k.', ms=1, alpha=0.2, label='Observations')
ax.plot(t_idx[zoom], kf_means[zoom], 'g-', lw=1.5, label='Kalman filter')
ax.plot(t_idx[zoom], ks_means[zoom], 'g--', lw=1.5, alpha=0.7, label='Kalman smoother')
ax.plot(t_idx[zoom], bpf_means[zoom], color='gray', lw=1, alpha=0.7,
        label=f'Bootstrap PF (RMSE vs filt={rmse_bpf:.3f})')
ax.plot(t_idx[zoom], par_means[zoom], 'r-', lw=1.5,
        label=f'Twisted par (RMSE vs smth={rmse_par_s:.4f})')
ax.legend(fontsize=8, ncol=3)
ax.set_title('State estimation (first 500 steps)')
ax.set_xlabel('t')

#  Panel 2: ESS over time 
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_idx, bpf_ess, color='gray', alpha=0.5, lw=0.5, label='Bootstrap PF')
ax.plot(t_idx, ess_seq, 'b-', alpha=0.5, lw=0.5, label='Twisted seq')
ax.plot(t_idx, ess_par, 'r-', alpha=0.5, lw=0.5, label='Twisted par')
ax.axhline(N, color='k', ls=':', lw=0.5, alpha=0.3)
ax.set_ylabel('ESS')
ax.set_xlabel('t')
ax.set_title('Effective sample size')
ax.legend(fontsize=8)

#  Panel 3: ESS histogram 
ax = fig.add_subplot(gs[1, 1])
ax.hist(bpf_ess, bins=50, alpha=0.4, color='gray', density=True, label='Bootstrap PF')
ax.hist(ess_seq, bins=50, alpha=0.4, color='b', density=True, label='Twisted seq')
ax.hist(ess_par, bins=50, alpha=0.4, color='r', density=True, label='Twisted par')
ax.set_xlabel('ESS')
ax.set_ylabel('Density')
ax.set_title('ESS distribution')
ax.legend(fontsize=8)

#  Panel 4: Timing bar chart 
ax = fig.add_subplot(gs[2, 0])
methods = ['Bootstrap\nPF', 'Twisted\n(seq)', 'Twisted\n(parallel)']
times = [t_bpf, t_seq, t_par]
colors = ['gray', 'steelblue', 'indianred']
bars = ax.bar(methods, times, color=colors, edgecolor='k', linewidth=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(times),
            f'{t:.3f}s', ha='center', fontsize=9)
ax.set_ylabel('Wall-clock time (s)')
ax.set_title('Execution time')

#  Panel 5: Summary table as text 
ax = fig.add_subplot(gs[2, 1])
ax.axis('off')
table_data = [
    ['Metric', 'Bootstrap', 'Twisted seq', 'Twisted par'],
    ['Time (s)', f'{t_bpf:.3f}', f'{t_seq:.3f}', f'{t_par:.3f}'],
    ['Speedup', '1.0x', f'{t_bpf/t_seq:.1f}x', f'{t_bpf/t_par:.1f}x'],
    ['RMSE (filter)', f'{rmse_bpf:.4f}', f'{rmse_seq:.4f}', f'{rmse_par:.4f}'],
    ['RMSE (smoother)', f'{rmse_bpf_s:.4f}', f'{rmse_seq_s:.4f}', f'{rmse_par_s:.4f}'],
    ['log-ML', f'{float(bpf_log_ml):.1f}', f'{float(logml_seq):.1f}', f'{float(logml_par):.1f}'],
    ['Kalman log-ML', f'{float(kf_log_ml):.1f}', '', ''],
    ['Mean ESS', f'{mean_ess_bpf:.1f}', f'{mean_ess_seq:.1f}', f'{mean_ess_par:.1f}'],
    ['DEER iters', '—', '—', f'{deer_iters}'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#d4e6f1')
        cell.set_text_props(weight='bold')
    cell.set_edgecolor('#888888')
ax.set_title('Summary', pad=15)

plt.suptitle(
    f'Particle Filter Comparison — LGSSM (T={T}, N={N})',
    fontsize=13, y=0.98,
)
plt.savefig('pf_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved plot to pf_comparison.png")
