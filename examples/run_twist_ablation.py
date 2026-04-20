"""
run_twist_ablation.py

Ablation: how does twist quality affect DEER iters / speedup / RMSE?

On the 1-D LGSSM (T=4096, N=256), we run the twisted parallel PF with
four different twist choices — from uninformative prior through Kalman
filter to Kalman smoother (BIF). The twist enters (a) the proposal and
(b) the implicit-DEER Jacobian approximation. A sharp twist collapses
both weight variance *and* the fixed-point iteration: DEER converges
in 1–2 iters; a loose twist forces more iters and more wall time.

Twists compared:
  1. Prior  — Gaussian with huge variance, mean 0 (≈ bootstrap proposal)
  2. Local obs — mu = y_t, var = sigma_r^2 (no transition structure)
  3. Kalman filter — kf_means, kf_vars (uses y_{1:t})
  4. BIF smoother — bif_alpha, bif_beta (uses y_{1:T})
"""
import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter


PHI, SQ, SR = 0.9, 0.5, 1.0
T, D, N = 4096, 1, 256


# Closed-form filters/smoother for the twist 
def kalman_filter(ys):
    def step(carry, y):
        m, P = carry
        m_p = PHI * m
        P_p = PHI**2 * P + SQ**2
        K = P_p / (P_p + SR**2)
        m_new = m_p + K * (y - m_p)
        P_new = (1.0 - K) * P_p
        return (m_new, P_new), (m_p, P_p, m_new, P_new)
    _, (mp, Pp, mf, Pf) = jax.lax.scan(step, (0.0, 1.0), ys)
    return mp, Pp, mf, Pf


def rts_smoother(ys):
    mp, Pp, mf, Pf = kalman_filter(ys)
    def bwd(carry, t):
        m_s, P_s = carry
        G = Pf[t] * PHI / Pp[t+1]
        m_new = mf[t] + G * (m_s - mp[t+1])
        P_new = Pf[t] + G**2 * (P_s - Pp[t+1])
        return (m_new, P_new), (m_new, P_new)
    _, (ms_rev, Ps_rev) = jax.lax.scan(bwd, (mf[-1], Pf[-1]),
                                         jnp.arange(T-2, -1, -1))
    ks_m = jnp.concatenate([ms_rev[::-1], mf[-1:]])
    ks_v = jnp.concatenate([Ps_rev[::-1], Pf[-1:]])
    return ks_m, ks_v


def bif_1d(ys):
    """Backward information filter: psi_t(x_t) ∝ p(y_{t:T} | x_t)."""
    def bwd(carry, y_prev):
        alpha, Beta = carry
        alpha_prop = alpha / PHI
        Beta_prop = (Beta + SQ**2) / PHI**2
        inv = 1.0/Beta_prop + 1.0/SR**2
        Beta_new = 1.0 / inv
        alpha_new = Beta_new * (alpha_prop/Beta_prop + y_prev/SR**2)
        return (alpha_new, Beta_new), (alpha_new, Beta_new)
    alpha_T, Beta_T = ys[-1], SR**2
    _, (alphas_rev, Betas_rev) = jax.lax.scan(bwd, (alpha_T, Beta_T), ys[-2::-1])
    alphas = jnp.concatenate([alphas_rev[::-1], jnp.array([alpha_T])])
    Betas  = jnp.concatenate([Betas_rev[::-1], jnp.array([Beta_T])])
    return alphas, Betas


#  Twisted step + weight fn (diagonal twist) 
def step_fn(x_prev, driver, p):
    _, noise, twist = driver
    mu_t = twist['mean']
    tau_t = jnp.exp(twist['log_scale'])
    prec_q = 1.0 / p['sigma_q']**2
    prec_t = 1.0 / tau_t**2
    var = 1.0 / (prec_q + prec_t)
    mean = var * (prec_q * p['phi'] * x_prev + prec_t * mu_t)
    return mean + jnp.sqrt(var) * noise


def log_w_fn(x_prev, x_next, obs, twist, p):
    mu_t = twist['mean']; tau_t = jnp.exp(twist['log_scale'])
    log_lik = (-0.5 * jnp.sum((obs - x_next)**2) / p['sigma_r']**2
               - 0.5 * jnp.log(2 * jnp.pi * p['sigma_r']**2))
    log_trans = (-0.5 * jnp.sum((x_next - p['phi']*x_prev)**2) / p['sigma_q']**2
                 - 0.5 * jnp.log(2 * jnp.pi * p['sigma_q']**2))
    prec_q = 1.0 / p['sigma_q']**2; prec_t = 1.0 / tau_t**2
    var = 1.0 / (prec_q + prec_t)
    mean = var * (prec_q * p['phi']*x_prev + prec_t * mu_t)
    log_prop = (-0.5 * jnp.sum((x_next - mean)**2 / var)
                - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * var)))
    return log_lik + log_trans - log_prop


def run_with_twist(twist, label, tpf, x0s, ys_2d, pars, key):
    # JIT + warmup
    key, tk = jr.split(key)
    tape = tpf.make_tape(ys_2d, twist, tk, N)

    run_seq = jax.jit(tpf.run_sequential)
    _ = run_seq(x0s, tape, pars).block_until_ready()
    times_seq = []
    for _ in range(5):
        t0 = time.perf_counter()
        states_seq = run_seq(x0s, tape, pars).block_until_ready()
        times_seq.append(time.perf_counter() - t0)
    t_seq = min(times_seq)

    key, dk = jr.split(key)
    pars_par = {**pars, 'key': dk}
    run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
    _ = run_par(x0s, tape, pars_par, max_iter=100)
    times_par = []
    all_iters = []
    for _ in range(5):
        t0 = time.perf_counter()
        states_par, iters = run_par(x0s, tape, pars_par, max_iter=100)
        states_par.block_until_ready()
        times_par.append(time.perf_counter() - t0)
        all_iters.append(int(iters[0]))
    t_par = min(times_par)
    deer_iters = max(all_iters)

    log_ws = jax.jit(tpf.compute_log_weights)(x0s, states_par, tape, pars)
    w_path = jax.nn.softmax(jnp.sum(log_ws, axis=1))
    means_par = jnp.einsum('n,ntd->td', w_path, states_par)[:, 0]

    # Cumulative ESS for a "weight-balancedness" proxy
    cum = jnp.cumsum(log_ws, axis=1)
    log_norm = jax.scipy.special.logsumexp(cum, axis=0)
    w_t = jnp.exp(cum - log_norm[None, :])
    ess_min = float(1.0 / jnp.max(jnp.sum(w_t**2, axis=0)))

    return {
        'label': label,
        'deer_iters': deer_iters,
        't_seq': t_seq, 't_par': t_par,
        'speedup': t_seq / t_par,
        'means_par': means_par,
        'ess_min': ess_min,
    }


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    key, k1, k2, k3 = jr.split(key, 4)
    x0 = jr.normal(k1)
    q_noise = jr.normal(k2, (T,))
    r_noise = jr.normal(k3, (T,))

    def sim(x, inp):
        q, r = inp
        xn = PHI * x + SQ * q
        return xn, (xn, xn + SR * r)
    _, (true_xs, ys) = jax.lax.scan(sim, x0, (q_noise, r_noise))
    ys_2d = ys[:, None]

    kp, kP, kf_m, kf_v = kalman_filter(ys)
    ks_m, ks_v = rts_smoother(ys)
    bif_a, bif_b = bif_1d(ys)

    # Four twists
    twists = {
        'Prior (wide)': {
            'mean': jnp.zeros((T, D)),
            'log_scale': jnp.full((T, D), 0.5 * jnp.log(1e4)),
        },
        'Local obs':   {
            'mean': ys[:, None],
            'log_scale': jnp.full((T, D), 0.5 * jnp.log(SR**2)),
        },
        'Kalman filter': {
            'mean': kf_m[:, None],
            'log_scale': 0.5 * jnp.log(kf_v)[:, None],
        },
        'BIF smoother': {
            'mean': bif_a[:, None],
            'log_scale': 0.5 * jnp.log(bif_b)[:, None],
        },
    }

    pars = {'phi': PHI, 'sigma_q': SQ, 'sigma_r': SR}
    tpf = TwistedParallelParticleFilter(step_fn, log_w_fn, D=D)

    key, sk = jr.split(key)
    x0s = jr.normal(sk, (N, D))

    results = []
    for label, tw in twists.items():
        print(f"\n--- Twist: {label} ---")
        key, rk = jr.split(key)
        r = run_with_twist(tw, label, tpf, x0s, ys_2d, pars, rk)
        results.append(r)
        rmse = float(jnp.sqrt(jnp.mean((r['means_par'] - ks_m)**2)))
        r['rmse'] = rmse
        print(f"  DEER iters:       {r['deer_iters']}")
        print(f"  t_seq / t_par:    {r['t_seq']*1e3:.2f} ms / {r['t_par']*1e3:.2f} ms")
        print(f"  speedup:          {r['speedup']:.1f}x")
        print(f"  min cum-ESS:      {r['ess_min']:.2f} / {N}")
        print(f"  RMSE vs KS:       {rmse:.4f}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Twist':<18} {'DEER iters':>10} {'t_seq (ms)':>11} {'t_par (ms)':>11} "
          f"{'speedup':>8} {'min ESS':>8} {'RMSE':>8}")
    print('-' * 80)
    for r in results:
        print(f"{r['label']:<18} {r['deer_iters']:>10d} "
              f"{r['t_seq']*1e3:>11.2f} {r['t_par']*1e3:>11.2f} "
              f"{r['speedup']:>7.1f}x {r['ess_min']:>8.1f} {r['rmse']:>8.4f}")

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    labels = [r['label'] for r in results]
    x_pos = jnp.arange(len(labels))

    ax = axes[0]
    ax.bar(x_pos, [r['deer_iters'] for r in results], color='#d62728', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('DEER iterations')
    ax.set_title('Convergence depth vs twist quality')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x_pos, [r['speedup'] for r in results], color='#1f77b4', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Parallel / sequential speedup')
    ax.set_title('Wall-clock gain')
    ax.axhline(1, ls='--', c='k', alpha=0.4)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(x_pos, [r['rmse'] for r in results], color='#2ca02c', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('RMSE vs Kalman smoother')
    ax.set_title('Estimator quality')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Twist-quality ablation on LGSSM (T={T}, N={N})', fontsize=12)
    plt.tight_layout()
    plt.savefig('twist_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved twist_ablation.png")
