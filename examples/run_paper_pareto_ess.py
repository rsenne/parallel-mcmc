"""
run_paper_pareto_ess.py

Unified benchmark producing the two headline paper figures:

  1. Wall-clock vs RMSE Pareto — one subplot per model, each with
     bootstrap PF (filter), FFBSi (smoother), twisted seq, twisted par.
  2. ESS-over-time — shows why twisted PF can skip resampling (cumulative
     path-ESS stays close to N) vs bootstrap collapse without resampling.

Models: 1-D LGSSM (T=10000) and stochvol (T=1000).
Both match the settings in run_twisted_pf_linear_gaussian.py and
run_twisted_pf_stochvol.py, so results are directly comparable.
"""
import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter
from src.baselines import bootstrap_pf, ffbsi_smoother


def cumulative_ess(log_ws):
    """Path-ESS from incremental log-weights. log_ws: (N, T). Returns (T,)."""
    cum = jnp.cumsum(log_ws, axis=1)
    log_norm = jax.scipy.special.logsumexp(cum, axis=0)
    w = jnp.exp(cum - log_norm[None, :])
    return 1.0 / jnp.sum(w ** 2, axis=0)


def bpf_cumulative_ess(all_log_w):
    """For bootstrap PF with resampling, per-step ESS (pre-resample)."""
    w = jax.nn.softmax(all_log_w, axis=1)  # (T, N)
    return 1.0 / jnp.sum(w ** 2, axis=1)    # (T,)


# 1) LGSSM
def kalman_filter_lgssm(ys, phi, sigma_q, sigma_r):
    def step(carry, y):
        m, P = carry
        m_p = phi * m
        P_p = phi**2 * P + sigma_q**2
        K = P_p / (P_p + sigma_r**2)
        m_new = m_p + K * (y - m_p)
        P_new = (1.0 - K) * P_p
        return (m_new, P_new), (m_new, P_new)
    _, (mus, vars_) = jax.lax.scan(step, (0.0, 1.0), ys)
    return mus, vars_


def run_lgssm():
    PHI, SQ, SR = 0.9, 0.5, 1.0
    T, D, N = 10000, 1, 256
    key = jr.PRNGKey(0)

    # ---- simulate ----
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

    kf_means, kf_vars = kalman_filter_lgssm(ys, PHI, SQ, SR)

    # RTS smoother for building a ground-truth posterior baseline
    def rts_smoother(ys, phi, sq, sr):
        # Forward filter: store m_pred, P_pred, m_filt, P_filt
        def fwd(carry, y):
            m, P = carry
            m_p = phi * m; P_p = phi**2 * P + sq**2
            K = P_p / (P_p + sr**2)
            m_n = m_p + K * (y - m_p); P_n = (1.0 - K) * P_p
            return (m_n, P_n), (m_p, P_p, m_n, P_n)
        _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(fwd, (0.0, 1.0), ys)
        # Backward RTS
        def bwd(carry, t):
            m_s, P_s = carry
            G = P_filts[t] * phi / P_preds[t+1]
            m_new = m_filts[t] + G * (m_s - m_preds[t+1])
            P_new = P_filts[t] + G**2 * (P_s - P_preds[t+1])
            return (m_new, P_new), (m_new, P_new)
        _, (ms_rev, Ps_rev) = jax.lax.scan(bwd, (m_filts[-1], P_filts[-1]),
                                            jnp.arange(T-2, -1, -1))
        return (jnp.concatenate([ms_rev[::-1], m_filts[-1:]]),
                jnp.concatenate([Ps_rev[::-1], P_filts[-1:]]))

    ks_means, ks_vars = rts_smoother(ys, PHI, SQ, SR)

    # Backward information filter: psi_t(x_t) ∝ p(y_{t:T}|x_t). Each step
    # gives a Gaussian twist that — combined with the forward prior —
    # yields the smoother marginal, not just the filter.
    def bif_1d(ys, phi, sq, sr):
        def bwd(carry, y_prev):
            alpha, Beta = carry
            # Backward-propagate through x_t = phi*x_{t-1} + sq*eps:
            alpha_prop = alpha / phi
            Beta_prop = (Beta + sq**2) / phi**2
            # Fuse with p(y_{t-1}|x_{t-1}) = N(y_{t-1}; x_{t-1}, sr^2):
            inv = 1.0/Beta_prop + 1.0/sr**2
            Beta_new = 1.0/inv
            alpha_new = Beta_new * (alpha_prop/Beta_prop + y_prev/sr**2)
            return (alpha_new, Beta_new), (alpha_new, Beta_new)
        alpha_T, Beta_T = ys[-1], sr**2
        _, (alphas_rev, Betas_rev) = jax.lax.scan(
            bwd, (alpha_T, Beta_T), ys[-2::-1])
        alphas = jnp.concatenate([alphas_rev[::-1], jnp.array([alpha_T])])
        Betas  = jnp.concatenate([Betas_rev[::-1],  jnp.array([Beta_T])])
        return alphas, Betas

    bif_alpha, bif_beta = bif_1d(ys, PHI, SQ, SR)

    # twisted pf
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

    # BIF twist: psi_t(x_t) ≈ N(alpha_t, Beta_t) encodes all future obs.
    twist = {'mean':      bif_alpha[:, None],
             'log_scale': 0.5 * jnp.log(bif_beta)[:, None]}
    pars = {'phi': PHI, 'sigma_q': SQ, 'sigma_r': SR}
    tpf = TwistedParallelParticleFilter(step_fn, log_w_fn, D=D)

    key, tk = jr.split(key)
    tape = tpf.make_tape(ys_2d, twist, tk, N)
    key, sk = jr.split(key)
    x0s = jr.normal(sk, (N, D))

    run_seq = jax.jit(tpf.run_sequential)
    _ = run_seq(x0s, tape, pars)
    t0 = time.perf_counter()
    states_seq = run_seq(x0s, tape, pars).block_until_ready()
    t_seq = time.perf_counter() - t0

    key, dk = jr.split(key)
    pars_par = {**pars, 'key': dk}
    run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
    _ = run_par(x0s, tape, pars_par, max_iter=100)
    t0 = time.perf_counter()
    states_par, _ = run_par(x0s, tape, pars_par, max_iter=100)
    states_par.block_until_ready()
    t_par = time.perf_counter() - t0

    log_ws = jax.jit(tpf.compute_log_weights)(x0s, states_par, tape, pars)
    # Full-path weights give smoother estimate E[x_t | y_{1:T}].
    w_path = jax.nn.softmax(jnp.sum(log_ws, axis=1))
    means_par = jnp.einsum('n,ntd->td', w_path, states_par)[:, 0]

    log_ws_s = jax.jit(tpf.compute_log_weights)(x0s, states_seq, tape, pars)
    w_path_s = jax.nn.softmax(jnp.sum(log_ws_s, axis=1))
    means_seq = jnp.einsum('n,ntd->td', w_path_s, states_seq)[:, 0]

    # Compare to Kalman smoother (ground-truth posterior).
    rmse_seq = float(jnp.sqrt(jnp.mean((means_seq - ks_means)**2)))
    rmse_par = float(jnp.sqrt(jnp.mean((means_par - ks_means)**2)))
    ess_par = cumulative_ess(log_ws)

    # BPF
    def lgssm_trans(x, p): return p['phi'] * x
    def lgssm_ll(y, x, p): return -0.5 * jnp.sum((y - x)**2) / p['sigma_r']**2

    key, bpf_key, ff_key, ik = jr.split(key, 4)
    x0p = jr.normal(ik, (N, D))
    run_bpf = jax.jit(bootstrap_pf, static_argnames=("transition_mean", "log_lik"))
    _ = run_bpf(ys_2d, x0p, lgssm_trans, lgssm_ll, SQ, pars, bpf_key)
    t0 = time.perf_counter()
    bpf_filt, bpf_parts, bpf_lw, bpf_ess, _ = run_bpf(
        ys_2d, x0p, lgssm_trans, lgssm_ll, SQ, pars, bpf_key)
    bpf_filt.block_until_ready()
    t_bpf = time.perf_counter() - t0
    # BPF is a filter; still compare to smoother ground truth so all
    # methods are judged by the same "posterior posterior estimate" metric.
    rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_filt[:, 0] - ks_means)**2)))

    run_ff = jax.jit(ffbsi_smoother, static_argnames=("M", "transition_mean"))
    _ = run_ff(bpf_parts, bpf_lw, lgssm_trans, SQ, pars, ff_key, M=N)
    t0 = time.perf_counter()
    bps = run_ff(bpf_parts, bpf_lw, lgssm_trans, SQ, pars, ff_key, M=N)
    bps.block_until_ready()
    t_ff = time.perf_counter() - t0
    rmse_ff = float(jnp.sqrt(jnp.mean((bps.mean(axis=0)[:, 0] - ks_means)**2)))

    return {
        'name': 'LGSSM', 'T': T, 'N': N,
        'results': [
            ('Bootstrap PF',   t_bpf, rmse_bpf),
            ('FFBSi',          t_ff,  rmse_ff),
            ('Twisted (seq)',  t_seq, rmse_seq),
            ('Twisted (par)',  t_par, rmse_par),
        ],
        'ess_tpf': ess_par,
        'ess_bpf': bpf_ess,
    }


# 2) Stochvol model

def kalman_approx_sv(ys, phi, sigma_q, beta):
    """Log-space smoother for stochvol's observation model.

    Approximation: log(y^2) ~= x + log(beta^2) + log(eps^2),
    where log(eps^2) has mean c=-1.27, variance R=pi^2/2.
    Runs forward EKF + backward RTS on the linear approximation, so the
    returned marginals are smoothed (incorporate y_{t+1:T}), not just
    filtered. Gives a much better twist than the single-pass version.
    """
    z = jnp.log(ys**2 + 1e-6)
    c = -1.27
    R = jnp.pi**2 / 2.0

    def fwd(carry, zt):
        m, P = carry
        m_p = phi * m
        P_p = phi**2 * P + sigma_q**2
        S = P_p + R
        K = P_p / S
        m_n = m_p + K * (zt - c - jnp.log(beta**2) - m_p)
        P_n = (1.0 - K) * P_p
        return (m_n, P_n), (m_p, P_p, m_n, P_n)
    _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(fwd, (0.0, 1.0), z)

    T = z.shape[0]
    def bwd(carry, t):
        m_s, P_s = carry
        G = P_filts[t] * phi / P_preds[t+1]
        m_new = m_filts[t] + G * (m_s - m_preds[t+1])
        P_new = P_filts[t] + G**2 * (P_s - P_preds[t+1])
        return (m_new, P_new), (m_new, P_new)
    _, (ms_rev, Ps_rev) = jax.lax.scan(bwd, (m_filts[-1], P_filts[-1]),
                                        jnp.arange(T-2, -1, -1))
    ms = jnp.concatenate([ms_rev[::-1], m_filts[-1:]])
    Ps = jnp.concatenate([Ps_rev[::-1], P_filts[-1:]])
    return ms, Ps


def run_stochvol():
    PHI, SQ, BETA = 0.95, 0.3, 1.0
    T, D, N = 1000, 1, 128
    key = jr.PRNGKey(1)

    # ---- simulate ----
    key, k1, k2, k3 = jr.split(key, 4)
    x0 = jr.normal(k1) * 0.5
    q_noise = jr.normal(k2, (T,))
    obs_noise = jr.normal(k3, (T,))

    def sim(x, inp):
        q, r = inp
        xn = PHI * x + SQ * q
        y = BETA * jnp.exp(0.5 * xn) * r
        return xn, (xn, y)
    _, (true_xs, ys) = jax.lax.scan(sim, x0, (q_noise, obs_noise))
    ys_2d = ys[:, None]

    # EKF-like twist (single pass)
    kf_m, kf_v = kalman_approx_sv(ys, PHI, SQ, BETA)
    twist = {'mean': kf_m[:, None], 'log_scale': 0.5 * jnp.log(kf_v)[:, None]}

    # ---- twisted PF ----
    def step_fn(x_prev, driver, p):
        _, noise, tw = driver
        mu_t = tw['mean']; tau_t = jnp.exp(tw['log_scale'])
        prec_q = 1.0 / p['sigma_q']**2; prec_t = 1.0 / tau_t**2
        var = 1.0 / (prec_q + prec_t)
        mean = var * (prec_q * p['phi'] * x_prev + prec_t * mu_t)
        return mean + jnp.sqrt(var) * noise

    def log_w_fn(x_prev, x_next, obs, tw, p):
        mu_t = tw['mean']; tau_t = jnp.exp(tw['log_scale'])
        obs_var = p['beta']**2 * jnp.exp(x_next)
        log_lik = -0.5 * jnp.sum(obs**2 / obs_var + jnp.log(2*jnp.pi*obs_var))
        log_trans = (-0.5 * jnp.sum((x_next - p['phi']*x_prev)**2) / p['sigma_q']**2
                     - 0.5 * jnp.log(2*jnp.pi*p['sigma_q']**2))
        prec_q = 1.0 / p['sigma_q']**2; prec_t = 1.0 / tau_t**2
        var = 1.0 / (prec_q + prec_t)
        mean = var * (prec_q * p['phi']*x_prev + prec_t * mu_t)
        log_prop = (-0.5 * jnp.sum((x_next - mean)**2 / var)
                    - 0.5 * jnp.sum(jnp.log(2*jnp.pi*var)))
        return log_lik + log_trans - log_prop

    pars = {'phi': PHI, 'sigma_q': SQ, 'beta': BETA}
    tpf = TwistedParallelParticleFilter(step_fn, log_w_fn, D=D)

    key, tk = jr.split(key)
    tape = tpf.make_tape(ys_2d, twist, tk, N)
    key, sk = jr.split(key)
    x0s = jr.normal(sk, (N, D)) * 0.5

    run_seq = jax.jit(tpf.run_sequential)
    _ = run_seq(x0s, tape, pars)
    t0 = time.perf_counter()
    states_seq = run_seq(x0s, tape, pars).block_until_ready()
    t_seq = time.perf_counter() - t0

    key, dk = jr.split(key)
    pars_par = {**pars, 'key': dk}
    run_par = jax.jit(tpf.run_parallel, static_argnames=("max_iter", "num_samples"))
    _ = run_par(x0s, tape, pars_par, max_iter=100)
    t0 = time.perf_counter()
    states_par, _ = run_par(x0s, tape, pars_par, max_iter=100)
    states_par.block_until_ready()
    t_par = time.perf_counter() - t0

    log_ws = jax.jit(tpf.compute_log_weights)(x0s, states_par, tape, pars)
    log_ws_s = jax.jit(tpf.compute_log_weights)(x0s, states_seq, tape, pars)

    def weighted_mean(states, lws):
        # Full-path weights — smoother estimate.
        w = jax.nn.softmax(jnp.sum(lws, axis=1))
        return jnp.einsum('n,ntd->td', w, states)[:, 0]

    m_seq = weighted_mean(states_seq, log_ws_s)
    m_par = weighted_mean(states_par, log_ws)

    rmse_seq = float(jnp.sqrt(jnp.mean((m_seq - true_xs)**2)))
    rmse_par = float(jnp.sqrt(jnp.mean((m_par - true_xs)**2)))
    ess_par = cumulative_ess(log_ws)

    # BPF
    def sv_trans(x, p): return p['phi'] * x
    def sv_ll(y, x, p):
        var = p['beta']**2 * jnp.exp(x)
        return -0.5 * jnp.sum(y**2 / var + jnp.log(2 * jnp.pi * var))

    key, bk, fk, ik = jr.split(key, 4)
    x0p = jr.normal(ik, (N, D)) * 0.5
    run_bpf = jax.jit(bootstrap_pf, static_argnames=("transition_mean", "log_lik"))
    _ = run_bpf(ys_2d, x0p, sv_trans, sv_ll, SQ, pars, bk)
    t0 = time.perf_counter()
    bpf_filt, bpf_parts, bpf_lw, bpf_ess, _ = run_bpf(
        ys_2d, x0p, sv_trans, sv_ll, SQ, pars, bk)
    bpf_filt.block_until_ready()
    t_bpf = time.perf_counter() - t0
    rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_filt[:, 0] - true_xs)**2)))

    run_ff = jax.jit(ffbsi_smoother, static_argnames=("M", "transition_mean"))
    _ = run_ff(bpf_parts, bpf_lw, sv_trans, SQ, pars, fk, M=N)
    t0 = time.perf_counter()
    bps = run_ff(bpf_parts, bpf_lw, sv_trans, SQ, pars, fk, M=N)
    bps.block_until_ready()
    t_ff = time.perf_counter() - t0
    rmse_ff = float(jnp.sqrt(jnp.mean((bps.mean(axis=0)[:, 0] - true_xs)**2)))

    return {
        'name': 'Stochvol', 'T': T, 'N': N,
        'results': [
            ('Bootstrap PF',   t_bpf, rmse_bpf),
            ('FFBSi',          t_ff,  rmse_ff),
            ('Twisted (seq)',  t_seq, rmse_seq),
            ('Twisted (par)',  t_par, rmse_par),
        ],
        'ess_tpf': ess_par,
        'ess_bpf': bpf_ess,
    }


def plot_all(models):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'Bootstrap PF':  '#888888',
        'FFBSi':         '#1f77b4',
        'Twisted (seq)': '#2ca02c',
        'Twisted (par)': '#d62728',
    }
    markers = {
        'Bootstrap PF':  'o',
        'FFBSi':         's',
        'Twisted (seq)': '^',
        'Twisted (par)': '*',
    }

    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 9))
    if len(models) == 1:
        axes = axes[:, None]

    for j, m in enumerate(models):
        ax = axes[0, j]
        for name, t, rmse in m['results']:
            ax.scatter(t, rmse, s=180 if 'par' in name else 110,
                       c=colors[name], marker=markers[name],
                       edgecolor='k', linewidth=0.8, label=name, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wall-clock time (s)')
        ax.set_ylabel('RMSE')
        ax.set_title(f"{m['name']} (T={m['T']}, N={m['N']})")
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, which='both', alpha=0.3)

        ax = axes[1, j]
        T = m['ess_tpf'].shape[0]
        t_idx = jnp.arange(T)
        ax.plot(t_idx, m['ess_tpf'], c=colors['Twisted (par)'],
                lw=1.5, label='Twisted PF (cum. path-ESS)')
        ax.plot(t_idx, m['ess_bpf'], c=colors['Bootstrap PF'],
                lw=1.5, label='Bootstrap PF (per-step ESS)')
        ax.axhline(m['N'], ls='--', c='k', alpha=0.4, label=f"N={m['N']}")
        ax.set_xlabel('Timestep')
        ax.set_ylabel('ESS')
        ax.set_title(f"{m['name']} — ESS over time")
        ax.legend(fontsize=9, loc='best')
        ax.set_ylim(0.5, m['N'] * 1.1)
        ax.set_yscale('log')

    plt.tight_layout()
    out = 'paper_pareto_ess.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("=" * 70)
    print("Running LGSSM...")
    print("=" * 70)
    m_lg = run_lgssm()
    for n, t, r in m_lg['results']:
        print(f"  {n:<18}  time={t:.4f}s   RMSE={r:.4f}")

    print("\n" + "=" * 70)
    print("Running Stochvol...")
    print("=" * 70)
    m_sv = run_stochvol()
    for n, t, r in m_sv['results']:
        print(f"  {n:<18}  time={t:.4f}s   RMSE={r:.4f}")

    plot_all([m_lg, m_sv])
