"""
run_twisted_pf_lorenz96.py

Twisted parallel particle smoother on the Lorenz-96 chaotic SSM.
This is the medium-D chaotic nonlinear benchmark: for F=8 the system is
strongly chaotic (positive Lyapunov exponents), and the state dimension
is moderate (D in {10, 20, 40}) — the regime where full-Jacobian DEER
dominates diagonal Hutchinson.

Model (RK4-discretised):
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    x_{t+1} = RK4(x_t, dt, F) + sigma_q * N(0, I_D)
    y_t     = x_t + sigma_r * N(0, I_D)

Twist:
    EKF forward pass + RTS backward pass, using smoother marginals
    N(x_t; ks_means[t], diag(ks_vars[t])) as the diagonal twist.

Compares:
    1. Bootstrap PF (filter)
    2. Bootstrap PS (FFBSi)
    3. Twisted PF (seq)
    4. Twisted PF (parallel, full-Jacobian DEER)
"""
import time
from functools import partial

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import TwistedParallelParticleFilter
from src.baselines import bootstrap_pf, ffbsi_smoother


#  Lorenz-96 dynamics (RK4) 
def l96_rhs(x, F):
    return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + F


def rk4_step(x, dt, F):
    k1 = l96_rhs(x, F)
    k2 = l96_rhs(x + 0.5 * dt * k1, F)
    k3 = l96_rhs(x + 0.5 * dt * k2, F)
    k4 = l96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


#  EKF + RTS smoother to build the twist 
def ekf_smoother_l96(ys, dt, F, sigma_q, sigma_r, x0, P0):
    D = x0.shape[0]
    Q = sigma_q**2 * jnp.eye(D)
    R = sigma_r**2 * jnp.eye(D)

    def fwd(carry, y):
        m, P = carry
        m_pred = rk4_step(m, dt, F)
        Fj = jax.jacfwd(lambda z: rk4_step(z, dt, F))(m)
        P_pred = Fj @ P @ Fj.T + Q
        S = P_pred + R
        K = jnp.linalg.solve(S.T, P_pred.T).T
        m_new = m_pred + K @ (y - m_pred)
        P_new = (jnp.eye(D) - K) @ P_pred
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new, Fj)

    _, (m_preds, P_preds, m_filts, P_filts, Fjs) = jax.lax.scan(fwd, (x0, P0), ys)

    def bwd(carry, t):
        m_s, P_s = carry
        G = jnp.linalg.solve(P_preds[t+1].T, (P_filts[t] @ Fjs[t+1].T).T).T
        m_new = m_filts[t] + G @ (m_s - m_preds[t+1])
        P_new = P_filts[t] + G @ (P_s - P_preds[t+1]) @ G.T
        return (m_new, P_new), (m_new, P_new)

    T = ys.shape[0]
    _, (ms_rev, Ps_rev) = jax.lax.scan(
        bwd, (m_filts[-1], P_filts[-1]), jnp.arange(T-2, -1, -1))
    ks_means = jnp.concatenate([ms_rev[::-1], m_filts[-1:]])
    ks_covs  = jnp.concatenate([Ps_rev[::-1], P_filts[-1:]])
    # Extract diagonal variances for the diagonal twist
    ks_vars = jax.vmap(jnp.diag)(ks_covs)
    return ks_means, ks_vars


#  Full-Jacobian DEER primitives (from debug_fulljac_coupled.py) 
def full_binary_operator(element_i, element_j):
    gti, hti = element_i
    gtj, htj = element_j
    a = jnp.einsum('...ij,...jk->...ik', gtj, gti)
    b = jnp.einsum('...ij,...j->...i', gtj, hti) + htj
    return a, b


def full_matmul_recursive(mats, vecs, y0):
    D = mats.shape[-1]
    eye = jnp.eye(D, dtype=mats.dtype)[None]
    first = jnp.concatenate((eye, mats), axis=0)
    second = jnp.concatenate((y0[None], vecs), axis=0)
    _, yt = jax.lax.associative_scan(full_binary_operator, (first, second))
    return yt


def full_jac_estimator(state, inputs, params, dyn_func):
    def f_of_z(z):
        return dyn_func(z, inputs, params)
    f_val = f_of_z(state)
    J = jax.jacfwd(f_of_z)(state)
    return f_val, J


def seq1d_fulljac(func, y0, xinp, params, yinit_guess=None,
                  max_iter=100, damp_factor=1.0, clip_val=1e8):
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    T_ = xinp_flat.shape[0]
    D = y0.shape[-1]
    if yinit_guess is None:
        yinit_guess = jnp.zeros((T_, D), dtype=y0.dtype)
    tol, rtol = 1e-4, 1e-3

    def shifter(y):
        return jnp.concatenate((y0[None, :], y[:-1, :]), axis=0)

    def iter_func(state):
        err, yt, it = state
        ytp = shifter(yt)
        f_vals, Js = jax.vmap(full_jac_estimator, in_axes=(0, 0, None, None))(
            ytp, xinp, params, func)
        Js = jnp.clip(damp_factor * Js, -clip_val, clip_val)
        gts = -Js
        rhs = f_vals + jnp.einsum('tij,tj->ti', gts, ytp)
        yt_next = full_matmul_recursive(-gts, rhs, y0)[1:]
        yt_next = jnp.clip(yt_next, -clip_val, clip_val)
        yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)
        err = jnp.max(jnp.abs(yt_next - yt) - rtol * jnp.abs(yt))
        return err, yt_next, it + 1

    def cond(state):
        err, _, it = state
        return jnp.logical_and(err > tol, it < max_iter)

    err0 = jnp.array(1e10, dtype=yinit_guess.dtype)
    _, yt, iters = jax.lax.while_loop(
        cond, iter_func, (err0, yinit_guess, jnp.array(0, dtype=jnp.int32)))
    return yt, iters


#  Twisted-PF step and weight fn (diagonal twist) 
def make_twisted_step_l96(dt, F):
    def step_fn(x_prev, driver, params):
        _, noise, twist = driver
        sq = params['sigma_q']
        mu_t = twist['mean']
        tau_t = jnp.exp(twist['log_scale'])
        f_prev = rk4_step(x_prev, dt, F)
        prec_q = 1.0 / sq**2
        prec_t = 1.0 / tau_t**2
        var = 1.0 / (prec_q + prec_t)
        mean = var * (prec_q * f_prev + prec_t * mu_t)
        return mean + jnp.sqrt(var) * noise
    return step_fn


def make_log_weight_fn_l96(dt, F):
    def log_weight_fn(x_prev, x_next, obs, twist, params):
        sq, sr = params['sigma_q'], params['sigma_r']
        mu_t = twist['mean']
        tau_t = jnp.exp(twist['log_scale'])
        D_ = x_next.shape[0]
        f_prev = rk4_step(x_prev, dt, F)
        log_lik = (-0.5 * jnp.sum((obs - x_next)**2) / sr**2
                   - 0.5 * D_ * jnp.log(2*jnp.pi*sr**2))
        log_trans = (-0.5 * jnp.sum((x_next - f_prev)**2) / sq**2
                     - 0.5 * D_ * jnp.log(2*jnp.pi*sq**2))
        prec_q = 1.0 / sq**2
        prec_t = 1.0 / tau_t**2
        var = 1.0 / (prec_q + prec_t)
        mean = var * (prec_q * f_prev + prec_t * mu_t)
        log_prop = (-0.5 * jnp.sum((x_next - mean)**2 / var)
                    - 0.5 * jnp.sum(jnp.log(2*jnp.pi*var)))
        return log_lik + log_trans - log_prop
    return log_weight_fn


def run_fulljac_tpf(tpf, x0s, tape, params, max_iter=100, damp=1.0):
    obs, noise, tw = tape

    def solve_one(x0, noise_i):
        ptape = (obs, noise_i, tw)
        return seq1d_fulljac(tpf._step, x0, ptape, params,
                              max_iter=max_iter, damp_factor=damp)
    states, iters = jax.vmap(solve_one)(x0s, noise)
    return states, iters


#  Simulate + run
# N=64 is deliberately small here, not the project-wide default (N=256).
# Full-Jacobian DEER stores (T, N, D, D) = 2000*N*D^2 floats per particle, so at
# D=40 the scan tensor is 2000 * 64 * 1600 ≈ 800 MB.  Bumping to N=256 OOMs on
# a 40 GB GPU at D=40, and the smoothing RMSE story (twisted flat at ~0.3 vs
# baselines diverging) is *stronger* at small N — so we keep N=64.
def run_l96(D, T=2000, N=64, dt=0.02, F=8.0, sigma_q=0.2, sigma_r=0.5,
            seed=0, use_damp=1.0):
    key = jr.PRNGKey(seed)
    key, ik, qk, rk = jr.split(key, 4)

    # Warm up the system onto the attractor before recording
    x0 = 1.0 + 0.01 * jr.normal(ik, (D,))
    x0 = x0.at[0].add(0.01)
    def spinup(x, _):
        return rk4_step(x, dt, F), None
    x_init, _ = jax.lax.scan(spinup, x0, None, length=1000)

    # Simulate
    q_noise = jr.normal(qk, (T, D))
    r_noise = jr.normal(rk, (T, D))
    def sim(x, inp):
        q, r = inp
        xn = rk4_step(x, dt, F) + sigma_q * q
        return xn, (xn, xn + sigma_r * r)
    _, (true_xs, ys) = jax.lax.scan(sim, x_init, (q_noise, r_noise))

    # Twist: EKF + RTS
    ks_means, ks_vars = ekf_smoother_l96(
        ys, dt, F, sigma_q, sigma_r, x_init, 0.5 * jnp.eye(D))
    # clamp variances to avoid log(0)
    ks_vars = jnp.maximum(ks_vars, 1e-4)
    twist = {'mean': ks_means, 'log_scale': 0.5 * jnp.log(ks_vars)}

    #  Twisted PF setup 
    step_fn = make_twisted_step_l96(dt, F)
    log_w_fn = make_log_weight_fn_l96(dt, F)
    tpf = TwistedParallelParticleFilter(step_fn, log_w_fn, D=D)

    key, tk, sk = jr.split(key, 3)
    tape = tpf.make_tape(ys, twist, tk, N)
    x0s = x_init[None, :] + 0.1 * jr.normal(sk, (N, D))

    params = {'sigma_q': sigma_q, 'sigma_r': sigma_r}

    #  TPF seq 
    run_seq = jax.jit(tpf.run_sequential)
    states_seq = run_seq(x0s, tape, params).block_until_ready()
    times_s = []
    for _ in range(3):
        t0 = time.perf_counter()
        states_seq = run_seq(x0s, tape, params).block_until_ready()
        times_s.append(time.perf_counter() - t0)
    t_seq = min(times_s)

    log_ws_s = jax.jit(tpf.compute_log_weights)(x0s, states_seq, tape, params)
    w_s = jax.nn.softmax(jnp.sum(log_ws_s, axis=1))
    tpf_seq_mean = jnp.einsum('n,ntd->td', w_s, states_seq)
    rmse_tpf_seq = float(jnp.sqrt(jnp.mean((tpf_seq_mean - true_xs)**2)))

    #  TPF par via full-Jacobian DEER 
    run_par = jax.jit(partial(run_fulljac_tpf, tpf), static_argnames=("max_iter",))
    states_par, iters = run_par(x0s, tape, params, max_iter=100)
    states_par.block_until_ready()
    times_p = []
    for _ in range(3):
        t0 = time.perf_counter()
        states_par, iters = run_par(x0s, tape, params, max_iter=100)
        states_par.block_until_ready()
        times_p.append(time.perf_counter() - t0)
    t_par = min(times_p)

    log_ws_p = jax.jit(tpf.compute_log_weights)(x0s, states_par, tape, params)
    w_p = jax.nn.softmax(jnp.sum(log_ws_p, axis=1))
    tpf_par_mean = jnp.einsum('n,ntd->td', w_p, states_par)
    rmse_tpf_par = float(jnp.sqrt(jnp.mean((tpf_par_mean - true_xs)**2)))

    #  Bootstrap PF 
    def l96_trans(x, p):
        return rk4_step(x, dt, F)
    def l96_ll(y, x, p):
        return -0.5 * jnp.sum((y - x)**2) / p['sigma_r']**2

    key, bk, fk, ik2 = jr.split(key, 4)
    x0_bpf = x_init[None, :] + 0.1 * jr.normal(ik2, (N, D))
    run_bpf = jax.jit(bootstrap_pf, static_argnames=("transition_mean", "log_lik"))
    _ = run_bpf(ys, x0_bpf, l96_trans, l96_ll, sigma_q, params, bk)
    t0 = time.perf_counter()
    bpf_filt, bpf_parts, bpf_lw, bpf_ess, _ = run_bpf(
        ys, x0_bpf, l96_trans, l96_ll, sigma_q, params, bk)
    bpf_filt.block_until_ready()
    t_bpf = time.perf_counter() - t0
    rmse_bpf = float(jnp.sqrt(jnp.mean((bpf_filt - true_xs)**2)))

    #  FFBSi 
    run_ff = jax.jit(ffbsi_smoother, static_argnames=("M", "transition_mean"))
    _ = run_ff(bpf_parts, bpf_lw, l96_trans, sigma_q, params, fk, M=N)
    t0 = time.perf_counter()
    bps = run_ff(bpf_parts, bpf_lw, l96_trans, sigma_q, params, fk, M=N)
    bps.block_until_ready()
    t_ff = time.perf_counter() - t0
    rmse_ff = float(jnp.sqrt(jnp.mean((bps.mean(axis=0) - true_xs)**2)))

    return {
        'D': D, 'T': T, 'N': N,
        'deer_iters': int(iters[0]),
        'results': [
            ('Bootstrap PF',   t_bpf, rmse_bpf),
            ('FFBSi',          t_ff,  rmse_ff),
            ('Twisted (seq)',  t_seq, rmse_tpf_seq),
            ('Twisted (par)',  t_par, rmse_tpf_par),
        ],
        'true_xs': true_xs,
        'tpf_par_mean': tpf_par_mean,
        'bpf_ess': bpf_ess,
    }


if __name__ == "__main__":
    outs = []
    for D in [10, 20, 40]:
        print(f"\n{'='*70}")
        print(f"Lorenz-96  D={D}, F=8, T=2000, N=64")
        print(f"{'='*70}")
        out = run_l96(D)
        outs.append(out)
        for n, t, r in out['results']:
            print(f"  {n:<18}  time={t:.4f}s   RMSE={r:.4f}")
        print(f"  full-Jac DEER iters: {out['deer_iters']}")
        seq_t = [t for n, t, _ in out['results'] if n == 'Twisted (seq)'][0]
        par_t = [t for n, t, _ in out['results'] if n == 'Twisted (par)'][0]
        ff_t  = [t for n, t, _ in out['results'] if n == 'FFBSi'][0]
        print(f"  Speedup vs seq:    {seq_t/par_t:.1f}x")
        print(f"  Speedup vs FFBSi:  {ff_t/par_t:.1f}x")

    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'Bootstrap PF': '#888888', 'FFBSi': '#1f77b4',
              'Twisted (seq)': '#2ca02c', 'Twisted (par)': '#d62728'}
    markers = {'Bootstrap PF': 'o', 'FFBSi': 's',
               'Twisted (seq)': '^', 'Twisted (par)': '*'}

    fig, axes = plt.subplots(1, len(outs), figsize=(5 * len(outs), 5))
    if len(outs) == 1: axes = [axes]
    for ax, out in zip(axes, outs):
        for n, t, r in out['results']:
            ax.scatter(t, r, s=200 if 'par' in n else 120,
                       c=colors[n], marker=markers[n], edgecolor='k',
                       linewidth=0.8, label=n, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wall-clock time (s)')
        ax.set_ylabel('RMSE vs true latent')
        ax.set_title(f"Lorenz-96 D={out['D']}  "
                     f"(DEER {out['deer_iters']} iters)")
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('twisted_pf_lorenz96.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved twisted_pf_lorenz96.png")
