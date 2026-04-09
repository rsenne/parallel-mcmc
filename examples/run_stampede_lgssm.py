"""
run_stampede_lgssm.py

STAMPEDE demo on a 1-D linear-Gaussian SSM.

Infers phi, sigma_q, sigma_r from simulated data using doubly-parallel PMCMC:
  - Outer loop: parallel MALA (DEER) across MCMC chain iterations
  - Inner loop: twisted particle smoother (sequential, fixed noise) for p(y|theta)

Uses adaptive smoother twists: the Kalman smoother is re-run at the current
theta inside each log-posterior evaluation, so twist quality never degrades.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.stampede import STAMPEDE

PHI_TRUE     = 0.9
SIGMA_Q_TRUE = 0.5
SIGMA_R_TRUE = 1.0

T = 500          # time series length
D = 1            # latent state dimension
N = 128          # particles for PF likelihood
K = 1000         # MCMC chain length
D_THETA = 3     # unconstrained parameter dimension

def kalman_smoother_1d(ys, phi, sigma_q, sigma_r):
    """RTS smoother for 1-D LGSSM. Returns smoother means and variances."""
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

def compute_twists(model_params, observations):
    """Compute RTS smoother twists at the current model parameters."""
    ys = observations[:, 0]  # (T,)
    sm, sP = kalman_smoother_1d(
        ys, model_params['phi'], model_params['sigma_q'], model_params['sigma_r'])
    sP = jnp.maximum(sP, 1e-8)  # numerical safety
    return {
        'mean':      sm[:, None],
        'log_scale': 0.5 * jnp.log(sP)[:, None],
    }

def twisted_step_fn(x_prev, driver, params):
    """Twisted transition combining prior with smoother twist."""
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

def pf_log_weight_fn(x_prev, x_next, obs, twist, params):
    """Incremental log importance weight (with full normalization constants)."""
    phi = params['phi']; sq = params['sigma_q']; sr = params['sigma_r']
    mu = twist['mean']; tau = jnp.exp(twist['log_scale'])

    # log p(y_t | x_t)  — full Gaussian log-pdf
    log_lik = (-0.5 * jnp.sum((obs - x_next)**2) / sr**2
               - 0.5 * D * jnp.log(2 * jnp.pi * sr**2))

    # log p(x_t | x_{t-1})
    log_trans = (-0.5 * jnp.sum((x_next - phi * x_prev)**2) / sq**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sq**2))

    # log q_twist(x_t | x_{t-1})
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    log_prop = (-0.5 * jnp.sum((x_next - mp)**2 / vp)
                - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * vp)))

    return log_lik + log_trans - log_prop

def theta_to_params(theta):
    return {
        'phi':     jnp.tanh(theta[0]),
        'sigma_q': jax.nn.softplus(theta[1]),
        'sigma_r': jax.nn.softplus(theta[2]),
    }

def params_to_theta(phi, sigma_q, sigma_r):
    return jnp.array([
        jnp.arctanh(phi),
        jnp.log(jnp.exp(sigma_q) - 1.0),
        jnp.log(jnp.exp(sigma_r) - 1.0),
    ])

def log_prior(theta):
    # Mildly informative: N(0, 2^2) on each unconstrained param
    # Prevents degenerate modes (phi->1, sigma_q->0)
    return -0.5 * jnp.sum(theta**2) / 2.0**2

key = jr.PRNGKey(42)
key, k1, k2, k3 = jr.split(key, 4)

def sim_step(x, qr):
    xn = PHI_TRUE * x + SIGMA_Q_TRUE * qr[0]
    return xn, (xn, xn + SIGMA_R_TRUE * qr[1])

_, (true_xs, ys) = jax.lax.scan(
    sim_step, jr.normal(k1),
    (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
)
ys_2d = ys[:, None]  # (T, 1)

key, noise_key, x0_key = jr.split(key, 3)
pf_noise = jr.normal(noise_key, (N, T, D))
x0s = jr.normal(x0_key, (N, D))

stampede = STAMPEDE(
    twisted_step_fn=twisted_step_fn,
    pf_log_weight_fn=pf_log_weight_fn,
    log_prior_fn=log_prior,
    theta_to_params_fn=theta_to_params,
    D_state=D,
    D_theta=D_THETA,
    N_particles=N,
    chain_length=K,
    max_deer_iter=50,
    damp_factor=1.0,
    clip_val=1.0,
    compute_twists_fn=compute_twists,
)

key, init_key = jr.split(key)
theta0 = params_to_theta(PHI_TRUE, SIGMA_Q_TRUE, SIGMA_R_TRUE)
theta0 = theta0 + 0.3 * jr.normal(init_key, theta0.shape)

print(f"True params:    phi={PHI_TRUE}, sigma_q={SIGMA_Q_TRUE}, sigma_r={SIGMA_R_TRUE}")
p0 = theta_to_params(theta0)
print(f"Initial params: phi={p0['phi']:.3f}, sigma_q={p0['sigma_q']:.3f}, sigma_r={p0['sigma_r']:.3f}")
print(f"\nT={T}, N={N} particles, K={K} MCMC steps")
print("=" * 70)

# twist_params placeholder
twist_params_dummy = {
    'mean':      jnp.zeros((T, D)),
    'log_scale': jnp.zeros((T, D)),
}
target_params = {
    'observations': ys_2d,
    'pf_noise':     pf_noise,
    'twist_params': twist_params_dummy,
    'x0s':          x0s,
}
params = {
    'step_size':     3e-3,
    'target_params': target_params,
}

print("\n--- Sequential PMCMC ---")
key, seq_key = jr.split(key)

run_seq = jax.jit(stampede.run_sequential)
_ = run_seq(seq_key, theta0, params)
jax.block_until_ready(_)

t0 = time.perf_counter()
chain_seq = run_seq(seq_key, theta0, params)
chain_seq.block_until_ready()
t_seq = time.perf_counter() - t0
print(f"  Time: {t_seq:.3f}s")

accept_ratio = 1.0 - jnp.mean(chain_seq[1:, 0] == chain_seq[:-1, 0])
print(f"  Accept ratio: {float(accept_ratio):.3f}")

print("\n--- Parallel PMCMC (STAMPEDE) ---")
key, par_key, deer_key = jr.split(key, 3)

params_par = {**params, 'key': deer_key}
key, yinit_key = jr.split(key)
yinit_guess = jr.normal(yinit_key, (K, D_THETA)) * 0.01

run_par = jax.jit(stampede.run_parallel)
chain_par, iters = run_par(par_key, theta0, yinit_guess, params_par)
jax.block_until_ready(chain_par)

t0 = time.perf_counter()
chain_par, iters = run_par(par_key, theta0, yinit_guess, params_par)
chain_par.block_until_ready()
t_par = time.perf_counter() - t0

deer_iters = int(iters)
speedup = t_seq / t_par
print(f"  Time: {t_par:.3f}s  ({deer_iters} DEER iters)")
print(f"  Speedup: {speedup:.1f}x")

def chain_to_params(chain):
    return jax.vmap(theta_to_params)(chain)

params_seq = chain_to_params(chain_seq)
params_par = chain_to_params(chain_par)

burnin = K // 4
print(f"\n{'':>12} {'True':>10} {'Seq mean':>10} {'Par mean':>10}"
      f" {'Seq std':>10} {'Par std':>10}")
print("-" * 62)
for name, true_val in [('phi', PHI_TRUE), ('sigma_q', SIGMA_Q_TRUE), ('sigma_r', SIGMA_R_TRUE)]:
    s = params_seq[name][burnin:]
    p = params_par[name][burnin:]
    print(f"{name:>12} {true_val:>10.3f} {float(s.mean()):>10.3f}"
          f" {float(p.mean()):>10.3f} {float(s.std()):>10.4f}"
          f" {float(p.std()):>10.4f}")

max_diff = float(jnp.max(jnp.abs(chain_seq - chain_par)))
print(f"\nMax |seq - par| in theta space: {max_diff:.6f}")

fig, axes = plt.subplots(3, 2, figsize=(14, 8))

names  = ['phi', 'sigma_q', 'sigma_r']
trues  = [PHI_TRUE, SIGMA_Q_TRUE, SIGMA_R_TRUE]
labels = [r'$\phi$', r'$\sigma_q$', r'$\sigma_r$']

for i, (name, tv, lab) in enumerate(zip(names, trues, labels)):
    ax = axes[i, 0]
    ax.plot(params_seq[name], 'b-', alpha=0.5, lw=0.5, label='Sequential')
    ax.plot(params_par[name], 'r-', alpha=0.5, lw=0.5, label='Parallel')
    ax.axhline(tv, color='k', ls='--', lw=1.5, label='True')
    ax.set_ylabel(lab)
    ax.set_title(f'{lab} trace')
    if i == 0:
        ax.legend(fontsize=8)

    ax = axes[i, 1]
    ax.hist(params_seq[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.5, color='b', label='Sequential')
    ax.hist(params_par[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.5, color='r', label='Parallel')
    ax.axvline(tv, color='k', ls='--', lw=1.5)
    ax.set_title(f'{lab} posterior')
    if i == 0:
        ax.legend(fontsize=8)

axes[-1, 0].set_xlabel('MCMC iteration')
axes[-1, 1].set_xlabel('Parameter value')

plt.suptitle(
    f'STAMPEDE — LGSSM (T={T}, N={N}, K={K})\n'
    f'Sequential: {t_seq:.2f}s | Parallel: {t_par:.2f}s '
    f'({deer_iters} DEER iters, {speedup:.1f}x speedup)',
    fontsize=12,
)
plt.tight_layout()
plt.savefig('stampede_lgssm.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved plot to stampede_lgssm.png")
