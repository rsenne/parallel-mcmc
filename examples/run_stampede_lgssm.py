"""
run_stampede_lgssm.py

Three-way comparison on a 1-D linear-Gaussian SSM:

  1. Standard PMCMC  — bootstrap PF (resampling) + random-walk MH  (sequential)
  2. Twisted PMCMC   — twisted PF (no resampling) + MALA            (sequential)
  3. STAMPEDE         — twisted PF (no resampling) + parallel MALA   (DEER)

The twisted methods use a block-refresh strategy: the PF noise is refreshed
every B MCMC steps.  Within each block DEER parallelises the MALA chain,
while refreshing between blocks ensures the chain targets the correct
posterior p(theta | y) rather than the noise-conditional p(theta | y, u).

Infers phi, sigma_q, sigma_r from simulated data.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.stampede import STAMPEDE

# True parameters
PHI_TRUE     = 0.9
SIGMA_Q_TRUE = 0.5
SIGMA_R_TRUE = 1.0

T = 500          # time series length
D = 1            # latent state dimension
N = 256          # particles for PF likelihood
K_TOTAL = 2000   # total MCMC chain length
B = 80           # number of blocks (noise refreshed between blocks) — K_BLOCK=25 sweet spot
K_BLOCK = K_TOTAL // B   # MCMC steps per block
D_THETA = 3      # unconstrained parameter dimension

# Shared model code
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
    return -0.5 * jnp.sum(theta**2) / 2.0**2

# Backward information filter (for optimal adaptive twists) 
def backward_information_filter_1d(ys, phi, sigma_q, sigma_r):
    """Optimal twist psi_t(x_t) = p(y_{t:T}|x_t) via backward info filter."""
    prec_obs = 1.0 / sigma_r**2
    sq2 = sigma_q**2
    def step(carry, y):
        J, h = carry
        denom = 1.0 + sq2 * J
        J_pred = phi**2 * J / denom
        h_pred = phi * h / denom
        return (J_pred + prec_obs, h_pred + y * prec_obs), \
               (J_pred + prec_obs, h_pred + y * prec_obs)
    J_T = prec_obs
    h_T = ys[-1] * prec_obs
    _, (Js_rev, hs_rev) = jax.lax.scan(step, (J_T, h_T), ys[-2::-1])
    Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_T])])
    hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_T])])
    return hs / Js, 1.0 / Js

def compute_twists(model_params, observations):
    ys = observations[:, 0]
    twist_means, twist_vars = backward_information_filter_1d(
        ys, model_params['phi'], model_params['sigma_q'], model_params['sigma_r'])
    return {'mean': twist_means[:, None], 'log_scale': 0.5 * jnp.log(twist_vars)[:, None]}

# Twisted PF step + weights (for methods 2 & 3) 
def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi, sq = params['phi'], params['sigma_q']
    mu  = twist['mean']; tau = jnp.exp(twist['log_scale'])
    pq  = 1.0 / sq**2; pt = 1.0 / tau**2
    vp  = 1.0 / (pq + pt)
    mp  = vp * (pq * phi * x_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise

def pf_log_weight_fn(x_prev, x_next, obs, twist, params):
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


# Method 1: Standard PMCMC (bootstrap PF + random-walk MH)
def bootstrap_pf_log_ml(ys, phi, sigma_q, sigma_r, key, N):
    """
    Standard bootstrap PF with systematic resampling.
    Returns log p̂(y_{1:T} | theta).
    """
    def _log_normal(x, mu, sigma):
        return -0.5 * ((x - mu) / sigma)**2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)

    def _systematic_resample(key, log_w, particles):
        w = jax.nn.softmax(log_w)
        N = w.shape[0]
        u = (jnp.arange(N) + jr.uniform(key)) / N
        cumw = jnp.cumsum(w)
        idx = jnp.searchsorted(cumw, u)
        idx = jnp.clip(idx, 0, N - 1)
        return particles[idx]

    def step(carry, yt):
        particles, key, log_ml_acc = carry
        key, trans_key, resamp_key = jr.split(key, 3)

        # propagate
        particles = phi * particles + sigma_q * jr.normal(trans_key, particles.shape)

        # weight
        log_w = jax.vmap(lambda x: _log_normal(yt, x, sigma_r))(particles)

        # accumulate log-ML: log(1/N * sum exp(log_w))
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))
        log_ml_acc = log_ml_acc + log_ml_inc

        # resample
        particles = _systematic_resample(resamp_key, log_w, particles)

        return (particles, key, log_ml_acc), None

    key, init_key = jr.split(key)
    particles = jr.normal(init_key, (N,))
    (_, _, log_ml), _ = jax.lax.scan(step, (particles, key, 0.0), ys)
    return log_ml


def standard_pmcmc(key, theta0, ys, N, K, proposal_scale):
    """
    Standard particle marginal MH (Andrieu et al. 2010).
    Random-walk proposal + bootstrap PF for likelihood.
    """
    def step(carry, key):
        theta, log_post = carry
        prop_key, pf_key, accept_key = jr.split(key, 3)

        # random-walk proposal
        theta_star = theta + proposal_scale * jr.normal(prop_key, theta.shape)

        # evaluate log posterior at proposal
        p_star = theta_to_params(theta_star)
        log_ml_star = bootstrap_pf_log_ml(
            ys, p_star['phi'], p_star['sigma_q'], p_star['sigma_r'], pf_key, N)
        log_post_star = log_ml_star + log_prior(theta_star)

        # MH accept/reject
        log_alpha = log_post_star - log_post
        accept = jnp.log(jr.uniform(accept_key)) < log_alpha
        theta_new = jnp.where(accept, theta_star, theta)
        log_post_new = jnp.where(accept, log_post_star, log_post)

        return (theta_new, log_post_new), theta_new

    # initial log posterior
    key, pf_key = jr.split(key)
    p0 = theta_to_params(theta0)
    log_post0 = bootstrap_pf_log_ml(
        ys, p0['phi'], p0['sigma_q'], p0['sigma_r'], pf_key, N) + log_prior(theta0)

    keys = jr.split(key, K)
    _, chain = jax.lax.scan(step, (theta0, log_post0), keys)
    return chain

# Simulate data
key = jr.PRNGKey(42)
key, k1, k2, k3 = jr.split(key, 4)

def sim_step(x, qr):
    xn = PHI_TRUE * x + SIGMA_Q_TRUE * qr[0]
    return xn, (xn, xn + SIGMA_R_TRUE * qr[1])

_, (true_xs, ys) = jax.lax.scan(
    sim_step, jr.normal(k1),
    (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
)
ys_2d = ys[:, None]

print(f"True params:    phi={PHI_TRUE}, sigma_q={SIGMA_Q_TRUE}, sigma_r={SIGMA_R_TRUE}")

key, init_key = jr.split(key)
theta0 = params_to_theta(PHI_TRUE, SIGMA_Q_TRUE, SIGMA_R_TRUE)
theta0 = theta0 + 0.3 * jr.normal(init_key, theta0.shape)

p0 = theta_to_params(theta0)
print(f"Initial params: phi={p0['phi']:.3f}, sigma_q={p0['sigma_q']:.3f}, sigma_r={p0['sigma_r']:.3f}")
print(f"\nT={T}, N={N} particles, K={K_TOTAL} MCMC steps ({B} blocks x {K_BLOCK})")
print("=" * 70)


# Method 1: Standard PMCMC
print("\n--- [1] Standard PMCMC (bootstrap PF + RW-MH) ---")
key, std_key = jr.split(key)

run_std = jax.jit(standard_pmcmc, static_argnames=("N", "K"))
chain_std = run_std(std_key, theta0, ys, N, K_TOTAL, proposal_scale=0.02)
chain_std.block_until_ready()

t0 = time.perf_counter()
chain_std = run_std(std_key, theta0, ys, N, K_TOTAL, proposal_scale=0.02)
chain_std.block_until_ready()
t_std = time.perf_counter() - t0

accept_std = 1.0 - jnp.mean(chain_std[1:, 0] == chain_std[:-1, 0])
print(f"  Time: {t_std:.3f}s | Accept: {float(accept_std):.3f}")


# Methods 2 & 3: Block-refresh twisted PMCMC and STAMPEDE
# Between blocks, refresh PF noise so the chain targets the correct
# marginal posterior p(theta|y), not the noise-conditional p(theta|y,u).

twist_params_dummy = {'mean': jnp.zeros((T, D)), 'log_scale': jnp.zeros((T, D))}

# Build one STAMPEDE instance per block length
stampede = STAMPEDE(
    twisted_step_fn=twisted_step_fn,
    pf_log_weight_fn=pf_log_weight_fn,
    log_prior_fn=log_prior,
    theta_to_params_fn=theta_to_params,
    D_state=D,
    D_theta=D_THETA,
    N_particles=N,
    chain_length=K_BLOCK,
    max_deer_iter=max(100, 2 * K_BLOCK),
    damp_factor=1.0,
    clip_val=1.0,
    compute_twists_fn=compute_twists,
)

run_seq_block = jax.jit(stampede.run_sequential)
run_par_block = jax.jit(stampede.run_parallel)

#  Method 2: Sequential twisted PMCMC with block-refresh 
print(f"\n--- [2] Twisted PMCMC (MALA, {B} blocks x {K_BLOCK}, noise refresh) ---")

# Warmup
key, wkey, wnkey, wx0key = jr.split(key, 4)
pf_noise_w = jr.normal(wnkey, (N, T, D))
x0s_w = jr.normal(wx0key, (N, D))
params_w = {'step_size': 3e-3, 'grad_max_norm': None, 'target_params': {
    'observations': ys_2d, 'pf_noise': pf_noise_w,
    'twist_params': twist_params_dummy, 'x0s': x0s_w,
}}
_ = run_seq_block(wkey, theta0, params_w)
jax.block_until_ready(_)

chain_seq_blocks = []
theta_curr = theta0
t_seq_total = 0.0

for b in range(B):
    key, seq_key, noise_key, x0_key = jr.split(key, 4)
    pf_noise = jr.normal(noise_key, (N, T, D))
    x0s = jr.normal(x0_key, (N, D))
    target_params = {
        'observations': ys_2d,
        'pf_noise':     pf_noise,
        'twist_params': twist_params_dummy,
        'x0s':          x0s,
    }
    params_block = {'step_size': 3e-3, 'grad_max_norm': None, 'target_params': target_params}

    t0 = time.perf_counter()
    chain_block = run_seq_block(seq_key, theta_curr, params_block)
    chain_block.block_until_ready()
    t_seq_total += time.perf_counter() - t0

    chain_seq_blocks.append(chain_block)
    theta_curr = chain_block[-1]

chain_seq = jnp.concatenate(chain_seq_blocks, axis=0)
accept_seq = 1.0 - jnp.mean(chain_seq[1:, 0] == chain_seq[:-1, 0])
print(f"  Time: {t_seq_total:.3f}s | Accept: {float(accept_seq):.3f}")

#  Method 3: STAMPEDE with block-refresh 
print(f"\n--- [3] STAMPEDE (parallel MALA, {B} blocks x {K_BLOCK}, noise refresh) ---")

# Warm-start: short sequential MALA using the SAME DEER key so the first
# K_WARMUP positions lie on DEER's chain and only the tail needs solving.
K_WARMUP = max(1, K_BLOCK // 5)

def _make_yinit(key, theta_curr, params_block):
    drivers = jr.split(key, K_BLOCK)[:K_WARMUP]
    def _step(state, driver):
        nxt = stampede.sampler.mala_fxn_for_seq(state, driver, params_block)
        return nxt, nxt
    _, warmup_chain = jax.lax.scan(_step, theta_curr, drivers)
    tail = jnp.broadcast_to(warmup_chain[-1], (K_BLOCK - K_WARMUP, D_THETA))
    return jnp.concatenate([warmup_chain, tail], axis=0)

_make_yinit_jit = jax.jit(_make_yinit)

# Warmup (JIT compile)
key, wkey2, wdkey, wnkey2, wx0key2 = jr.split(key, 5)
pf_noise_w2 = jr.normal(wnkey2, (N, T, D))
x0s_w2 = jr.normal(wx0key2, (N, D))
params_w2 = {'step_size': 3e-3, 'key': wdkey, 'grad_max_norm': None, 'target_params': {
    'observations': ys_2d, 'pf_noise': pf_noise_w2,
    'twist_params': twist_params_dummy, 'x0s': x0s_w2,
}}
yinit_w = _make_yinit_jit(wkey2, theta0, params_w2)
yinit_w.block_until_ready()
_ = run_par_block(wkey2, theta0, yinit_w, params_w2)
jax.block_until_ready(_[0])

chain_par_blocks = []
theta_curr = theta0
t_par_total = 0.0
deer_iters_total = 0

for b in range(B):
    key, par_key, deer_key, noise_key, x0_key = jr.split(key, 5)
    pf_noise = jr.normal(noise_key, (N, T, D))
    x0s = jr.normal(x0_key, (N, D))
    target_params = {
        'observations': ys_2d,
        'pf_noise':     pf_noise,
        'twist_params': twist_params_dummy,
        'x0s':          x0s,
    }
    params_block = {'step_size': 3e-3, 'key': deer_key, 'grad_max_norm': None,
                    'target_params': target_params}

    t0 = time.perf_counter()
    # warm-start yinit on the same key DEER will use
    yinit = _make_yinit_jit(par_key, theta_curr, params_block)
    chain_block, iters = run_par_block(par_key, theta_curr, yinit, params_block)
    chain_block.block_until_ready()
    t_par_total += time.perf_counter() - t0

    chain_par_blocks.append(chain_block)
    theta_curr = chain_block[-1]
    deer_iters_total += int(iters)

chain_par = jnp.concatenate(chain_par_blocks, axis=0)
deer_iters_avg = deer_iters_total / B
print(f"  Time: {t_par_total:.3f}s | DEER iters (avg): {deer_iters_avg:.0f}")


# Results
speedup_vs_seq = t_seq_total / t_par_total
speedup_vs_std = t_std / t_par_total

print(f"\n{'Method':>30} {'Time':>8} {'vs Std':>8} {'vs Seq':>8}")
print("-" * 58)
print(f"{'Standard PMCMC':>30} {t_std:>7.2f}s {'1.0x':>8} {t_std/t_seq_total:>7.1f}x")
print(f"{'Twisted seq PMCMC':>30} {t_seq_total:>7.2f}s {t_std/t_seq_total:>7.1f}x {'1.0x':>8}")
print(f"{'STAMPEDE':>30} {t_par_total:>7.2f}s {speedup_vs_std:>7.1f}x {speedup_vs_seq:>7.1f}x")

params_std = jax.vmap(theta_to_params)(chain_std)
params_seq_c = jax.vmap(theta_to_params)(chain_seq)
params_par_c = jax.vmap(theta_to_params)(chain_par)

burnin = K_TOTAL // 4
print(f"\n{'':>12} {'True':>8} {'Std':>8} {'TwSeq':>8} {'STAMPEDE':>8}"
      f" {'Std σ':>8} {'TwSeq σ':>8} {'STP σ':>8}")
print("-" * 80)
for name, tv in [('phi', PHI_TRUE), ('sigma_q', SIGMA_Q_TRUE), ('sigma_r', SIGMA_R_TRUE)]:
    s0 = params_std[name][burnin:]
    s1 = params_seq_c[name][burnin:]
    s2 = params_par_c[name][burnin:]
    print(f"{name:>12} {tv:>8.3f} {float(s0.mean()):>8.3f} {float(s1.mean()):>8.3f}"
          f" {float(s2.mean()):>8.3f} {float(s0.std()):>8.4f}"
          f" {float(s1.std()):>8.4f} {float(s2.std()):>8.4f}")

# Bias diagnostic (distance from true / posterior std)
print(f"\n{'':>12} {'Std bias':>12} {'TwSeq bias':>12} {'STP bias':>12}")
print("-" * 52)
for name, tv in [('phi', PHI_TRUE), ('sigma_q', SIGMA_Q_TRUE), ('sigma_r', SIGMA_R_TRUE)]:
    s0 = params_std[name][burnin:]
    s1 = params_seq_c[name][burnin:]
    s2 = params_par_c[name][burnin:]
    b0 = float(s0.mean() - tv) / float(s0.std())
    b1 = float(s1.mean() - tv) / float(s1.std())
    b2 = float(s2.mean() - tv) / float(s2.std())
    print(f"{name:>12} {b0:>+11.2f}σ {b1:>+11.2f}σ {b2:>+11.2f}σ")

# Plot

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(14, 9))

names  = ['phi', 'sigma_q', 'sigma_r']
trues  = [PHI_TRUE, SIGMA_Q_TRUE, SIGMA_R_TRUE]
labels = [r'$\phi$', r'$\sigma_q$', r'$\sigma_r$']

for i, (name, tv, lab) in enumerate(zip(names, trues, labels)):
    # Trace plots
    ax = axes[i, 0]
    ax.plot(params_std[name],    color='gray', alpha=0.4, lw=0.4, label='Standard')
    ax.plot(params_seq_c[name],  color='b',    alpha=0.5, lw=0.5, label='Twisted seq')
    ax.plot(params_par_c[name],  color='r',    alpha=0.5, lw=0.5, label='STAMPEDE')
    ax.axhline(tv, color='k', ls='--', lw=1.5, label='True')
    # Mark block boundaries
    for b_idx in range(1, B):
        ax.axvline(b_idx * K_BLOCK, color='r', ls=':', lw=0.5, alpha=0.3)
    ax.set_ylabel(lab)
    ax.set_title(f'{lab} trace')
    if i == 0:
        ax.legend(fontsize=7, ncol=2)

    # Histograms
    ax = axes[i, 1]
    ax.hist(params_std[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.35, color='gray', label='Standard')
    ax.hist(params_seq_c[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.45, color='b', label='Twisted seq')
    ax.hist(params_par_c[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.45, color='r', label='STAMPEDE')
    ax.axvline(tv, color='k', ls='--', lw=1.5)
    ax.set_title(f'{lab} posterior')
    if i == 0:
        ax.legend(fontsize=7)

axes[-1, 0].set_xlabel('MCMC iteration')
axes[-1, 1].set_xlabel('Parameter value')

plt.suptitle(
    f'STAMPEDE — LGSSM  (T={T}, N={N}, K={K_TOTAL}, {B} blocks)\n'
    f'Standard: {t_std:.1f}s | Twisted seq: {t_seq_total:.1f}s | '
    f'STAMPEDE: {t_par_total:.1f}s ({deer_iters_avg:.0f} DEER iters/block, '
    f'{speedup_vs_std:.1f}x vs std, {speedup_vs_seq:.1f}x vs seq)',
    fontsize=11,
)
plt.tight_layout()
plt.savefig('stampede_lgssm.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved plot to stampede_lgssm.png")
