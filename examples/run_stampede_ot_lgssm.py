"""
run_stampede_ot_lgssm.py

OT-PF + Parallel MALA — differentiable bootstrap-equivalent PMCMC with
DEER-parallelised MCMC chain.

Three-way comparison on a 1-D linear-Gaussian SSM:

  1. Standard PMCMC       — bootstrap PF (discrete resampling) + RW-MH
  2. OT-PF + MALA (seq)    — OT PF (Sinkhorn resampling) + sequential MALA
  3. OT-PF + MALA (par)    — OT PF + DEER-parallel MALA chain (STAMPEDE-style)

Unlike twisted STAMPEDE, the OT PF introduces no proposal approximation —
it is a drop-in for bootstrap with differentiable Sinkhorn resampling.
MALA provides gradient-based MCMC; DEER parallelises the chain over K.

Block-refresh (fresh PF noise per block) ensures correct targeting of
p(theta|y) rather than p(theta|y,u).
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.particle_filter import ot_resample, ot_resample_ste
from src.samplers import ParallelMALA

# True parameters 
PHI_TRUE     = 0.9
SIGMA_Q_TRUE = 0.5
SIGMA_R_TRUE = 1.0

T = 200          # time series length
D = 1            # latent state dimension
N = 128          # particles for PF likelihood
K_TOTAL = 496    # total MCMC chain length (sanity-check size)
B = 62           # number of blocks (noise refreshed between blocks)
K_BLOCK = K_TOTAL // B
D_THETA = 3      # unconstrained parameter dimension

OT_EPS   = 0.1
OT_ITERS = 30

# Parameter transforms (shared)
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



# OT-PF log marginal likelihood (differentiable)
def transition_fn(X, noise, params):
    """X_pred[:, i] = phi * X[:, i] + sigma_q * noise[:, i]."""
    return params['phi'] * X + params['sigma_q'] * noise


def log_lik_fn(X_pred, obs, params):
    """log N(obs; X_pred, sigma_r^2) — with the normalization constant."""
    sr = params['sigma_r']
    diff = obs - X_pred                             # (D, N) - (D,) broadcasts
    quad = -0.5 * jnp.sum(diff ** 2, axis=0) / sr**2
    norm = -D * (jnp.log(sr) + 0.5 * jnp.log(2.0 * jnp.pi))
    return quad + norm                              # (N,)


def ot_pf_log_ml(ys, noise_tape, x0s, model_params, ot_eps, ot_iters):
    """
    Run OT-PF forward and return log p̂(y_{1:T} | θ).

    Parameters
    ----------
    ys          : (T, D)     observations
    noise_tape  : (T, D, N)  pre-drawn transition noise
    x0s         : (D, N)     initial particles
    model_params: dict       phi, sigma_q, sigma_r
    """
    s0 = x0s.ravel()                  # (D*N,)

    def step(carry, driver):
        state, log_ml = carry
        obs, noise = driver
        X_pred = transition_fn(state.reshape(D, N), noise, model_params)  # (D, N)
        log_w = log_lik_fn(X_pred, obs, model_params)                      # (N,)
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))
        X_new = ot_resample_ste(X_pred, log_w, ot_eps, ot_iters)           # (D, N)
        return (X_new.ravel(), log_ml + log_ml_inc), None

    (_, log_ml), _ = jax.lax.scan(step, (s0, 0.0), (ys, noise_tape))
    return log_ml


# Log posterior (for MALA) — OT-PF likelihood + prior
def log_posterior(theta, target_params):
    model_params = theta_to_params(theta)
    log_ml = ot_pf_log_ml(
        target_params['observations'],
        target_params['pf_noise'],
        target_params['x0s'],
        model_params,
        OT_EPS, OT_ITERS,
    )
    return log_ml + log_prior(theta)


# Standard PMCMC (bootstrap PF + RW-MH) — baseline
def bootstrap_pf_log_ml(ys, phi, sigma_q, sigma_r, key, N):
    def _log_normal(x, mu, sigma):
        return -0.5 * ((x - mu) / sigma)**2 - jnp.log(sigma) - 0.5 * jnp.log(2*jnp.pi)

    def _systematic_resample(key, log_w, particles):
        w = jax.nn.softmax(log_w)
        n = w.shape[0]
        u = (jnp.arange(n) + jr.uniform(key)) / n
        cumw = jnp.cumsum(w)
        idx = jnp.clip(jnp.searchsorted(cumw, u), 0, n - 1)
        return particles[idx]

    def step(carry, yt):
        particles, key, log_ml_acc = carry
        key, trans_key, resamp_key = jr.split(key, 3)
        particles = phi * particles + sigma_q * jr.normal(trans_key, particles.shape)
        log_w = jax.vmap(lambda x: _log_normal(yt, x, sigma_r))(particles)
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))
        particles = _systematic_resample(resamp_key, log_w, particles)
        return (particles, key, log_ml_acc + log_ml_inc), None

    key, init_key = jr.split(key)
    particles = jr.normal(init_key, (N,))
    (_, _, log_ml), _ = jax.lax.scan(step, (particles, key, 0.0), ys)
    return log_ml


def standard_pmcmc(key, theta0, ys, N, K, proposal_scale):
    def step(carry, key):
        theta, log_post = carry
        prop_key, pf_key, accept_key = jr.split(key, 3)
        theta_star = theta + proposal_scale * jr.normal(prop_key, theta.shape)
        p_star = theta_to_params(theta_star)
        log_ml_star = bootstrap_pf_log_ml(
            ys, p_star['phi'], p_star['sigma_q'], p_star['sigma_r'], pf_key, N)
        log_post_star = log_ml_star + log_prior(theta_star)
        log_alpha = log_post_star - log_post
        accept = jnp.log(jr.uniform(accept_key)) < log_alpha
        theta_new = jnp.where(accept, theta_star, theta)
        log_post_new = jnp.where(accept, log_post_star, log_post)
        return (theta_new, log_post_new), theta_new

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
print(f"Initial params: phi={p0['phi']:.3f}, sigma_q={p0['sigma_q']:.3f}, "
      f"sigma_r={p0['sigma_r']:.3f}")
print(f"\nT={T}, N={N} (OT-PF), K={K_TOTAL} ({B} blocks x {K_BLOCK})")
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


# Methods 2 & 3: OT-PF + MALA (sequential / parallel)
sampler = ParallelMALA(
    log_prob=log_posterior,
    dim=D_THETA,
    chain_length=K_BLOCK,
    max_iter=1000,
    clip_val=1.0,
    damp_factor=1.0,
)

run_seq_block = jax.jit(sampler.run_sequential_mala)
run_par_block = jax.jit(sampler.run_parallel_mala)

# Method 2: Sequential MALA with OT-PF 
print(f"\n--- [2] OT-PF + MALA seq ({B} blocks x {K_BLOCK}, noise refresh) ---")

# Warmup
key, wkey, wnkey, wx0key = jr.split(key, 4)
pf_noise_w = jr.normal(wnkey, (T, D, N))
x0s_w = jr.normal(wx0key, (D, N))
params_w = {'step_size': 1.5e-2, 'target_params': {
    'observations': ys_2d, 'pf_noise': pf_noise_w, 'x0s': x0s_w,
}}
_ = run_seq_block(wkey, theta0, params_w)
jax.block_until_ready(_)

chain_seq_blocks = []
theta_curr = theta0
t_seq_total = 0.0

for b in range(B):
    key, seq_key, noise_key, x0_key = jr.split(key, 4)
    pf_noise = jr.normal(noise_key, (T, D, N))
    x0s = jr.normal(x0_key, (D, N))
    params_block = {'step_size': 1.5e-2, 'target_params': {
        'observations': ys_2d,
        'pf_noise':     pf_noise,
        'x0s':          x0s,
    }}
    t0 = time.perf_counter()
    chain_block = run_seq_block(seq_key, theta_curr, params_block)
    chain_block.block_until_ready()
    t_seq_total += time.perf_counter() - t0

    chain_seq_blocks.append(chain_block)
    theta_curr = chain_block[-1]

chain_seq = jnp.concatenate(chain_seq_blocks, axis=0)
accept_seq = 1.0 - jnp.mean(chain_seq[1:, 0] == chain_seq[:-1, 0])
print(f"  Time: {t_seq_total:.3f}s | Accept: {float(accept_seq):.3f}")

# Method 3: Parallel MALA (DEER) with OT-PF 
print(f"\n--- [3] OT-PF + parallel MALA ({B} blocks x {K_BLOCK}, noise refresh) ---")

# Warm-start strategy. The log-posterior has genuinely large ‖∇‖ at small
# σ (the likelihood is sharply peaked there), and DEER's default
# yinit_guess = broadcast(theta_curr, K) is a bad initialisation: the
# true position-K answer is K sequential MALA steps away from theta_curr,
# so intermediate DEER iterates pass through bad-σ regions before
# converging. We pay for K_WARMUP sequential MALA steps to initialise
# yinit[:K_WARMUP] exactly, then fill the tail with the warmup endpoint —
# puts DEER's starting guess inside the posterior mode so intermediate
# iterates never touch the steep regions that saturate the clip.
K_WARMUP = K_BLOCK // 5     # 20 of 100 steps; pay 20% sequential cost

def _make_yinit(key, theta_curr, params_block):
    """K_WARMUP sequential MALA steps using the SAME key DEER will use,
    then broadcast the endpoint. The head warmup[:K_WARMUP] lies exactly
    on the chain DEER is solving (since drivers are jr.split(key, K) and
    drivers[:K_WARMUP] matches), so DEER keeps those positions fixed and
    only refines the tail — collapsing the wandering problem."""
    drivers = jr.split(key, K_BLOCK)[:K_WARMUP]
    def _step(state, driver):
        nxt = sampler.mala_fxn_for_seq(state, driver, params_block)
        return nxt, nxt
    _, warmup_chain = jax.lax.scan(_step, theta_curr, drivers)
    tail = jnp.broadcast_to(warmup_chain[-1], (K_BLOCK - K_WARMUP, D_THETA))
    return jnp.concatenate([warmup_chain, tail], axis=0)

_make_yinit_jit = jax.jit(_make_yinit)

# JIT warmup
key, wkey2, wdkey, wnkey2, wx0key2, wykey = jr.split(key, 6)
pf_noise_w2 = jr.normal(wnkey2, (T, D, N))
x0s_w2 = jr.normal(wx0key2, (D, N))
params_w2 = {'step_size': 1.5e-2, 'key': wdkey, 'target_params': {
    'observations': ys_2d, 'pf_noise': pf_noise_w2, 'x0s': x0s_w2,
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
    pf_noise = jr.normal(noise_key, (T, D, N))
    x0s = jr.normal(x0_key, (D, N))
    params_block = {'step_size': 1.5e-2, 'key': deer_key, 'target_params': {
        'observations': ys_2d,
        'pf_noise':     pf_noise,
        'x0s':          x0s,
    }}

    t0 = time.perf_counter()
    # warmup and DEER share par_key so warmup_chain[:K_WARMUP] lies on the
    # exact chain DEER is solving
    yinit = _make_yinit_jit(par_key, theta_curr, params_block)
    chain_block, iters = run_par_block(par_key, theta_curr, yinit, params_block)
    chain_block.block_until_ready()
    t_par_total += time.perf_counter() - t0

    chain_par_blocks.append(chain_block)
    theta_curr = chain_block[-1]
    deer_iters_total += int(iters)

chain_par = jnp.concatenate(chain_par_blocks, axis=0)
deer_iters_avg = deer_iters_total / B
print(f"  Time: {t_par_total:.3f}s | DEER iters (avg): {deer_iters_avg:.0f} "
      f"(K_WARMUP={K_WARMUP})")


# Results
speedup_vs_seq = t_seq_total / t_par_total
speedup_vs_std = t_std / t_par_total

print(f"\n{'Method':>30} {'Time':>8} {'vs Std':>8} {'vs Seq':>8}")
print("-" * 58)
print(f"{'Standard PMCMC':>30} {t_std:>7.2f}s {'1.0x':>8} "
      f"{t_std/t_seq_total:>7.1f}x")
print(f"{'OT-PF + MALA seq':>30} {t_seq_total:>7.2f}s "
      f"{t_std/t_seq_total:>7.1f}x {'1.0x':>8}")
print(f"{'OT-PF + parallel MALA':>30} {t_par_total:>7.2f}s "
      f"{speedup_vs_std:>7.1f}x {speedup_vs_seq:>7.1f}x")

params_std = jax.vmap(theta_to_params)(chain_std)
params_seq_c = jax.vmap(theta_to_params)(chain_seq)
params_par_c = jax.vmap(theta_to_params)(chain_par)

burnin = K_TOTAL // 4
print(f"\n{'':>12} {'True':>8} {'Std':>8} {'OT-seq':>8} {'OT-par':>8}"
      f" {'Std σ':>8} {'OTseq σ':>8} {'OTpar σ':>8}")
print("-" * 80)
for name, tv in [('phi', PHI_TRUE), ('sigma_q', SIGMA_Q_TRUE),
                 ('sigma_r', SIGMA_R_TRUE)]:
    s0 = params_std[name][burnin:]
    s1 = params_seq_c[name][burnin:]
    s2 = params_par_c[name][burnin:]
    print(f"{name:>12} {tv:>8.3f} {float(s0.mean()):>8.3f} "
          f"{float(s1.mean()):>8.3f} {float(s2.mean()):>8.3f} "
          f"{float(s0.std()):>8.4f} {float(s1.std()):>8.4f} "
          f"{float(s2.std()):>8.4f}")

print(f"\n{'':>12} {'Std bias':>12} {'OT-seq bias':>12} {'OT-par bias':>12}")
print("-" * 52)
for name, tv in [('phi', PHI_TRUE), ('sigma_q', SIGMA_Q_TRUE),
                 ('sigma_r', SIGMA_R_TRUE)]:
    s0 = params_std[name][burnin:]
    s1 = params_seq_c[name][burnin:]
    s2 = params_par_c[name][burnin:]
    b0 = float(s0.mean() - tv) / max(float(s0.std()), 1e-8)
    b1 = float(s1.mean() - tv) / max(float(s1.std()), 1e-8)
    b2 = float(s2.mean() - tv) / max(float(s2.std()), 1e-8)
    print(f"{name:>12} {b0:>+11.2f}σ {b1:>+11.2f}σ {b2:>+11.2f}σ")


# Plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(14, 9))

names  = ['phi', 'sigma_q', 'sigma_r']
trues  = [PHI_TRUE, SIGMA_Q_TRUE, SIGMA_R_TRUE]
labels = [r'$\phi$', r'$\sigma_q$', r'$\sigma_r$']

for i, (name, tv, lab) in enumerate(zip(names, trues, labels)):
    ax = axes[i, 0]
    ax.plot(params_std[name],   color='gray', alpha=0.4, lw=0.4, label='Standard')
    ax.plot(params_seq_c[name], color='b',    alpha=0.5, lw=0.5, label='OT-PF MALA seq')
    ax.plot(params_par_c[name], color='r',    alpha=0.5, lw=0.5, label='OT-PF MALA par')
    ax.axhline(tv, color='k', ls='--', lw=1.5, label='True')
    for b_idx in range(1, B):
        ax.axvline(b_idx * K_BLOCK, color='r', ls=':', lw=0.5, alpha=0.3)
    ax.set_ylabel(lab)
    ax.set_title(f'{lab} trace')
    if i == 0:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[i, 1]
    ax.hist(params_std[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.35, color='gray', label='Standard')
    ax.hist(params_seq_c[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.45, color='b', label='OT-PF MALA seq')
    ax.hist(params_par_c[name][burnin:].ravel(),
            bins=40, density=True, alpha=0.45, color='r', label='OT-PF MALA par')
    ax.axvline(tv, color='k', ls='--', lw=1.5)
    ax.set_title(f'{lab} posterior')
    if i == 0:
        ax.legend(fontsize=7)

axes[-1, 0].set_xlabel('MCMC iteration')
axes[-1, 1].set_xlabel('Parameter value')

plt.suptitle(
    f'OT-PF + Parallel MALA — LGSSM (T={T}, N={N}, K={K_TOTAL}, {B} blocks)\n'
    f'Standard: {t_std:.1f}s | OT-PF seq: {t_seq_total:.1f}s | '
    f'OT-PF par: {t_par_total:.1f}s ({deer_iters_avg:.0f} DEER iters/block, '
    f'{speedup_vs_std:.1f}x vs std, {speedup_vs_seq:.1f}x vs seq)',
    fontsize=11,
)
plt.tight_layout()
plt.savefig('stampede_ot_lgssm.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved plot to stampede_ot_lgssm.png")
