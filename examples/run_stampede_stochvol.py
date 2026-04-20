"""
run_stampede_stochvol.py

STAMPEDE on the stochastic volatility model — Bayesian inference for
(phi, sigma_q, beta) via doubly-parallel particle MCMC.

Model
-----
  x_t = phi * x_{t-1} + sigma_q * eps_t          (log-volatility)
  y_t = beta * exp(x_t / 2) * eta_t              (observations)

Uses a learned amortised twist network trained offline to generalise
across theta values.  Block-refresh strategy (fresh PF noise every B
MCMC steps) for correct targeting.
"""

import time
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

from src.stampede import STAMPEDE

# True parameters 
PHI_TRUE     = 0.95
SIGMA_Q_TRUE = 0.2
BETA_TRUE    = 1.0

T = 1000         # time series length
D = 1            # latent state dimension
N = 256          # particles for PF likelihood
K_TOTAL = 2000   # total MCMC chain length
B = 80           # number of blocks (noise refreshed between blocks) — K_BLOCK=25 sweet spot
K_BLOCK = K_TOTAL // B
D_THETA = 3      # unconstrained parameter dimension


# Parameter transforms
def theta_to_params(theta):
    return {
        'phi':     jnp.tanh(theta[0]),
        'sigma_q': jax.nn.softplus(theta[1]),
        'beta':    jax.nn.softplus(theta[2]),
    }

def params_to_theta(phi, sigma_q, beta):
    return jnp.array([
        jnp.arctanh(phi),
        jnp.log(jnp.exp(sigma_q) - 1.0),
        jnp.log(jnp.exp(beta) - 1.0),
    ])

def log_prior(theta):
    """Weakly informative prior: N(0, 2^2) on unconstrained params."""
    return -0.5 * jnp.sum(theta**2) / 2.0**2


# BIF for SV
def approx_bif_sv(ys, phi, sigma_q, beta, n_iter=5):
    """
    Approximate backward information filter for SV.

    Iteratively linearises the observation log-likelihood
        log p(y|x) = -0.5*log(2pi) - log(beta) - x/2 - y^2/(2*beta^2*exp(x))
    around the current backward message mean.
    """
    T_ = ys.shape[0]
    sq2 = sigma_q**2

    def obs_info(y, x0):
        x0_c = jnp.clip(x0, -20.0, 20.0)
        r = y**2 / (2.0 * beta**2 * jnp.exp(x0_c))
        r = jnp.clip(r, 1e-8, 100.0)
        J_obs = r
        g = -0.5 + r
        h_obs = g + J_obs * x0_c
        return jnp.maximum(J_obs, 1e-6), h_obs

    twist_means = jnp.zeros(T_)

    def bif_iter(twist_means, _):
        def step(carry, inputs):
            J, h = carry
            y_t, lin_pt = inputs
            denom = 1.0 + sq2 * J
            J_pred = phi**2 * J / denom
            h_pred = phi * h / denom
            J_obs, h_obs = obs_info(y_t, lin_pt)
            return (J_pred + J_obs, h_pred + h_obs), (J_pred + J_obs, h_pred + h_obs)

        J_obs_T, h_obs_T = obs_info(ys[-1], twist_means[-1])
        _, (Js_rev, hs_rev) = jax.lax.scan(
            step, (J_obs_T, h_obs_T),
            (ys[-2::-1], twist_means[-2::-1]),
        )
        Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_obs_T])])
        hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_obs_T])])
        return hs / Js, None

    twist_means, _ = jax.lax.scan(bif_iter, twist_means, None, length=n_iter)
    # Run one more time to get Js for variances
    def final_step(carry, inputs):
        J, h = carry
        y_t, lin_pt = inputs
        denom = 1.0 + sq2 * J
        J_pred = phi**2 * J / denom
        h_pred = phi * h / denom
        J_obs, h_obs = obs_info(y_t, lin_pt)
        return (J_pred + J_obs, h_pred + h_obs), (J_pred + J_obs, h_pred + h_obs)

    J_obs_T, h_obs_T = obs_info(ys[-1], twist_means[-1])
    _, (Js_rev, hs_rev) = jax.lax.scan(
        final_step, (J_obs_T, h_obs_T),
        (ys[-2::-1], twist_means[-2::-1]),
    )
    Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_obs_T])])
    hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_obs_T])])
    return hs / Js, 1.0 / Js


def compute_twists_sv(model_params, observations):
    """Adaptive twist computation: BIF at the proposed theta."""
    ys = observations[:, 0]
    twist_means, twist_vars = approx_bif_sv(
        ys, model_params['phi'], model_params['sigma_q'], model_params['beta'])
    return {
        'mean':      twist_means[:, None],
        'log_scale': 0.5 * jnp.log(twist_vars)[:, None],
    }


# Amortised twist network
# Architecture: backward GRU over observations, conditioned on theta.
#   1. Encode theta via a small MLP: theta (3,) -> h_theta (H,)
#   2. Backward GRU scan over y_{T:1} with h_theta concatenated at each step
#   3. Readout MLP: (gru_hidden_t, h_theta) -> (mu_t, log_scale_t)
#
# Trained offline: sample theta ~ prior, compute BIF targets, MSE loss.

H_GRU = 32     # GRU hidden size
H_MLP = 32     # MLP hidden size


def init_amortised_twist_params(key, D_theta=3):
    """Initialise all parameters for the amortised twist network."""
    keys = jr.split(key, 8)

    def _init_linear(key, d_in, d_out, scale=0.01):
        return {
            'W': jr.normal(key, (d_in, d_out)) * scale,
            'b': jnp.zeros(d_out),
        }

    params = {}
    # Theta encoder: D_theta -> H_MLP -> H_GRU (used as conditioning)
    params['theta_enc1'] = _init_linear(keys[0], D_theta, H_MLP)
    params['theta_enc2'] = _init_linear(keys[1], H_MLP, H_GRU)

    # GRU: input is (1 + H_GRU) = obs + theta_emb, hidden is H_GRU
    d_gru_in = 1 + H_GRU
    # GRU gates: reset, update, candidate
    params['gru_Wz'] = _init_linear(keys[2], d_gru_in + H_GRU, H_GRU, scale=0.02)
    params['gru_Wr'] = _init_linear(keys[3], d_gru_in + H_GRU, H_GRU, scale=0.02)
    params['gru_Wh'] = _init_linear(keys[4], d_gru_in + H_GRU, H_GRU, scale=0.02)

    # Readout: (H_GRU + H_GRU) -> H_MLP -> 2 (mu, log_scale)
    params['readout1'] = _init_linear(keys[5], H_GRU + H_GRU, H_MLP)
    params['readout2'] = _init_linear(keys[6], H_MLP, 2)

    # Initialise readout bias: log_scale ~ 0 means scale ~ 1
    params['readout2']['b'] = jnp.array([0.0, 0.0])

    return params


def apply_amortised_twist(net_params, theta, observations):
    """
    Forward pass of the amortised twist network.

    Parameters
    ----------
    net_params : dict of network parameters
    theta      : (D_theta,) unconstrained parameter vector
    observations : (T, 1) observations

    Returns
    -------
    twist_params : dict with 'mean' (T, 1) and 'log_scale' (T, 1)
    """
    ys = observations[:, 0]  # (T,)
    T_ = ys.shape[0]

    def linear(p, x):
        return x @ p['W'] + p['b']

    # Encode theta
    h_theta = jax.nn.relu(linear(net_params['theta_enc1'], theta))
    h_theta = jax.nn.tanh(linear(net_params['theta_enc2'], h_theta))  # (H_GRU,)

    # Backward GRU scan over observations
    def gru_step(h, y_t):
        # Input: concatenate observation with theta embedding
        inp = jnp.concatenate([jnp.array([y_t]), h_theta])  # (1 + H_GRU,)
        combined = jnp.concatenate([inp, h])  # (1 + H_GRU + H_GRU,)

        z = jax.nn.sigmoid(linear(net_params['gru_Wz'], combined))  # update gate
        r = jax.nn.sigmoid(linear(net_params['gru_Wr'], combined))  # reset gate

        combined_r = jnp.concatenate([inp, r * h])
        h_tilde = jnp.tanh(linear(net_params['gru_Wh'], combined_r))  # candidate

        h_new = (1.0 - z) * h + z * h_tilde
        return h_new, h_new

    h0 = jnp.zeros(H_GRU)
    # Scan backward: reverse observations, then reverse output
    _, gru_hiddens_rev = jax.lax.scan(gru_step, h0, ys[::-1])
    gru_hiddens = gru_hiddens_rev[::-1]  # (T, H_GRU)

    # Readout: (gru_hidden_t, h_theta) -> (mu_t, log_scale_t)
    h_theta_broad = jnp.broadcast_to(h_theta, (T_, H_GRU))
    readout_in = jnp.concatenate([gru_hiddens, h_theta_broad], axis=1)  # (T, 2*H_GRU)
    hidden = jax.nn.relu(readout_in @ net_params['readout1']['W']
                         + net_params['readout1']['b'])
    out = hidden @ net_params['readout2']['W'] + net_params['readout2']['b']  # (T, 2)

    return {
        'mean':      out[:, 0:1],      # (T, 1)
        'log_scale': out[:, 1:2],      # (T, 1)
    }


def train_amortised_twist(key, ys_2d, n_samples=512, n_steps=3000, lr=3e-4,
                          n_particles_train=64):
    """
    Train the amortised twist network offline via ELBO maximisation.

    Phase 1: Warm-start by matching BIF targets (MSE loss).
    Phase 2: Fine-tune by maximising the log marginal likelihood estimate
             (ELBO) of the twisted PF — this directly optimises twist quality
             rather than matching the approximate BIF.

    Parameters
    ----------
    key               : PRNG key
    ys_2d             : (T, 1) observed data (fixed)
    n_samples         : number of training theta samples for phase 1
    n_steps           : total optimisation steps (split between phases)
    lr                : learning rate
    n_particles_train : particles per PF for ELBO phase

    Returns
    -------
    net_params : trained network parameters
    """
    key, init_key, sample_key = jr.split(key, 3)
    net_params = init_amortised_twist_params(init_key)

    T_ = ys_2d.shape[0]
    ys_flat = ys_2d[:, 0]

    #  Phase 1: BIF warm-start 
    n_phase1 = n_steps // 3
    n_phase2 = n_steps - n_phase1

    print("  Phase 1: BIF warm-start...")
    print("  Generating training data (BIF targets for many theta samples)...")
    theta_samples = 2.0 * jr.normal(sample_key, (n_samples, D_THETA))

    @jax.jit
    def compute_bif_targets(theta):
        p = theta_to_params(theta)
        means, variances = approx_bif_sv(ys_flat, p['phi'], p['sigma_q'], p['beta'])
        log_scales = 0.5 * jnp.log(variances)
        return means, log_scales

    target_means, target_log_scales = jax.vmap(compute_bif_targets)(theta_samples)
    print(f"  Training data ready: {n_samples} theta samples, T={T_}")

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(net_params)

    @jax.jit
    def mse_loss_fn(net_params, theta_batch, target_mean_batch, target_ls_batch):
        def single_loss(theta, tgt_m, tgt_ls):
            pred = apply_amortised_twist(net_params, theta, ys_2d)
            loss_m = jnp.mean((pred['mean'][:, 0] - tgt_m) ** 2)
            loss_ls = jnp.mean((pred['log_scale'][:, 0] - tgt_ls) ** 2)
            return loss_m + loss_ls
        return jnp.mean(jax.vmap(single_loss)(theta_batch, target_mean_batch, target_ls_batch))

    @jax.jit
    def mse_train_step(net_params, opt_state, theta_batch, tgt_m_batch, tgt_ls_batch):
        loss, grads = jax.value_and_grad(mse_loss_fn)(
            net_params, theta_batch, tgt_m_batch, tgt_ls_batch)
        updates, opt_state_new = optimizer.update(grads, opt_state, net_params)
        net_params_new = optax.apply_updates(net_params, updates)
        return net_params_new, opt_state_new, loss

    batch_size = min(64, n_samples)
    for step in range(n_phase1):
        key, batch_key = jr.split(key)
        idx = jr.choice(batch_key, n_samples, (batch_size,), replace=False)
        net_params, opt_state, loss = mse_train_step(
            net_params, opt_state, theta_samples[idx],
            target_means[idx], target_log_scales[idx])
        if step % 300 == 0 or step == n_phase1 - 1:
            print(f"    Step {step:>5d}/{n_phase1}: MSE loss = {float(loss):.6f}")

    #  Phase 2: ELBO fine-tuning 
    print(f"  Phase 2: ELBO fine-tuning ({n_phase2} steps, "
          f"{n_particles_train} particles)...")

    # Narrower theta distribution around plausible values for ELBO phase
    key, narrow_key = jr.split(key)
    theta_narrow = 1.0 * jr.normal(narrow_key, (n_samples, D_THETA))

    optimizer2 = optax.adam(lr * 0.3)
    opt_state2 = optimizer2.init(net_params)

    def elbo_single(net_params, theta, pf_key):
        """Run twisted PF and return negative log ML estimate."""
        model_params = theta_to_params(theta)
        twist = apply_amortised_twist(net_params, theta, ys_2d)

        # Run sequential twisted PF
        noise_key, x0_key = jr.split(pf_key)
        noise = jr.normal(noise_key, (n_particles_train, T_, D))
        x0s = jr.normal(x0_key, (n_particles_train, D)) * 0.5

        def run_one_particle(x0, noise_i):
            def scan_step(x_prev, driver):
                x_next = twisted_step_fn(x_prev, driver, model_params)
                return x_next, x_next
            _, states = jax.lax.scan(
                scan_step, x0, (ys_2d, noise_i, twist))
            return states

        all_states = jax.vmap(run_one_particle)(x0s, noise)  # (N, T, D)

        # Compute log weights
        def weights_one(x0, trajectory):
            x_prevs = jnp.concatenate([x0[None, :], trajectory[:-1]], axis=0)
            def w_step(x_prev, x_next, obs_t, twist_t):
                return pf_log_weight_fn(x_prev, x_next, obs_t, twist_t, model_params)
            return jax.vmap(w_step)(x_prevs, trajectory, ys_2d, twist)

        log_ws = jax.vmap(weights_one)(x0s, all_states)  # (N, T)
        total = jnp.sum(log_ws, axis=1)  # (N,)
        log_ml = jax.scipy.special.logsumexp(total) - jnp.log(float(n_particles_train))
        return -log_ml  # minimise negative ELBO

    @jax.jit
    def elbo_train_step(net_params, opt_state, theta_batch, pf_keys):
        def batch_loss(net_params):
            losses = jax.vmap(
                lambda th, k: elbo_single(net_params, th, k)
            )(theta_batch, pf_keys)
            return jnp.mean(losses)
        loss, grads = jax.value_and_grad(batch_loss)(net_params)
        # Clip gradients for stability
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, opt_state_new = optimizer2.update(grads, opt_state, net_params)
        net_params_new = optax.apply_updates(net_params, updates)
        return net_params_new, opt_state_new, loss

    elbo_batch = min(8, n_samples)  # smaller batch for ELBO (expensive)
    for step in range(n_phase2):
        key, batch_key, pf_key = jr.split(key, 3)
        idx = jr.choice(batch_key, n_samples, (elbo_batch,), replace=False)
        pf_keys = jr.split(pf_key, elbo_batch)
        net_params, opt_state2, loss = elbo_train_step(
            net_params, opt_state2, theta_narrow[idx], pf_keys)
        if step % 500 == 0 or step == n_phase2 - 1:
            print(f"    Step {step:>5d}/{n_phase2}: neg ELBO = {float(loss):.2f}")

    return net_params


# 
# Twisted PF step + weight functions (D=1 diagonal twist)
# 
def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi, sq = params['phi'], params['sigma_q']
    mu  = twist['mean']; tau = jnp.exp(twist['log_scale'])
    pq  = 1.0 / sq**2; pt = 1.0 / tau**2
    vp  = 1.0 / (pq + pt)
    mp  = vp * (pq * phi * x_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise


def pf_log_weight_fn(x_prev, x_next, obs, twist, params):
    phi, sq = params['phi'], params['sigma_q']
    beta    = params['beta']
    mu  = twist['mean']; tau = jnp.exp(twist['log_scale'])

    # log p(y_t | x_t) — SV likelihood
    log_lik = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(beta)
               - x_next[0] / 2.0
               - obs[0]**2 / (2.0 * beta**2 * jnp.exp(x_next[0])))

    # log p(x_t | x_{t-1})
    log_trans = (-0.5 * (x_next[0] - phi * x_prev[0])**2 / sq**2
                 - 0.5 * jnp.log(2.0 * jnp.pi * sq**2))

    # log q(x_t | x_{t-1})
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    log_prop = (-0.5 * (x_next[0] - mp[0])**2 / vp[0]
                - 0.5 * jnp.log(2.0 * jnp.pi * vp[0]))
    return log_lik + log_trans - log_prop


# 
# Bootstrap PF for standard PMCMC
# 
def bootstrap_pf_log_ml_sv(ys, phi, sigma_q, beta, key, N):
    """Bootstrap PF for SV: returns log marginal likelihood estimate."""
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
        log_w = jax.vmap(lambda x: (
            -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(beta)
            - x / 2.0 - yt**2 / (2.0 * beta**2 * jnp.exp(x))
        ))(particles)
        log_ml_inc = jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))
        particles = _systematic_resample(resamp_key, log_w, particles)
        return (particles, key, log_ml_acc + log_ml_inc), None

    key, init_key = jr.split(key)
    particles = jr.normal(init_key, (N,)) * 0.5
    (_, _, log_ml), _ = jax.lax.scan(step, (particles, key, 0.0), ys)
    return log_ml


def standard_pmcmc_sv(key, theta0, ys, N, K, proposal_scale):
    """Standard PMCMC: bootstrap PF + random-walk MH for SV."""
    def step(carry, key):
        theta, log_post = carry
        prop_key, pf_key, accept_key = jr.split(key, 3)
        theta_star = theta + proposal_scale * jr.normal(prop_key, theta.shape)
        p_star = theta_to_params(theta_star)
        log_ml_star = bootstrap_pf_log_ml_sv(
            ys, p_star['phi'], p_star['sigma_q'], p_star['beta'], pf_key, N)
        log_post_star = log_ml_star + log_prior(theta_star)
        log_alpha = log_post_star - log_post
        accept = jnp.log(jr.uniform(accept_key)) < log_alpha
        theta_new = jnp.where(accept, theta_star, theta)
        log_post_new = jnp.where(accept, log_post_star, log_post)
        return (theta_new, log_post_new), theta_new

    key, pf_key = jr.split(key)
    p0 = theta_to_params(theta0)
    log_post0 = bootstrap_pf_log_ml_sv(
        ys, p0['phi'], p0['sigma_q'], p0['beta'], pf_key, N) + log_prior(theta0)
    keys = jr.split(key, K)
    _, chain = jax.lax.scan(step, (theta0, log_post0), keys)
    return chain



# Simulate data

key = jr.PRNGKey(42)
key, k1, k2, k3 = jr.split(key, 4)
x0_true = jr.normal(k1) * 0.5

def sim_sv_step(x, noise_obs):
    q, r = noise_obs
    xn = PHI_TRUE * x + SIGMA_Q_TRUE * q
    y  = BETA_TRUE * jnp.exp(xn / 2.0) * r
    return xn, (xn, y)

_, (true_xs, ys) = jax.lax.scan(
    sim_sv_step, x0_true, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
)
ys_2d = ys[:, None]

print(f"True params:    phi={PHI_TRUE}, sigma_q={SIGMA_Q_TRUE}, beta={BETA_TRUE}")
key, init_key = jr.split(key)
theta0 = params_to_theta(PHI_TRUE, SIGMA_Q_TRUE, BETA_TRUE)
theta0 = theta0 + 0.3 * jr.normal(init_key, theta0.shape)

p0 = theta_to_params(theta0)
print(f"Initial params: phi={p0['phi']:.3f}, sigma_q={p0['sigma_q']:.3f}, "
      f"beta={p0['beta']:.3f}")
print(f"\nT={T}, N={N} particles, K={K_TOTAL} MCMC steps ({B} blocks x {K_BLOCK})")
print("=" * 70)


# Tran amortised twist network
print("\n--- [0] Training amortised twist network ---")
key, twist_train_key = jr.split(key)
t0_train = time.perf_counter()
trained_twist_params = train_amortised_twist(
    twist_train_key, ys_2d, n_samples=512, n_steps=2000, lr=3e-4)
t_train = time.perf_counter() - t0_train
print(f"  Training time: {t_train:.1f}s")

# Verify on true theta
twist_at_true = apply_amortised_twist(
    trained_twist_params, params_to_theta(PHI_TRUE, SIGMA_Q_TRUE, BETA_TRUE), ys_2d)
bif_means, bif_vars = approx_bif_sv(ys, PHI_TRUE, SIGMA_Q_TRUE, BETA_TRUE)
bif_ls = 0.5 * jnp.log(bif_vars)
rmse_mean = float(jnp.sqrt(jnp.mean(
    (twist_at_true['mean'][:, 0] - bif_means) ** 2)))
rmse_ls = float(jnp.sqrt(jnp.mean(
    (twist_at_true['log_scale'][:, 0] - bif_ls) ** 2)))
print(f"  Verification at true theta: RMSE(mean)={rmse_mean:.4f}, "
      f"RMSE(log_scale)={rmse_ls:.4f}")


def compute_twists_amortised(model_params, observations):
    """Amortised twist: use trained network instead of BIF."""
    # Recover theta from model_params (inverse of theta_to_params)
    theta = jnp.array([
        jnp.arctanh(model_params['phi']),
        jnp.log(jnp.exp(model_params['sigma_q']) - 1.0),
        jnp.log(jnp.exp(model_params['beta']) - 1.0),
    ])
    return apply_amortised_twist(trained_twist_params, theta, observations)



# Method 1: Standard PMCMC
print("\n--- [1] Standard PMCMC (bootstrap PF + RW-MH) ---")
key, std_key = jr.split(key)

run_std = jax.jit(standard_pmcmc_sv, static_argnames=("N", "K"))
chain_std = run_std(std_key, theta0, ys, N, K_TOTAL, proposal_scale=0.02)
chain_std.block_until_ready()

t0 = time.perf_counter()
chain_std = run_std(std_key, theta0, ys, N, K_TOTAL, proposal_scale=0.02)
chain_std.block_until_ready()
t_std = time.perf_counter() - t0

accept_std = 1.0 - jnp.mean(chain_std[1:, 0] == chain_std[:-1, 0])
print(f"  Time: {t_std:.3f}s | Accept: {float(accept_std):.3f}")


# Methods 2 & 3: Block-refresh twisted PMCMC and STAMPEDE
twist_params_dummy = {'mean': jnp.zeros((T, D)), 'log_scale': jnp.zeros((T, D))}

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
    compute_twists_fn=compute_twists_amortised,
)

run_seq_block = jax.jit(stampede.run_sequential)
run_par_block = jax.jit(stampede.run_parallel)

#  Method 2: Sequential twisted PMCMC with block-refresh 
print(f"\n--- [2] Twisted PMCMC (MALA, {B} blocks x {K_BLOCK}, noise refresh) ---")

# Warmup
key, wkey, wnkey, wx0key = jr.split(key, 4)
pf_noise_w = jr.normal(wnkey, (N, T, D))
x0s_w = jr.normal(wx0key, (N, D)) * 0.5
params_w = {'step_size': 5e-4, 'target_params': {
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
    x0s = jr.normal(x0_key, (N, D)) * 0.5
    target_params = {
        'observations': ys_2d,
        'pf_noise':     pf_noise,
        'twist_params': twist_params_dummy,
        'x0s':          x0s,
    }
    params_block = {'step_size': 5e-4, 'target_params': target_params}

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
x0s_w2 = jr.normal(wx0key2, (N, D)) * 0.5
params_w2 = {'step_size': 5e-4, 'key': wdkey, 'target_params': {
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
    x0s = jr.normal(x0_key, (N, D)) * 0.5
    target_params = {
        'observations': ys_2d,
        'pf_noise':     pf_noise,
        'twist_params': twist_params_dummy,
        'x0s':          x0s,
    }
    params_block = {'step_size': 5e-4, 'key': deer_key,
                    'target_params': target_params}

    t0 = time.perf_counter()
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
for name, tv in [('phi', PHI_TRUE), ('sigma_q', SIGMA_Q_TRUE), ('beta', BETA_TRUE)]:
    s0 = params_std[name][burnin:]
    s1 = params_seq_c[name][burnin:]
    s2 = params_par_c[name][burnin:]
    print(f"{name:>12} {tv:>8.3f} {float(s0.mean()):>8.3f} {float(s1.mean()):>8.3f}"
          f" {float(s2.mean()):>8.3f} {float(s0.std()):>8.4f}"
          f" {float(s1.std()):>8.4f} {float(s2.std()):>8.4f}")

# Bias diagnostic
print(f"\n{'':>12} {'Std bias':>12} {'TwSeq bias':>12} {'STP bias':>12}")
print("-" * 52)
for name, tv in [('phi', PHI_TRUE), ('sigma_q', SIGMA_Q_TRUE), ('beta', BETA_TRUE)]:
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

names  = ['phi', 'sigma_q', 'beta']
trues  = [PHI_TRUE, SIGMA_Q_TRUE, BETA_TRUE]
labels = [r'$\phi$', r'$\sigma_q$', r'$\beta$']

for i, (name, tv, lab) in enumerate(zip(names, trues, labels)):
    ax = axes[i, 0]
    ax.plot(params_std[name],    color='gray', alpha=0.4, lw=0.4, label='Standard')
    ax.plot(params_seq_c[name],  color='b',    alpha=0.5, lw=0.5, label='Twisted seq')
    ax.plot(params_par_c[name],  color='r',    alpha=0.5, lw=0.5, label='STAMPEDE')
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
    f'STAMPEDE — Stochastic Volatility (T={T}, N={N}, K={K_TOTAL}, {B} blocks)\n'
    f'Standard: {t_std:.1f}s | Twisted seq: {t_seq_total:.1f}s | '
    f'STAMPEDE: {t_par_total:.1f}s ({deer_iters_avg:.0f} DEER iters/block, '
    f'{speedup_vs_std:.1f}x vs std, {speedup_vs_seq:.1f}x vs seq)',
    fontsize=11,
)
plt.tight_layout()
plt.savefig('stampede_stochvol.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved plot to stampede_stochvol.png")
