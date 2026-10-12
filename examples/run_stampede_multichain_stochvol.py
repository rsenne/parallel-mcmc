"""
run_stampede_multichain_stochvol.py

Multi-chain STAMPEDE on the stochastic-volatility model -- a real
nonlinear SSM where the BIF twist is approximate (iterated-linearised
backward info filter) and per-step log p(y|theta) is more expensive
than the LGSSM case.

Model
-----
  x_t = phi * x_{t-1} + sigma_q * eps_t            (log-volatility, latent)
  y_t = beta * exp(x_t / 2) * eta_t                (observation)

We sample only phi for now; sigma_q and beta are fixed.

Architecture (same axis we showed works on LGSSM):
  M independent MALA chains, each running SEQUENTIAL MALA whose
  log_prob is the *parallel* windowed Gumbel-STE PF.
  Chains are vmap'd onto one GPU.

Outputs:
  - per-chain final phi
  - cross-chain mean, std, range
  - R-hat (Gelman-Rubin)  [reuse second half of each chain as "burn-in"]
  - ESS per chain and total ESS
  - wall-clock vs single chain
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src import samplers
from src.gumbel_particle_filter import TwistedGumbelParticleFilter


# Model truth
PHI_TRUE   = 0.95
SIGMA_Q    = 0.2
BETA       = 1.0
T          = 1024

# PF (inner)
N   = 64
K_W = 64
TAU = 1.0
PF_MAX_ITER = 100

# MALA outer
D_PARAM    = 1
K_MALA     = 1024              # longer chain for better posterior
STEP_SIZE  = 1e-2

# Multi-chain
M_CHAINS_SWEEP = [1, 2, 4, 8, 16, 32]
BURN_IN_FRAC   = 0.5           # fraction of each chain discarded for diagnostics


# ---- Twisted PF step + log_weight for SV ----

def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi     = params["phi"]
    sigma_q = params["sigma_q"]
    mu_twist  = twist["mean"]
    tau_twist = jnp.exp(twist["log_scale"])
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    return mean_post + jnp.sqrt(var_post) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    phi     = params["phi"]
    sigma_q = params["sigma_q"]
    beta    = params["beta"]
    mu_twist  = twist["mean"]
    tau_twist = jnp.exp(twist["log_scale"])

    # log p(y_t | x_t)  --  SV likelihood
    log_lik = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(beta)
               - x_next[0] / 2.0
               - obs[0]**2 / (2.0 * beta**2 * jnp.exp(x_next[0])))

    # log p(x_t | x_{t-1})
    log_trans = (-0.5 * (x_next[0] - phi * x_prev[0])**2 / sigma_q**2
                 - 0.5 * jnp.log(2.0 * jnp.pi * sigma_q**2))

    # log q_twist(x_t | x_{t-1})
    prec_q     = 1.0 / sigma_q**2
    prec_twist = 1.0 / tau_twist**2
    var_post   = 1.0 / (prec_q + prec_twist)
    mean_post  = var_post * (prec_q * phi * x_prev + prec_twist * mu_twist)
    log_prop = (-0.5 * (x_next[0] - mean_post[0])**2 / var_post[0]
                - 0.5 * jnp.log(2.0 * jnp.pi * var_post[0]))
    return log_lik + log_trans - log_prop


def approx_backward_info_filter_sv(ys, phi, sigma_q, beta, n_iter=10):
    """Iterated-linearised Gaussian BIF for SV. Anchored at phi (we use
    phi=0 below -- non-cheating). Serial backward scans for reference."""
    T = ys.shape[0]
    sq2 = sigma_q**2

    def obs_info(y, x0):
        x0_c = jnp.clip(x0, -20.0, 20.0)
        r = y**2 / (2.0 * beta**2 * jnp.exp(x0_c))
        r = jnp.clip(r, 1e-8, 100.0)
        J_obs = r
        g = -0.5 + r
        h_obs = g + J_obs * x0_c
        return J_obs, h_obs

    twist_means = jnp.zeros(T)
    for _ in range(n_iter):
        def step(carry, inputs):
            J, h = carry
            y_t, lin_pt = inputs
            denom = 1.0 + sq2 * J
            J_pred = phi**2 * J / denom
            h_pred = phi * h / denom
            J_obs, h_obs = obs_info(y_t, lin_pt)
            J_obs = jnp.maximum(J_obs, 1e-6)
            return (J_pred + J_obs, h_pred + h_obs), (J_pred + J_obs, h_pred + h_obs)

        J_obs_T, h_obs_T = obs_info(ys[-1], twist_means[-1])
        J_obs_T = jnp.maximum(J_obs_T, 1e-6)
        _, (Js_rev, hs_rev) = jax.lax.scan(
            step, (J_obs_T, h_obs_T), (ys[-2::-1], twist_means[-2::-1])
        )
        Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_obs_T])])
        hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_obs_T])])
        twist_means = hs / Js
    return twist_means, 1.0 / Js


def parallel_backward_info_filter_sv(ys, phi, sigma_q, beta, n_iter=10):
    """O(log T) parallel BIF for SV.

    Each outer linearization iter uses TWO associative scans:
      (i)  Riccati on J as a Mobius transform with time-varying matrices
              M_t = [[phi^2 + sq*J_obs_t, J_obs_t],
                     [sq,                  1   ]].
           Composes via 2x2 matmul -> associative_scan(matmul, Ms).
      (ii) Affine recurrence on h given J:
              h_t = phi/(1 + sq*J_{t+1}) * h_{t+1} + h_obs_t.
           Composes via (a, b) o (a', b') = (a'a, a'b + b')
           -> associative_scan over (alpha, beta) pairs.

    Mathematically equivalent to the serial backward scan but with
    parallel depth O(log T) per outer iter.
    """
    T = ys.shape[0]
    sq = sigma_q**2

    def obs_info(y, x0):
        x0_c = jnp.clip(x0, -20.0, 20.0)
        r = y**2 / (2.0 * beta**2 * jnp.exp(x0_c))
        r = jnp.clip(r, 1e-8, 100.0)
        return r, (-0.5 + r) + r * x0_c

    twist_means = jnp.zeros(T)
    for _ in range(n_iter):
        # Per-t observation info, evaluated at the current linearization points.
        J_obs_all, h_obs_all = jax.vmap(obs_info)(ys, twist_means)  # (T,), (T,)
        J_obs_all = jnp.maximum(J_obs_all, 1e-6)

        # Boundary: J_T = J_obs[T-1], h_T = h_obs[T-1].
        J_T_init = J_obs_all[-1]
        h_T_init = h_obs_all[-1]

        # The backward chain advances from t = T-1 down to t = 0, i.e. T-1
        # steps. The k-th step (k = 0..T-2) takes (J_{T-1-k+1}, h_{T-1-k+1})
        # to (J_{T-1-k}, h_{T-1-k}) using obs info at index T-2-k.
        J_obs_back = J_obs_all[-2::-1]    # (T-1,)  J_obs at t = T-2, T-3, ..., 0
        h_obs_back = h_obs_all[-2::-1]    # (T-1,)  h_obs at same t

        # ---- (i) parallel Mobius scan for J ----
        # M_k = [[phi^2 + sq*J_obs_back[k], J_obs_back[k]], [sq, 1]]
        a = phi**2 + sq * J_obs_back
        b = J_obs_back
        c = jnp.full_like(a, sq)
        d = jnp.ones_like(a)
        Ms = jnp.stack([
            jnp.stack([a, b], axis=-1),
            jnp.stack([c, d], axis=-1),
        ], axis=-2)                       # (T-1, 2, 2)

        # J_{T-2-k} = M_{T-2-k}(M_{T-2-k+1}(... M_{T-2}(J_T))), and Mobius
        # composition f_B(f_A(x)) has matrix B @ A, so the k-th cumulative
        # transform is Ms[k] @ Ms[k-1] @ ... @ Ms[0] — a REVERSED product.
        # op(A, B) = B @ A is associative, so the scan builds exactly that.
        # Mobius transforms are invariant to scalar rescaling of the matrix,
        # so normalize each partial product: the raw cumulative product's
        # entries grow like lambda_max^T and overflow float32 for T ~ 1e3.
        def mobius_matmul(A, B):
            C = B @ A
            scale = jnp.max(jnp.abs(C), axis=(-2, -1), keepdims=True)
            return C / jnp.maximum(scale, 1e-30)
        cum_Ms = jax.lax.associative_scan(mobius_matmul, Ms)

        def apply_mobius(M, J):
            return (M[0, 0] * J + M[0, 1]) / (M[1, 0] * J + M[1, 1])
        Js_back = jax.vmap(apply_mobius, in_axes=(0, None))(cum_Ms, J_T_init)
        # Js_back[k] = J at t = T-2-k.   Reverse + append J_T -> Js[0..T-1].
        Js = jnp.concatenate([Js_back[::-1], jnp.array([J_T_init])])

        # ---- (ii) parallel affine scan for h given J ----
        # h_{t-1} = (phi / (1 + sq * J_t)) * h_t + h_obs_{t-1}.
        # In backward step index k (k=0..T-2), J at source is J_{T-1-k+1}=J_{T-k}.
        # So the source-J sequence is [J_T_init, J_{T-1}, ..., J_2].
        J_sources = jnp.concatenate([jnp.array([J_T_init]), Js_back])[:T-1]
        alphas = phi / (1.0 + sq * J_sources)
        betas  = h_obs_back

        def affine_combine(left, right):
            a_l, b_l = left
            a_r, b_r = right
            return (a_r * a_l, a_r * b_l + b_r)
        cum_alphas, cum_betas = jax.lax.associative_scan(
            affine_combine, (alphas, betas)
        )
        hs_back = cum_alphas * h_T_init + cum_betas
        hs = jnp.concatenate([hs_back[::-1], jnp.array([h_T_init])])

        twist_means = hs / Js

    return twist_means, 1.0 / Js


# ---- MCMC diagnostics ----

def r_hat_1d(chains):
    """Gelman-Rubin R-hat for (M, N) chain matrix (M chains of N samples each)."""
    M, N = chains.shape
    chain_means = jnp.mean(chains, axis=1)         # (M,)
    chain_vars  = jnp.var(chains, axis=1, ddof=1)  # (M,)
    W = jnp.mean(chain_vars)
    B = N * jnp.var(chain_means, ddof=1) if M > 1 else jnp.array(0.0)
    var_hat = (1.0 - 1.0 / N) * W + (1.0 / N) * B
    return jnp.sqrt(var_hat / (W + 1e-12))


def autocorr_1d(x):
    """Autocorrelation up to lag N//4 for a 1D chain, via FFT.
    Returns rho_0=1, rho_1, ..., rho_{K-1} as a fixed-shape array."""
    N = x.shape[0]
    x = x - jnp.mean(x)
    # Zero-pad to >= 2N for linear (not circular) autocorrelation.
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    x_padded = jnp.concatenate([x, jnp.zeros(fft_size - N)])
    F = jnp.fft.fft(x_padded)
    ac_full = jnp.fft.ifft(F * jnp.conj(F)).real
    ac = ac_full[:N // 4] / (ac_full[0] + 1e-12)
    return ac


def ess_1d(chains):
    """Effective sample size for (M, N) chain matrix.
    Per-chain Sokal initial-positive-sequence ESS, summed across chains."""
    def ess_one(x):
        ac = autocorr_1d(x)                       # (K,) with rho_0 = 1
        # Sokal initial-positive: cumulative mask 1 up to first negative.
        neg = ac < 0.0
        # cummax-style: 1 until any negative seen
        cum_any_neg = jnp.cumsum(neg.astype(jnp.int32))
        valid = (cum_any_neg == 0).astype(ac.dtype)
        # tau = 1 + 2 * sum_{k>=1} valid_k * rho_k
        tau = 1.0 + 2.0 * jnp.sum(valid[1:] * ac[1:])
        return x.shape[0] / jnp.maximum(tau, 1.0)
    ess_per_chain = jax.vmap(ess_one)(chains)
    return ess_per_chain, jnp.sum(ess_per_chain)


# ---- timing helper ----

def time_min(fn, *args, repeat=2):
    out = fn(*args); jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    return min(times), out


def main():
    print(f"Device: {jax.devices()}")
    print(f"STAMPEDE multi-chain on SV   T={T}  N={N}  K_W={K_W}  K_MALA={K_MALA}  "
          f"step={STEP_SIZE}")
    print(f"PHI_TRUE={PHI_TRUE}  SIGMA_Q={SIGMA_Q}  BETA={BETA}")

    key = jr.PRNGKey(0)
    key, k1, k2, k3, k4, k5 = jr.split(key, 6)

    # --- correctness check: parallel BIF vs serial BIF ---
    # (one-shot, on data that we'll generate just below)
    pass  # (the check is interleaved after data generation below)

    # simulate SV data
    x0_true = jr.normal(k1) * 0.5
    def sim(x, qr):
        q, r = qr
        xn = PHI_TRUE * x + SIGMA_Q * q
        return xn, (xn, BETA * jnp.exp(xn / 2.0) * r)
    _, (true_xs, ys) = jax.lax.scan(
        sim, x0_true, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
    )

    # Check parallel BIF == serial BIF (sanity).
    par_m, par_v = parallel_backward_info_filter_sv(ys, 0.5, SIGMA_Q, BETA, n_iter=10)
    ser_m, ser_v = approx_backward_info_filter_sv(ys, 0.5, SIGMA_Q, BETA, n_iter=10)
    print(f"  parallel vs serial BIF @ phi=0.5:  "
          f"max |mean diff| = {float(jnp.max(jnp.abs(par_m - ser_m))):.3e}  "
          f"max |var diff|  = {float(jnp.max(jnp.abs(par_v - ser_v))):.3e}")

    # Build a PLACEHOLDER tape -- only noise + gumbel + obs are reused.
    # The twist slot is overwritten inside log_prob with the iterated-BIF
    # twist computed at the *proposed* phi each step (adaptive twist).
    placeholder_twist = {
        "mean":      jnp.zeros((T, 1)),
        "log_scale": jnp.zeros((T, 1)),
    }
    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=1, tau=TAU)
    base_tape = tgpf.make_tape(ys[:, None], placeholder_twist, k4, N=N, K=K_W)
    x0s_pf = jr.normal(k5, (N, 1)) * 0.5

    target_params = {
        "ys":        ys,
        "base_tape": base_tape,
        "x0s":       x0s_pf,
        "deer_key":  jr.PRNGKey(1),
    }

    BIF_N_ITER = 10

    def log_prob(theta, tp):
        phi = jnp.tanh(theta[0])
        # Iterated-linearised BIF at the PROPOSED phi -> high-quality
        # twist for this MALA step. We stop_gradient on the twist outputs:
        # the PF log_Z estimator's *expectation* is invariant to the
        # proposal/twist (it only affects variance), so propagating
        # d/dphi through the BIF Riccati is unnecessary and numerically
        # unstable near |phi|->1. Forward value of phi still flows into
        # the twist, giving the adaptive low-variance proposal we want.
        bif_means_raw, bif_vars_raw = parallel_backward_info_filter_sv(
            tp["ys"], phi, SIGMA_Q, BETA, n_iter=BIF_N_ITER
        )
        bif_means = jax.lax.stop_gradient(bif_means_raw)
        bif_vars  = jax.lax.stop_gradient(bif_vars_raw)
        twist_params_t = {
            "mean":      bif_means[:, None],
            "log_scale": 0.5 * jnp.log(bif_vars)[:, None],
        }
        obs_w, noise_w, _, gumbel_w = tp["base_tape"]
        num_windows = obs_w.shape[0]
        twist_w = jax.tree_util.tree_map(
            lambda x: x.reshape((num_windows, K_W) + x.shape[1:]),
            twist_params_t,
        )
        tape = (obs_w, noise_w, twist_w, gumbel_w)

        model_params = {
            "phi": phi, "sigma_q": SIGMA_Q, "beta": BETA,
            "tau": TAU, "key": tp["deer_key"],
        }
        _, _, log_Z, _ = tgpf.run_parallel(
            tp["x0s"], tape, model_params, max_iter=PF_MAX_ITER
        )
        log_jac = jnp.log(1.0 - phi**2 + 1e-12)
        return log_Z + log_jac

    sampler = samplers.ParallelMALA(
        log_prob, D_PARAM, K_MALA, max_iter=1,
        autodiff_mode="forward",
    )

    def run_one_chain(chain_key, init_state):
        mala_params = {
            "step_size":     STEP_SIZE,
            "target_params": target_params,
            "grad_max_norm": 50.0,
        }
        return sampler.run_sequential_mala(chain_key, init_state, mala_params)

    run_multi = jax.jit(jax.vmap(run_one_chain, in_axes=(0, 0)))

    print("=" * 110)
    print(f"{'M chains':>9}  {'t_total (s)':>13}  {'speedup-vs-1':>13}  "
          f"{'phi mean':>9}  {'phi std':>8}  {'R-hat':>7}  "
          f"{'ESS/chain':>10}  {'ESS total':>10}")
    print("-" * 110)

    t_one_chain = None
    for M in M_CHAINS_SWEEP:
        key, key_seeds, key_inits = jr.split(key, 3)
        chain_keys = jr.split(key_seeds, M)
        # Overdispersed inits within the region where the pseudo-marginal
        # estimate is usable: far from the truth (phi < ~0.8) the twisted
        # log-Z estimate degenerates (huge negative values, ~0 acceptance)
        # and chains stick at their inits. phi ~ U(0.85, 0.99) is still
        # ~8 posterior std devs wide, so R-hat remains meaningful.
        chain_inits = jr.uniform(
            key_inits, (M, D_PARAM),
            minval=jnp.arctanh(0.85), maxval=jnp.arctanh(0.99),
        )

        t, states = time_min(run_multi, chain_keys, chain_inits)
        states.block_until_ready()                            # (M, K_MALA, D)
        phis = jnp.tanh(states[:, :, 0])                       # (M, K_MALA)

        # Diagnostics on post-burn-in samples
        burn = int(BURN_IN_FRAC * K_MALA)
        post = phis[:, burn:]                                  # (M, K_MALA-burn)

        if M == 1:
            r_hat = jnp.array(jnp.nan)                         # need >=2 chains
        else:
            r_hat = r_hat_1d(post)
        ess_pc, ess_tot = ess_1d(post)

        phi_mean = float(jnp.mean(post))
        phi_std  = float(jnp.std(post))

        if t_one_chain is None and M == 1:
            t_one_chain = t
        speedup = (t_one_chain * M / t) if t_one_chain else 1.0

        rhat_str = "n/a" if jnp.isnan(r_hat) else f"{float(r_hat):.4f}"
        print(f"{M:>9}  {t:>13.4f}  {speedup:>12.2f}x  "
              f"{phi_mean:>+9.4f}  {phi_std:>8.4f}  {rhat_str:>7}  "
              f"{float(jnp.mean(ess_pc)):>10.1f}  {float(ess_tot):>10.1f}")


if __name__ == "__main__":
    main()
