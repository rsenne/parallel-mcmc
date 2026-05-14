"""
run_stampede_lgssm.py

Smallest end-to-end demo of the revived "STAMPEDE" idea:
  parallel-in-time MALA  +  parallel-in-time twisted-Gumbel PF.

Each MALA step accepts/rejects a proposed parameter theta using a
log-likelihood log p(y | theta) computed by the windowed Gumbel-STE
particle filter built in this branch. The MALA chain is solved in
parallel-in-time via DEER (src.samplers.ParallelMALA), and inside
each DEER iteration the K_MALA log-likelihood evaluations are vmap'd
into K_MALA simultaneous PF runs -- each itself parallel-in-time.

The compositional parallel depth is therefore
    O( <const> * <log T> )
because the K_MALA evaluations all live on the vmap axis (width),
and each PF is internally parallel-in-time (depth log T-ish).

Model
-----
1D LGSSM with a SINGLE unknown parameter phi (state-transition coeff):
    x_t = phi * x_{t-1}  +  sigma_q * eps_t
    y_t = x_t            +  sigma_r * eta_t
sigma_q and sigma_r are fixed at their true values for simplicity --
this isolates the wiring question (does parallel MALA + parallel PF
compose?) from any tuning of multi-parameter MALA.

Sanity check
------------
Sequential MALA and parallel MALA are run on the SAME PRNG drivers
and SAME PF tape. With DEER converged the two chains must agree
up to numerical tolerance. Reporting:
    - final-state agreement,
    - max trajectory difference,
    - DEER iters on MALA,
    - wall-clock.
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src import samplers
from src.gumbel_particle_filter import TwistedGumbelParticleFilter


# --- model truth ---
PHI_TRUE = 0.9
SIGMA_Q  = 0.5
SIGMA_R  = 1.0

# --- PF (inner) ---
T   = 1024
N   = 32
K_W = 64      # PF window size
TAU = 1.0
PF_MAX_ITER = 100

# --- MALA (outer) ---
D_PARAM       = 1
CHAIN_LEN     = 256
STEP_SIZE     = 5e-3
MALA_MAX_ITER = 1000
MALA_K_BLOCK  = 50          # windowed MALA block size (memory: ~50 is sweet spot)
PHI_ANCHOR    = 0.0         # anchor for the twist; NOT using PHI_TRUE


# ---- twisted-PF step / log-weight (LGSSM, diagonal twist) ----

def backward_information_filter_1d(ys, phi, sigma_q, sigma_r):
    """Serial RTS-optimal BIF for the 1D LGSSM. Kept as a reference for
    parallel_bif_1d's correctness check."""
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2
    def step(carry, y):
        J, h = carry
        denom = 1.0 + sq * J
        J_new = phi**2 * J / denom + prec_obs
        h_new = phi * h / denom + y * prec_obs
        return (J_new, h_new), (J_new, h_new)
    J_T = prec_obs
    h_T = ys[-1] * prec_obs
    _, (Js_rev, hs_rev) = jax.lax.scan(step, (J_T, h_T), ys[-2::-1])
    Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_T])])
    hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_T])])
    return hs / Js, 1.0 / Js


def parallel_bif_1d(ys, phi, sigma_q, sigma_r):
    """Parallel BIF for the 1D LGSSM in O(log T) depth.

    For stationary (phi, sigma_q, sigma_r) the Riccati update on J has a
    closed-form fixed point J* (positive root of a quadratic):

        sq * J*^2 + (1 - phi^2 - sq * prec_obs) * J* - prec_obs = 0

    Away from the t=T boundary the BIF J value decays into J* exponentially
    fast, so using J_t = J* everywhere is an excellent approximation. With
    J constant, the h recursion

        h_{t-1} = alpha * h_t + y_{t-1} * prec_obs,   alpha = phi / (1 + sq*J*)

    is a 1D affine recurrence with constant coefficient -- solvable in
    O(log T) depth via jax.lax.associative_scan with the affine
    composition (a,b) o (a',b') = (a'a, a'b + b').
    """
    T = ys.shape[0]
    prec_obs = 1.0 / sigma_r**2
    sq = sigma_q**2

    # Closed-form stationary Riccati J*.
    b = 1.0 - phi**2 - sq * prec_obs
    J_star = (-b + jnp.sqrt(b**2 + 4.0 * sq * prec_obs)) / (2.0 * sq)
    alpha = phi / (1.0 + sq * J_star)

    # Affine associative scan to recover h_t = sum_{k>=t} alpha^{k-t} y_k prec_obs.
    # Steps applied in backward-time order: t = T-1, T-2, ..., 0.
    # Each step is the affine map (a_step, b_step) = (alpha, y_t * prec_obs)
    # acting on the running h (starting from h=0). After scanning, the
    # cumulative affine at scan position k, applied to h=0, gives h_{T-1-k}.
    alphas    = jnp.full(T, alpha)
    betas_rev = ys[::-1] * prec_obs

    def compose(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_r * a_l, a_r * b_l + b_r)

    _, cum_betas = jax.lax.associative_scan(compose, (alphas, betas_rev))
    hs = cum_betas[::-1]                   # h_0, h_1, ..., h_{T-1}

    twist_means = hs / J_star
    twist_vars  = jnp.full(T, 1.0 / J_star)
    return twist_means, twist_vars


def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi = params["phi"]; sq = params["sigma_q"]
    mu = twist["mean"]; tau = jnp.exp(twist["log_scale"])
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    phi = params["phi"]; sq = params["sigma_q"]; sr = params["sigma_r"]
    mu = twist["mean"]; tau = jnp.exp(twist["log_scale"])
    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sr**2
                 - 0.5 * jnp.log(2 * jnp.pi * sr**2))
    log_trans = (-0.5 * jnp.sum((x_next - phi * x_prev)**2) / sq**2
                 - 0.5 * jnp.log(2 * jnp.pi * sq**2))
    pq = 1.0 / sq**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    log_prop  = (-0.5 * jnp.sum((x_next - mp)**2 / vp)
                 - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * vp)))
    return log_lik + log_trans - log_prop


# ---- log-posterior used by MALA ----
# theta = [phi]. Prior on phi: N(0, 1). Likelihood from the parallel-in-time PF.
# The PF tape and initial particles are passed via target_params and held
# fixed across MALA steps -> log_prob is a deterministic function of theta,
# which makes the sequential-vs-parallel MALA comparison crisp.

def build_log_prob(tgpf):
    # Twist is precomputed ONCE (at PHI_ANCHOR) and lives inside tp["tape"].
    # The PF is unbiased for any valid twist -- anchoring at the wrong phi
    # only increases variance, not bias. So we keep the BIF out of the hot
    # path and let log_prob just run the parallel PF at the proposed phi.
    def log_prob(theta, tp):
        phi = jnp.tanh(theta[0])
        model_params = {
            "phi":     phi,
            "sigma_q": SIGMA_Q,
            "sigma_r": SIGMA_R,
            "tau":     TAU,
            "key":     tp["deer_key"],
        }
        _, _, log_Z, _ = tgpf.run_parallel(
            tp["x0s"], tp["tape"], model_params, max_iter=PF_MAX_ITER
        )
        # Uniform(-1, 1) prior on phi -> log-jacobian in theta-space.
        log_jac = jnp.log(1.0 - phi**2 + 1e-12)
        return log_Z + log_jac
    return log_prob


def main():
    print(f"Device: {jax.devices()}")
    print(f"STAMPEDE LGSSM sanity check  "
          f"T={T}  N={N}  K_W={K_W}  CHAIN_LEN={CHAIN_LEN}  STEP={STEP_SIZE}")

    key = jr.PRNGKey(0)
    key, k_data1, k_data2, k_data3 = jr.split(key, 4)

    # --- simulate observations under PHI_TRUE ---
    x0 = jr.normal(k_data1)
    def sim_step(x, qr):
        xn = PHI_TRUE * x + SIGMA_Q * qr[0]
        return xn, (xn, xn + SIGMA_R * qr[1])
    _, (true_xs, ys) = jax.lax.scan(
        sim_step, x0, (jr.normal(k_data2, (T,)), jr.normal(k_data3, (T,)))
    )

    # Precompute the BIF twist ONCE at PHI_ANCHOR=0 via the PARALLEL BIF
    # (closed-form Riccati J* + associative-scan affine recurrence for h).
    # Cross-check vs the serial BIF for correctness.
    twist_means, twist_vars = parallel_bif_1d(ys, PHI_ANCHOR, SIGMA_Q, SIGMA_R)
    twist_means_ref, twist_vars_ref = backward_information_filter_1d(
        ys, PHI_ANCHOR, SIGMA_Q, SIGMA_R
    )
    bif_mean_err = float(jnp.max(jnp.abs(twist_means - twist_means_ref)))
    bif_var_err  = float(jnp.max(jnp.abs(twist_vars  - twist_vars_ref)))
    print(f"  parallel_bif vs serial_bif:  max mean err = {bif_mean_err:.3e}  "
          f"max var err = {bif_var_err:.3e}")
    twist_params = {
        "mean":      twist_means[:, None],
        "log_scale": 0.5 * jnp.log(twist_vars)[:, None],
    }

    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=1, tau=TAU)
    key, tape_key, x0s_key, deer_key = jr.split(key, 4)
    tape = tgpf.make_tape(ys[:, None], twist_params, tape_key, N=N, K=K_W)
    x0s = jr.normal(x0s_key, (N, 1))

    target_params = {
        "tape":     tape,
        "x0s":      x0s,
        "deer_key": deer_key,
    }
    log_prob = build_log_prob(tgpf)

    # --- one-off correctness probe of log_prob ---
    theta_05    = jnp.atanh(jnp.array([0.5]))
    theta_true  = jnp.atanh(jnp.array([PHI_TRUE]))
    lp0 = log_prob(theta_05, target_params)
    lp1 = log_prob(theta_true, target_params)
    print(f"  log p(phi=0.5)       = {float(lp0):+.4f}")
    print(f"  log p(phi=phi_true)  = {float(lp1):+.4f}   "
          f"(should be the higher one)")

    # --- MALA setup ---
    # autodiff_mode='forward' is REQUIRED because log_prob contains
    # jax.lax.while_loop (inside qdeer.seq1d) which reverse-mode autodiff
    # cannot handle.
    # window_size=MALA_K_BLOCK -> use run_parallel_mala_window, which
    # solves the chain in chunks via windowed_qdeer. Iters/window stays
    # O(1) rather than scaling with the full chain length.
    sampler = samplers.ParallelMALA(
        log_prob, D_PARAM, CHAIN_LEN, MALA_MAX_ITER,
        autodiff_mode="forward",
        window_size=MALA_K_BLOCK,
    )

    key, mala_key, mala_deer_key = jr.split(key, 3)
    initial_state = jnp.array([0.0])                       # start at prior mean
    mala_params = {
        "step_size":     STEP_SIZE,
        "target_params": target_params,
        "grad_max_norm": 50.0,
        "key":           mala_deer_key,                    # for outer DEER Hutchinson
    }

    # --- sequential MALA ---
    run_seq = jax.jit(sampler.run_sequential_mala)
    _ = run_seq(mala_key, initial_state, mala_params)              # warmup
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_seq = run_seq(mala_key, initial_state, mala_params)
    states_seq.block_until_ready()
    t_seq = time.perf_counter() - t0

    # --- parallel MALA (windowed) ---
    yinit_guess = jnp.tile(initial_state[None, :], (CHAIN_LEN, 1))
    run_par = jax.jit(sampler.run_parallel_mala_window)
    _ = run_par(mala_key, initial_state, yinit_guess, mala_params)  # warmup
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_par, iters = run_par(mala_key, initial_state, yinit_guess, mala_params)
    states_par.block_until_ready()
    t_par = time.perf_counter() - t0

    max_diff = float(jnp.max(jnp.abs(states_seq - states_par)))
    # Chains are in theta-space; report phi = tanh(theta) for interpretability.
    phis_seq = jnp.tanh(states_seq[:, 0])
    phis_par = jnp.tanh(states_par[:, 0])

    print("=" * 72)
    print(f"Sequential MALA : time = {t_seq:.4f}s   final phi = {float(phis_seq[-1]):+.4f}")
    print(f"Parallel   MALA : time = {t_par:.4f}s   final phi = {float(phis_par[-1]):+.4f}   "
          f"DEER iters = {int(iters)}")
    print(f"Speedup         : {t_seq / t_par:.2f}x")
    print(f"max |theta_seq - theta_par| = {max_diff:.3e}")
    print(f"true phi                    = {PHI_TRUE:+.4f}")
    print(f"chain phi (seq):  {[round(float(p), 4) for p in phis_seq]}")
    print(f"chain phi (par):  {[round(float(p), 4) for p in phis_par]}")


if __name__ == "__main__":
    main()
