"""
run_twisted_gumbel_pf_stochvol.py

Stress test: twisted-Gumbel parallel PF on the stochastic-volatility
model, where the observation is non-Gaussian and there is no
closed-form optimal twist. We use the iterated-linearised backward
information filter from examples/run_twisted_pf_stochvol.py to get an
*approximate* Gaussian twist.

What this tests:
  - Forward law equivalence (max |par - seq|) on a nonlinear model.
  - ESS at each resample point — with an approximate twist, weights
    are NOT exactly uniform, so ESS will reflect the proposal quality.
    A twist that holds ESS near N validates the windowed-resample
    design (rare, information-poor resamples).
  - Comparison vs a flat twist (mean=0, large scale) which has poor
    proposal quality and forces resampling to do real work.

Model
-----
  x_t = phi * x_{t-1} + sigma_q * eps_t       (log-volatility, latent)
  y_t = beta * exp(x_t / 2) * eta_t           (observation)
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src.gumbel_particle_filter import TwistedGumbelParticleFilter


PHI     = 0.95
SIGMA_Q = 0.2
BETA    = 1.0

T   = 1024
D   = 1
N   = 64
K   = 32
TAU = 1.0
MAX_ITER = 200


def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    phi = params["phi"]; sigma_q = params["sigma_q"]
    mu = twist["mean"]; tau = jnp.exp(twist["log_scale"])
    pq = 1.0 / sigma_q**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    phi = params["phi"]; sigma_q = params["sigma_q"]; beta = params["beta"]
    mu = twist["mean"]; tau = jnp.exp(twist["log_scale"])

    # log p(y_t | x_t)  --  SV likelihood (non-Gaussian in obs)
    log_lik = (-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(beta)
               - x_next[0] / 2.0
               - obs[0]**2 / (2.0 * beta**2 * jnp.exp(x_next[0])))

    log_trans = (-0.5 * (x_next[0] - phi * x_prev[0])**2 / sigma_q**2
                 - 0.5 * jnp.log(2.0 * jnp.pi * sigma_q**2))

    pq = 1.0 / sigma_q**2; pt = 1.0 / tau**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * phi * x_prev + pt * mu)
    log_prop = (-0.5 * (x_next[0] - mp[0])**2 / vp[0]
                - 0.5 * jnp.log(2.0 * jnp.pi * vp[0]))

    return log_lik + log_trans - log_prop


def approx_backward_info_filter_sv(ys, phi, sigma_q, beta, n_iter=10):
    """Iterated-linearisation Gaussian backward info filter for SV.

    Returns twist_means (T,), twist_vars (T,). Copied from
    examples/run_twisted_pf_stochvol.py.
    """
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
            step, (J_obs_T, h_obs_T),
            (ys[-2::-1], twist_means[-2::-1]),
        )
        Js = jnp.concatenate([Js_rev[::-1], jnp.array([J_obs_T])])
        hs = jnp.concatenate([hs_rev[::-1], jnp.array([h_obs_T])])
        twist_means = hs / Js

    return twist_means, 1.0 / Js


def run_with_twist(twist_means, twist_vars, label, x0s, ys_2d, key):
    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=D, tau=TAU)
    twist_params = {
        "mean":      twist_means[:, None],
        "log_scale": 0.5 * jnp.log(twist_vars)[:, None],
    }
    model_params = {"phi": PHI, "sigma_q": SIGMA_Q, "beta": BETA}

    tape_key, deer_key = jr.split(key)
    tape = tgpf.make_tape(ys_2d, twist_params, tape_key, N=N, K=K)

    run_seq = jax.jit(tgpf.run_sequential)
    run_par = jax.jit(tgpf.run_parallel, static_argnames=("max_iter",))

    _ = run_seq(x0s, tape, model_params)
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_seq, _, logZ_seq, ess_seq = run_seq(x0s, tape, model_params)
    jax.block_until_ready(states_seq)
    t_seq = time.perf_counter() - t0

    params_par = {**model_params, "key": deer_key, "tau": TAU}
    _ = run_par(x0s, tape, params_par, max_iter=MAX_ITER)
    jax.block_until_ready(_)
    t0 = time.perf_counter()
    states_par, iters, logZ_par, ess_par = run_par(
        x0s, tape, params_par, max_iter=MAX_ITER
    )
    jax.block_until_ready(states_par)
    t_par = time.perf_counter() - t0

    max_diff   = float(jnp.max(jnp.abs(states_par - states_seq)))
    max_iters  = int(jnp.max(iters))
    mean_iters = float(jnp.mean(iters))

    print(f"\n--- {label} ---")
    print(f"Seq : time = {t_seq:.4f}s   log Z = {float(logZ_seq):+.3f}")
    print(f"Par : time = {t_par:.4f}s   log Z = {float(logZ_par):+.3f}   "
          f"speedup = {t_seq/t_par:.2f}x")
    print(f"max |par - seq|              = {max_diff:.3e}")
    print(f"DEER iters per window  max   = {max_iters}    mean = {mean_iters:.2f}")
    print(f"ESS/N (seq, per resample)    min={float(jnp.min(ess_seq))/N:.3f}  "
          f"mean={float(jnp.mean(ess_seq))/N:.3f}  "
          f"frac < 0.5: {float(jnp.mean(ess_seq < 0.5*N)):.3f}")
    print(f"ESS/N (par, per resample)    min={float(jnp.min(ess_par))/N:.3f}  "
          f"mean={float(jnp.mean(ess_par))/N:.3f}")


def main():
    print(f"Device: {jax.devices()}")
    print(f"SV model:  phi={PHI}  sigma_q={SIGMA_Q}  beta={BETA}")
    print(f"T={T}  N={N}  D={D}  K={K}  num_windows={T//K}  tau={TAU}")

    key = jr.PRNGKey(0)
    key, k1, k2, k3, k4, k5 = jr.split(key, 6)

    x0_true = jr.normal(k1) * 0.5

    def sim_step(x, qr):
        q, r = qr
        xn = PHI * x + SIGMA_Q * q
        return xn, (xn, BETA * jnp.exp(xn / 2.0) * r)
    _, (true_xs, ys) = jax.lax.scan(
        sim_step, x0_true, (jr.normal(k2, (T,)), jr.normal(k3, (T,)))
    )
    ys_2d = ys[:, None]

    x0s = jr.normal(k4, (N, D)) * 0.5

    # --- Twist 1: approximate BIF (good twist) ---
    bif_means, bif_vars = approx_backward_info_filter_sv(
        ys, PHI, SIGMA_Q, BETA, n_iter=10
    )
    run_with_twist(bif_means, bif_vars, "approx-BIF twist (good)",
                   x0s, ys_2d, k5)

    # --- Twist 2: flat / weak twist (bootstrap-ish) ---
    flat_means = jnp.zeros(T)
    flat_vars  = 100.0 * jnp.ones(T)   # near-uninformative twist
    run_with_twist(flat_means, flat_vars, "flat twist (~bootstrap)",
                   x0s, ys_2d, k5)


if __name__ == "__main__":
    main()
