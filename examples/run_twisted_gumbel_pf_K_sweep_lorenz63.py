"""
run_twisted_gumbel_pf_K_sweep_lorenz63.py

Twisted-Gumbel parallel PF on Lorenz-63 (D=3, chaotic), with a
window-size sweep.

Model
-----
  x_t = x_{t-1} + dt * f_lorenz(x_{t-1}) + sigma_q * eps_t
  y_t = x_t + sigma_r * eta_t

The twist is a per-step Gaussian with diagonal covariance, initialised
from an EKF smoother + extended backward information filter (EBIF).
Both come from examples/run_twisted_pf_lorenz63.py.
"""

import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from src.gumbel_particle_filter import TwistedGumbelParticleFilter


SIGMA_L = 10.0
RHO     = 28.0
BETA_L  = 8.0 / 3.0
DT      = 0.01

SIGMA_Q = 0.5
SIGMA_R = 2.0

D = 3
N = 64
T = 4096
TAU = 1.0
MAX_ITER = 200


def lorenz_deriv(x):
    return jnp.array([
        SIGMA_L * (x[1] - x[0]),
        x[0] * (RHO - x[2]) - x[1],
        x[0] * x[1] - BETA_L * x[2],
    ])


def lorenz_step(x):
    return x + DT * lorenz_deriv(x)


def lorenz_jacobian(x):
    J = jnp.array([
        [-SIGMA_L,    SIGMA_L,   0.0],
        [RHO - x[2], -1.0,       -x[0]],
        [x[1],        x[0],      -BETA_L],
    ])
    return jnp.eye(D) + DT * J


# ---- EKF smoother (gives twist linearisation points) ----

def ekf_smoother(ys, x0):
    Q = SIGMA_Q**2 * jnp.eye(D)
    R = SIGMA_R**2 * jnp.eye(D)

    def fwd_step(carry, y):
        m, P = carry
        m_pred = lorenz_step(m)
        J = lorenz_jacobian(m)
        P_pred = J @ P @ J.T + Q
        S = P_pred + R
        K = jnp.linalg.solve(S.T, P_pred.T).T
        m_new = m_pred + K @ (y - m_pred)
        P_new = (jnp.eye(D) - K) @ P_pred
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new)

    _, (m_preds, P_preds, m_filts, P_filts) = jax.lax.scan(
        fwd_step, (x0, jnp.eye(D)), ys
    )

    def bwd_step(carry, t):
        m_s, P_s = carry
        J = lorenz_jacobian(m_filts[t])
        G = jnp.linalg.solve(P_preds[t + 1].T, (P_filts[t] @ J.T).T).T
        m_s_new = m_filts[t] + G @ (m_s - m_preds[t + 1])
        P_s_new = P_filts[t] + G @ (P_s - P_preds[t + 1]) @ G.T
        return (m_s_new, P_s_new), (m_s_new, P_s_new)

    ts = jnp.arange(ys.shape[0] - 2, -1, -1)
    _, (ms_rev, Ps_rev) = jax.lax.scan(
        bwd_step, (m_filts[-1], P_filts[-1]), ts
    )
    smooth_means = jnp.concatenate([ms_rev[::-1], m_filts[-1:]])
    return smooth_means


def ebif_diag(ys, ekf_means):
    """Extended backward information filter — diagonal-twist outputs."""
    Q = SIGMA_Q**2 * jnp.eye(D)
    R_inv = (1.0 / SIGMA_R**2) * jnp.eye(D)

    def step(carry, inputs):
        J, h = carry
        y_t, lin_pt = inputs
        F = lorenz_jacobian(lin_pt)
        b = lorenz_step(lin_pt) - F @ lin_pt
        J_inv = jnp.linalg.inv(J)
        P = J_inv + Q
        P_inv = jnp.linalg.inv(P)
        J_pred = F.T @ P_inv @ F
        h_pred = F.T @ P_inv @ (J_inv @ h - b)
        J_new = J_pred + R_inv
        h_new = h_pred + R_inv @ y_t
        return (J_new, h_new), (J_new, h_new)

    J_T = R_inv
    h_T = R_inv @ ys[-1]
    _, (Js_rev, hs_rev) = jax.lax.scan(
        step, (J_T, h_T), (ys[-2::-1], ekf_means[-2::-1])
    )
    Js = jnp.concatenate([Js_rev[::-1], J_T[None]], axis=0)
    hs = jnp.concatenate([hs_rev[::-1], h_T[None]], axis=0)

    def extract(J, h):
        J_inv = jnp.linalg.inv(J)
        return J_inv @ h, jnp.diag(J_inv)

    twist_means, twist_vars = jax.vmap(extract)(Js, hs)
    return twist_means, twist_vars


# ---- Twisted step + log-weight, diagonal twist ----

def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    sq = params["sigma_q"]
    mu = twist["mean"]              # (D,)
    tau_d = jnp.exp(twist["log_scale"])  # (D,) diagonal std
    f_prev = lorenz_step(x_prev)
    pq = 1.0 / sq**2
    pt = 1.0 / tau_d**2
    vp = 1.0 / (pq + pt)            # (D,)
    mp = vp * (pq * f_prev + pt * mu)
    return mp + jnp.sqrt(vp) * noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    sq = params["sigma_q"]; sr = params["sigma_r"]
    mu = twist["mean"]
    tau_d = jnp.exp(twist["log_scale"])
    f_prev = lorenz_step(x_prev)
    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sr**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sr**2))
    log_trans = (-0.5 * jnp.sum((x_next - f_prev)**2) / sq**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sq**2))
    pq = 1.0 / sq**2
    pt = 1.0 / tau_d**2
    vp = 1.0 / (pq + pt)
    mp = vp * (pq * f_prev + pt * mu)
    log_prop  = (-0.5 * jnp.sum((x_next - mp)**2 / vp)
                 - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * vp)))
    return log_lik + log_trans - log_prop


# ---- Timing helper ----

def time_min(fn, *args, repeat=3):
    out = fn(*args); jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    return min(times), out


def run_one_K(K, ys, twist_params, x0s, key):
    model_params = {"sigma_q": SIGMA_Q, "sigma_r": SIGMA_R}
    tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                       D=D, tau=TAU)
    tape_key, deer_key = jr.split(key)
    tape = tgpf.make_tape(ys, twist_params, tape_key, N=N, K=K)

    run_seq = jax.jit(tgpf.run_sequential)
    run_par = jax.jit(tgpf.run_parallel, static_argnames=("max_iter",))
    params_par = {**model_params, "key": deer_key, "tau": TAU}

    t_seq, (states_seq, _, _, ess_seq) = time_min(run_seq, x0s, tape, model_params)
    t_par, (states_par, iters, _, ess_par) = time_min(
        lambda x0_, t_, p_: run_par(x0_, t_, p_, max_iter=MAX_ITER),
        x0s, tape, params_par,
    )
    err = float(jnp.max(jnp.abs(states_par - states_seq)))
    return (
        t_seq, t_par,
        int(jnp.max(iters)), float(jnp.mean(iters)),
        err,
        float(jnp.min(ess_par)) / N, float(jnp.mean(ess_par)) / N,
    )


def main():
    print(f"Device: {jax.devices()}")
    print(f"Lorenz63 K-sweep   T={T}  N={N}  D={D}  tau={TAU}")

    key = jr.PRNGKey(0)
    key, k1, k2, k3, k4, k5 = jr.split(key, 6)

    # Spin up Lorenz initial condition off the attractor
    x0_true = jnp.array([1.0, 1.0, 1.0])
    x0_true, _ = jax.lax.scan(lambda x, _: (lorenz_step(x), None),
                              x0_true, None, length=5000)

    def sim_step(x, qr):
        q, r = qr
        xn = lorenz_step(x) + SIGMA_Q * q
        return xn, (xn, xn + SIGMA_R * r)
    _, (true_xs, ys) = jax.lax.scan(
        sim_step, x0_true,
        (jr.normal(k2, (T, D)), jr.normal(k3, (T, D)))
    )

    ekf_means = jax.jit(ekf_smoother)(ys, x0_true)
    ebif_means, ebif_vars = jax.jit(ebif_diag)(ys, ekf_means)
    ekf_rmse  = float(jnp.sqrt(jnp.mean((ekf_means  - true_xs)**2)))
    ebif_rmse = float(jnp.sqrt(jnp.mean((ebif_means - true_xs)**2)))
    print(f"  EKF smoother RMSE = {ekf_rmse:.4f}")
    print(f"  EBIF twist  RMSE = {ebif_rmse:.4f}")

    twist_params = {
        "mean":      ebif_means,                       # (T, D)
        "log_scale": 0.5 * jnp.log(ebif_vars),         # (T, D)
    }

    x0s = x0_true[None, :] + 0.5 * jr.normal(k4, (N, D))

    print("=" * 105)
    print(f"{'K':>5}  {'num_w':>6}  {'t_seq (s)':>11}  {'t_par (s)':>11}  "
          f"{'speedup':>8}  {'iters max':>10}  {'max|par-seq|':>13}  "
          f"{'ESS/N min':>10}  {'ESS/N mean':>10}")
    print("-" * 105)

    Ks = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for K in Ks:
        if T % K != 0:
            continue
        try:
            r = run_one_K(K, ys, twist_params, x0s, k5)
            t_seq, t_par, max_it, mean_it, err, ess_min, ess_mean = r
            print(f"{K:>5}  {T//K:>6}  {t_seq:>11.4f}  {t_par:>11.4f}  "
                  f"{t_seq/t_par:>7.2f}x  {max_it:>10}  {err:>13.2e}  "
                  f"{ess_min:>10.3f}  {ess_mean:>10.3f}")
        except Exception as e:
            print(f"{K:>5}  failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
