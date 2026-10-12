"""
run_twisted_gumbel_pf_lorenz63_learned.py

Twisted-Gumbel parallel PF on Lorenz-63 with a *learned* full-covariance
Gaussian twist (mean + Cholesky), refined by Adam on the no-resampling
sequential ELBO.

What this script does
---------------------
  1. Simulate Lorenz-63 with observation noise.
  2. Build initial full-cov twist from the EKF smoother posterior.
  3. Refine the twist via gradient descent on
         -log sum_i exp( sum_t log w_t^i )  /  N
     using the differentiable no-resampling twisted PF
     (src.particle_filter.TwistedParallelParticleFilter).
  4. Deploy the learned twist in our windowed Gumbel-STE PF and run a
     K-sweep, comparing wall-clock vs sequential, ESS, and DEER iters.

Why the no-resampling ELBO for training
---------------------------------------
The Gumbel-STE forward is non-differentiable at the argmax. Training
through it via the soft Jacobian would give biased gradients (same as
the gradient-through-Gumbel-Softmax literature). Instead we train the
twist on the equivalent no-resampling proposal — the same
twisted_step_fn / log_weight_fn — and only deploy resampling at
inference time. The trained twist is a property of the proposal, not
of the resampling schedule, so this is a clean factorisation.
"""

import os
import pickle
import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.gumbel_particle_filter import TwistedGumbelParticleFilter
from src.particle_filter import TwistedParallelParticleFilter


# Model
SIGMA_L = 10.0
RHO     = 28.0
BETA_L  = 8.0 / 3.0
DT      = 0.01

SIGMA_Q = 0.5
SIGMA_R = 2.0

D = 3
N = 64
T = 2048
TAU = 1.0
MAX_ITER = 200

# Training
N_TRAIN_STEPS = 1500
LR_MAX = 1e-2
LR_MIN = 1e-4
CACHE_PATH = "examples/_lorenz_learned_twist_T2048.pkl"


# -------- Lorenz-63 dynamics --------

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


# -------- EKF smoother for twist init --------

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
    smooth_covs  = jnp.concatenate([Ps_rev[::-1], P_filts[-1:]])
    return smooth_means, smooth_covs


# -------- Full-cov twist parameterisation --------

def build_chol(chol_log_diag, chol_offdiag):
    return jnp.tril(chol_offdiag, k=-1) + jnp.diag(jnp.exp(chol_log_diag))


def _twist_posterior(f_prev, twist, sigma_q):
    mu_twist = twist["mean"]
    L_twist  = build_chol(twist["chol_log_diag"], twist["chol_offdiag"])
    prec_q = 1.0 / sigma_q**2
    Sigma_twist_inv = jax.scipy.linalg.cho_solve((L_twist, True), jnp.eye(D))
    prec_post = prec_q * jnp.eye(D) + Sigma_twist_inv
    Sigma_post = jnp.linalg.inv(prec_post)
    mean_post = Sigma_post @ (prec_q * f_prev + Sigma_twist_inv @ mu_twist)
    L_post = jnp.linalg.cholesky(Sigma_post)
    return mean_post, L_post


def twisted_step_fn(x_prev, driver, params):
    obs, noise, twist = driver
    sigma_q = params["sigma_q"]
    f_prev = lorenz_step(x_prev)
    mean_post, L_post = _twist_posterior(f_prev, twist, sigma_q)
    return mean_post + L_post @ noise


def log_weight_fn(x_prev, x_next, obs, twist, params):
    sigma_q = params["sigma_q"]; sigma_r = params["sigma_r"]
    f_prev = lorenz_step(x_prev)
    log_lik   = (-0.5 * jnp.sum((obs - x_next)**2) / sigma_r**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_r**2))
    log_trans = (-0.5 * jnp.sum((x_next - f_prev)**2) / sigma_q**2
                 - 0.5 * D * jnp.log(2 * jnp.pi * sigma_q**2))
    mean_post, L_post = _twist_posterior(f_prev, twist, sigma_q)
    diff = x_next - mean_post
    z    = jax.scipy.linalg.solve_triangular(L_post, diff, lower=True)
    log_prop = (-0.5 * jnp.sum(z**2)
                - jnp.sum(jnp.log(jnp.diag(L_post)))
                - 0.5 * D * jnp.log(2 * jnp.pi))
    return log_lik + log_trans - log_prop


# -------- Adam (full-cov twist) --------

def adam_init_tree(*params):
    return tuple((jnp.zeros_like(p), jnp.zeros_like(p)) for p in params) + (0,)


def adam_step_tree(grads, state, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
    *mv_pairs, t = state
    t = t + 1
    updates, new_pairs = [], []
    for g, (m, v) in zip(grads, mv_pairs):
        m_new = b1 * m + (1 - b1) * g
        v_new = b2 * v + (1 - b2) * g**2
        m_hat = m_new / (1 - b1**t)
        v_hat = v_new / (1 - b2**t)
        updates.append(-lr * m_hat / (jnp.sqrt(v_hat) + eps))
        new_pairs.append((m_new, v_new))
    return tuple(updates), tuple(new_pairs) + (t,)


def train_twist(ekf_means, chol_log_diag_init, chol_offdiag_init,
                x0s, noise, ys_2d, model_params):
    tpf = TwistedParallelParticleFilter(twisted_step_fn, log_weight_fn, D=D)

    def loss_fn(twist_mean, twist_log_diag, twist_offdiag):
        twist_params = {
            "mean":          twist_mean,
            "chol_log_diag": twist_log_diag,
            "chol_offdiag":  twist_offdiag,
        }
        tape   = (ys_2d, noise, twist_params)
        states = tpf.run_sequential(x0s, tape, model_params)
        log_ws = tpf.compute_log_weights(x0s, states, tape, model_params)
        total  = jnp.sum(log_ws, axis=1)
        elbo   = jax.scipy.special.logsumexp(total) - jnp.log(float(N))
        return -elbo

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))

    twist_mean     = ekf_means
    twist_log_diag = chol_log_diag_init
    twist_offdiag  = chol_offdiag_init
    opt_state = adam_init_tree(twist_mean, twist_log_diag, twist_offdiag)
    losses = []

    t0 = time.perf_counter()
    for i in range(N_TRAIN_STEPS):
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (
            1 + jnp.cos(jnp.pi * i / N_TRAIN_STEPS)
        )
        loss, grads = grad_fn(twist_mean, twist_log_diag, twist_offdiag)
        updates, opt_state = adam_step_tree(grads, opt_state, lr=lr)
        twist_mean     = twist_mean     + updates[0]
        twist_log_diag = twist_log_diag + updates[1]
        twist_offdiag  = twist_offdiag  + updates[2]
        losses.append(float(loss))
        if (i + 1) % 100 == 0:
            print(f"  step {i+1:>4}/{N_TRAIN_STEPS}  neg-ELBO = {loss:.2f}")
    t_train = time.perf_counter() - t0

    return twist_mean, twist_log_diag, twist_offdiag, losses, t_train


# -------- Timing helper --------

def time_min(fn, *args, repeat=3):
    out = fn(*args); jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    return min(times), out


# -------- K-sweep --------

def K_sweep(twist_params, x0s, ys, key, label, damp_factor=1.0):
    print(f"\n--- K-sweep: {label} ---")
    print("=" * 105)
    print(f"{'K':>5}  {'num_w':>6}  {'t_seq (s)':>11}  {'t_par (s)':>11}  "
          f"{'speedup':>8}  {'iters max':>10}  {'max|par-seq|':>13}  "
          f"{'ESS/N min':>10}  {'ESS/N mean':>10}")
    print("-" * 105)

    model_params = {"sigma_q": SIGMA_Q, "sigma_r": SIGMA_R}
    Ks = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for K in Ks:
        if T % K != 0:
            continue
        k_seed, k_run = jr.split(key)
        key = k_seed
        tgpf = TwistedGumbelParticleFilter(twisted_step_fn, log_weight_fn,
                                           D=D, tau=TAU)
        tape = tgpf.make_tape(ys, twist_params, k_run, N=N, K=K)
        run_seq = jax.jit(tgpf.run_sequential)
        run_par = jax.jit(tgpf.run_parallel, static_argnames=("max_iter",))
        params_par = {**model_params, "key": k_run, "tau": TAU}

        t_seq, (states_seq, _, _, ess_seq) = time_min(run_seq, x0s, tape, model_params)
        t_par, (states_par, iters, _, ess_par) = time_min(
            lambda x0_, t_, p_: run_par(x0_, t_, p_, max_iter=MAX_ITER,
                                        damp_factor=damp_factor),
            x0s, tape, params_par,
        )
        err = float(jnp.max(jnp.abs(states_par - states_seq)))
        print(f"{K:>5}  {T//K:>6}  {t_seq:>11.4f}  {t_par:>11.4f}  "
              f"{t_seq/t_par:>7.2f}x  {int(jnp.max(iters)):>10}  "
              f"{err:>13.2e}  "
              f"{float(jnp.min(ess_par))/N:>10.3f}  "
              f"{float(jnp.mean(ess_par))/N:>10.3f}")


def main():
    print(f"Device: {jax.devices()}")
    print(f"Lorenz63 LEARNED twist   T={T}  N={N}  D={D}")

    key = jr.PRNGKey(0)
    key, k1, k2, k3, k4 = jr.split(key, 5)

    # Spin up on attractor
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

    # EKF smoother init for twist
    ekf_means, ekf_covs = jax.jit(ekf_smoother)(ys, x0_true)
    ekf_rmse = float(jnp.sqrt(jnp.mean((ekf_means - true_xs)**2)))
    print(f"  EKF smoother RMSE = {ekf_rmse:.4f}")

    L_ekf = jax.vmap(jnp.linalg.cholesky)(ekf_covs + 1e-6 * jnp.eye(D))
    chol_log_diag_init = jnp.log(jax.vmap(jnp.diag)(L_ekf))
    chol_offdiag_init  = jax.vmap(lambda M: jnp.tril(M, k=-1))(L_ekf)

    x0s = x0_true[None, :] + 0.5 * jr.normal(k4, (N, D))
    noise_train = jr.normal(jr.PRNGKey(42), (N, T, D))   # fixed for training
    model_params = {"sigma_q": SIGMA_Q, "sigma_r": SIGMA_R}

    # Load or train
    if os.path.exists(CACHE_PATH):
        print(f"\n--- Loading cached learned twist from {CACHE_PATH} ---")
        with open(CACHE_PATH, "rb") as f:
            d = pickle.load(f)
        twist_mean     = jnp.array(d["mean"])
        twist_log_diag = jnp.array(d["log_diag"])
        twist_offdiag  = jnp.array(d["offdiag"])
        print(f"  cache loaded  (final neg-ELBO = {d['losses'][-1]:.2f}, "
              f"learn time {d['t_learn']:.1f}s)")
    else:
        print(f"\n--- Training full-cov twist ({N_TRAIN_STEPS} Adam steps) ---")
        twist_mean, twist_log_diag, twist_offdiag, losses, t_train = train_twist(
            ekf_means, chol_log_diag_init, chol_offdiag_init,
            x0s, noise_train, ys, model_params,
        )
        print(f"  done in {t_train:.1f}s   final neg-ELBO = {losses[-1]:.2f}")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "mean":     np.array(twist_mean),
                "log_diag": np.array(twist_log_diag),
                "offdiag":  np.array(twist_offdiag),
                "losses":   losses,
                "t_train":  t_train,
                # Compatibility alias for naming convention of other caches
                "t_learn":  t_train,
            }, f)
        print(f"  cached to {CACHE_PATH}")

    # Baseline: K-sweep with EKF-init full-cov twist (no training)
    twist_ekf = {
        "mean":          ekf_means,
        "chol_log_diag": chol_log_diag_init,
        "chol_offdiag":  chol_offdiag_init,
    }
    K_sweep(twist_ekf, x0s, ys, jr.PRNGKey(7),
            f"EKF-init full-cov twist (T={T}, N={N})  [baseline]")

    # K-sweep with learned twist, at several Newton damping levels.
    # damp_factor=1.0 reproduces the K>=256 divergence (complex eigvals in
    # the learned twist's effective Jacobian); damped runs test the rescue.
    twist_learned = {
        "mean":          twist_mean,
        "chol_log_diag": twist_log_diag,
        "chol_offdiag":  twist_offdiag,
    }
    for damp in (1.0, 0.95, 0.9, 0.8):
        K_sweep(twist_learned, x0s, ys, jr.PRNGKey(7),
                f"learned full-cov twist (T={T}, N={N}, damp={damp})",
                damp_factor=damp)


if __name__ == "__main__":
    main()
