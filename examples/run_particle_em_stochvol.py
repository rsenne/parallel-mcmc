"""
run_particle_em_stochvol.py

Particle EM for the stochastic volatility model with a theta-conditional
normalizing-flow twist.

Model
-----
  x_t = phi * x_{t-1} + sigma_q * eps_t          (log-volatility)
  y_t = beta * exp(x_t / 2) * eta_t              (heavy-tailed obs)

Theta = (phi, log sigma_q, log beta).

Approach
--------
1. Offline meta-training of a theta-conditional NF twist:
     theta -> theta_enc -> GRU(y) hidden h_t -> MLP([x_{t-1}, h_t, theta_enc])
                                              -> (a, b, k) of a monotonic 1-D
                                                 flow x = a + e^b (z + tanh(k) tanh(z))
   trained by NLL on FFBSi-paired (x_{t-1}, x_t) samples drawn at a panel of
   sampled theta values.

2. Particle EM outer loop. The flow is frozen; at each EM iteration the
   E-step queries the amortized flow at theta_k via a single forward pass
   (theta encoder + backward GRU + per-particle MLP), then runs a parallel
   twisted PF (DEER-parallel-in-time, no resampling). Closed-form M-step.

Two baselines for the headline plot:
  * Bootstrap PF only EM           (filter-mean plug-in; biased, fast)
  * Bootstrap PF + FFBSi EM        (gold standard; expensive sequential)
"""

import os
import time
from functools import partial

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

from src.particle_filter import TwistedParallelParticleFilter
from src.baselines import bootstrap_pf, ffbsi_smoother


# Configuration
T = int(os.environ.get("SV_T", 5000))
N = int(os.environ.get("SV_N", 256))
N_EM = int(os.environ.get("SV_N_EM", 60))

# Truth and EM init
PHI_TRUE     = float(os.environ.get("SV_PHI_TRUE",     0.95))
SIGMA_Q_TRUE = float(os.environ.get("SV_SIGMA_Q_TRUE", 0.25))
BETA_TRUE    = float(os.environ.get("SV_BETA_TRUE",    0.65))

PHI_INIT     = float(os.environ.get("SV_PHI_INIT",     0.50))
SIGMA_Q_INIT = float(os.environ.get("SV_SIGMA_Q_INIT", 0.50))
BETA_INIT    = float(os.environ.get("SV_BETA_INIT",    1.20))

# Flow architecture (stacked monotonic-tanh layers + 2-hidden-layer readout)
H_GRU         = int(os.environ.get("SV_H_GRU", 64))
H_MLP         = int(os.environ.get("SV_H_MLP", 128))
N_FLOW_LAYERS = int(os.environ.get("SV_N_FLOW_LAYERS", 2))
NEWTON_ITERS  = 12

# Meta-training panel
N_META       = int(os.environ.get("SV_N_META", 32))
M_FFBSI      = int(os.environ.get("SV_M_FFBSI", 128))
N_FFBSI_PARTS = int(os.environ.get("SV_N_FFBSI_PARTS", 256))
N_TRAIN_STEPS = int(os.environ.get("SV_N_TRAIN_STEPS", 3000))
TRAIN_LR     = float(os.environ.get("SV_TRAIN_LR", 1e-3))
TRAIN_BATCH  = int(os.environ.get("SV_TRAIN_BATCH", 16))

CACHE_PATH = os.environ.get(
    "SV_CACHE_PATH",
    os.path.join(os.path.dirname(__file__), "_em_stochvol_meta.npz"),
)

D = 1


# Stochastic-volatility model + simulation
def sv_transition_mean(x_prev, params):
    return params["phi"] * x_prev


def sv_log_lik(y_t, x_t, params):
    beta = params["beta"]
    var = beta**2 * jnp.exp(x_t)
    return -0.5 * jnp.sum(y_t**2 / var + jnp.log(2.0 * jnp.pi * var))


def simulate_sv(key, T, phi, sigma_q, beta):
    k0, kq, kr = jr.split(key, 3)
    x0 = jr.normal(k0) * 0.5
    eps = jr.normal(kq, (T,))
    eta = jr.normal(kr, (T,))

    def step(x, inputs):
        e_q, e_r = inputs
        xn = phi * x + sigma_q * e_q
        yn = beta * jnp.exp(xn / 2.0) * e_r
        return xn, (xn, yn)

    _, (xs, ys) = jax.lax.scan(step, x0, (eps, eta))
    return xs, ys


# Theta-conditional normalizing flow
def _init_linear(key, d_in, d_out, scale=1.0):
    return {
        "W": jr.normal(key, (d_in, d_out)) * (scale / np.sqrt(max(d_in, 1))),
        "b": jnp.zeros(d_out),
    }


def init_flow_params(key, sigma_q_anchor=0.3):
    """Theta-conditional stacked flow + backward-GRU encoder.

    Readout outputs 3 * N_FLOW_LAYERS params: (a, b, k) for each layer
    of a stacked monotonic-tanh transform.  Two hidden layers in the
    readout MLP (mlp1 -> mlp2 -> mlp_out).
    """
    keys = jr.split(key, 12)
    p = {}

    # Theta encoder: theta in R^3 -> theta_enc in R^{H_GRU}
    p["theta_enc1"] = _init_linear(keys[0], 3, H_MLP, scale=1.0)
    p["theta_enc2"] = _init_linear(keys[1], H_MLP, H_GRU, scale=1.0)

    # Backward GRU on [y_t, theta_enc]
    d_gru_in = 1 + H_GRU
    d_combined = d_gru_in + H_GRU
    p["gru_Wz"] = _init_linear(keys[2], d_combined, H_GRU, scale=0.5)
    p["gru_Wr"] = _init_linear(keys[3], d_combined, H_GRU, scale=0.5)
    p["gru_Wh"] = _init_linear(keys[4], d_combined, H_GRU, scale=0.5)

    # Readout MLP: [x_prev (1), h_t, theta_enc] -> 3 * N_FLOW_LAYERS
    n_out = 3 * N_FLOW_LAYERS
    p["mlp1"]    = _init_linear(keys[5], 1 + 2 * H_GRU, H_MLP, scale=0.5)
    p["mlp2"]    = _init_linear(keys[6], H_MLP, H_MLP, scale=0.5)
    p["mlp_out"] = _init_linear(keys[7], H_MLP, n_out, scale=0.05)

    # Bias init: every layer near identity (a=0, b=0, k=0) except the LAST
    # layer's b = log(sigma_q_anchor) so the initial composed flow is
    # x ≈ sigma_q_anchor * z (a Gaussian proposal at the anchor scale).
    bias = jnp.zeros(n_out)
    last_b_idx = 3 * (N_FLOW_LAYERS - 1) + 1
    bias = bias.at[last_b_idx].set(jnp.log(sigma_q_anchor))
    p["mlp_out"]["b"] = bias
    return p


def linear_(p, x):
    return x @ p["W"] + p["b"]


def encode_theta(net, theta):
    h = jax.nn.relu(linear_(net["theta_enc1"], theta))
    return jnp.tanh(linear_(net["theta_enc2"], h))


def compute_gru_hiddens(net, theta_enc, ys_flat):
    """Backward GRU on [y_t, theta_enc]. Returns (T, H_GRU)."""
    def gru_step(h, y_t):
        inp = jnp.concatenate([jnp.array([y_t]), theta_enc])
        combined = jnp.concatenate([inp, h])
        z = jax.nn.sigmoid(linear_(net["gru_Wz"], combined))
        r = jax.nn.sigmoid(linear_(net["gru_Wr"], combined))
        combined_r = jnp.concatenate([inp, r * h])
        h_tilde = jnp.tanh(linear_(net["gru_Wh"], combined_r))
        return (1.0 - z) * h + z * h_tilde, (1.0 - z) * h + z * h_tilde

    h0 = jnp.zeros(H_GRU)
    _, h_rev = jax.lax.scan(gru_step, h0, ys_flat[::-1])
    return h_rev[::-1]


def flow_readout(net, x_prev_s, h_t, theta_enc):
    """Returns abks: (N_FLOW_LAYERS, 3) — one (a, b, k) triple per layer."""
    inp = jnp.concatenate([jnp.array([x_prev_s]), h_t, theta_enc])
    h1 = jax.nn.tanh(linear_(net["mlp1"], inp))
    h2 = jax.nn.tanh(linear_(net["mlp2"], h1))
    out = linear_(net["mlp_out"], h2).reshape(N_FLOW_LAYERS, 3)
    a = out[:, 0]
    b = _bound_b(out[:, 1])
    k = _bound_k(out[:, 2])
    return jnp.stack([a, b, k], axis=-1)


# Bounds on the raw network outputs to keep the flow well-conditioned:
#   |b_eff| < B_MAX  ⇒  exp(b_eff) ∈ (exp(-B_MAX), exp(B_MAX))
#   |tanh(k_eff)| < tanh(K_MAX) < 1  ⇒  Jacobian 1 + tanh(k)·(1−tanh²z) ∈ (1−tanh K_MAX, 1+tanh K_MAX)
# Both keep the inverse Newton stable and prevent the log-Jacobian from
# diverging to ±∞.
B_MAX = 3.0
K_MAX = 2.5


def _bound_b(b_raw):
    return B_MAX * jnp.tanh(b_raw / B_MAX)


def _bound_k(k_raw):
    return K_MAX * jnp.tanh(k_raw / K_MAX)


def _layer_forward(z, a, b, k):
    return a + jnp.exp(b) * (z + jnp.tanh(k) * jnp.tanh(z))


def _layer_log_jac_dz(z, b, k):
    # sech²(z) = 1 − tanh²(z): same value, but bounded gradient. The raw
    # 1/cosh(z)² overflows for |z| ≳ 89 in float32 and the autodiff path
    # (which produces sinh/cosh³) goes NaN long before that.
    sech2 = 1.0 - jnp.tanh(z) ** 2
    return b + jnp.log(1.0 + jnp.tanh(k) * sech2 + 1e-6)


def _layer_inverse(x, a, b, k, n_iter=NEWTON_ITERS):
    """Newton-iterate to find z such that _layer_forward(z, a, b, k) = x."""
    tk = jnp.tanh(k)
    y = (x - a) / jnp.exp(b)
    # Robust init: when |tk| is small, z ≈ y/(1+tk); when tk → ±1, z ≈ y.
    z = y / jnp.maximum(1.0 + tk, 1e-2)
    for _ in range(n_iter):
        f = z + tk * jnp.tanh(z) - y
        df = 1.0 + tk * (1.0 - jnp.tanh(z) ** 2) + 1e-6
        z = z - f / df
    return z


def flow_forward(z, abks):
    """Apply N_FLOW_LAYERS monotonic-tanh layers in sequence."""
    for i in range(N_FLOW_LAYERS):
        a, b, k = abks[i, 0], abks[i, 1], abks[i, 2]
        z = _layer_forward(z, a, b, k)
    return z


def flow_log_density(x, abks):
    """log q(x | x_prev, h_t, theta_enc) for the stacked flow.

    Inverts each layer in reverse, accumulating -log|dz_{i+1}/dz_i| from the
    layer Jacobian evaluated at z_i.  Final z after inverting all layers is
    a standard normal sample.
    """
    log_q = 0.0
    z = x
    for i in range(N_FLOW_LAYERS - 1, -1, -1):
        a, b, k = abks[i, 0], abks[i, 1], abks[i, 2]
        z_prev = _layer_inverse(z, a, b, k)
        log_q = log_q - _layer_log_jac_dz(z_prev, b, k)
        z = z_prev
    log_phi = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi)
    return log_phi + log_q


# Twisted PF step / weight (theta-conditional stacked flow)
def flow_step_fn(state, driver, params):
    obs, noise, twist_t = driver  # twist_t = {'h': (H_GRU,)}
    x_prev = state[0]
    z = noise[0]
    abks = flow_readout(
        params["flow_net"], x_prev, twist_t["h"], params["theta_enc"]
    )
    return jnp.array([flow_forward(z, abks)])


def flow_log_weight_fn(x_prev, x_next, obs, twist_t, params):
    phi = params["phi"]
    sq = params["sigma_q"]
    beta = params["beta"]
    x_prev_s = x_prev[0]
    x_t = x_next[0]

    abks = flow_readout(
        params["flow_net"], x_prev_s, twist_t["h"], params["theta_enc"]
    )

    log_lik = (
        -0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(beta)
        - x_t / 2.0
        - obs[0] ** 2 / (2.0 * beta**2 * jnp.exp(x_t))
    )
    log_trans = (
        -0.5 * (x_t - phi * x_prev_s) ** 2 / sq**2
        - 0.5 * jnp.log(2.0 * jnp.pi * sq**2)
    )
    log_q = flow_log_density(x_t, abks)
    return log_lik + log_trans - log_q


tpf_flow = TwistedParallelParticleFilter(flow_step_fn, flow_log_weight_fn, D=D)


def pack_theta(phi, sigma_q, beta):
    return jnp.array([phi, jnp.log(sigma_q), jnp.log(beta)])


def build_flow_tape(net, phi, sigma_q, beta, ys_2d, noise):
    """Run theta encoder + GRU once and assemble the twist tape."""
    theta = pack_theta(phi, sigma_q, beta)
    theta_enc = encode_theta(net, theta)
    h_seq = compute_gru_hiddens(net, theta_enc, ys_2d[:, 0])  # (T, H_GRU)
    tape = (ys_2d, noise, {"h": h_seq})
    return tape, theta_enc


# Offline meta-training: sample theta, FFBSi targets, NLL training
def sample_meta_theta(key, n_samples, anchor=(0.85, 0.30, 0.80)):
    """Draw theta panel covering the EM trajectory. Last sample = anchor."""
    k_phi, k_sq, k_b = jr.split(key, 3)
    phis = jr.uniform(k_phi, (n_samples,), minval=0.55, maxval=0.99)
    log_sqs = jr.uniform(k_sq, (n_samples,), minval=jnp.log(0.10),
                         maxval=jnp.log(0.80))
    log_bs = jr.uniform(k_b, (n_samples,), minval=jnp.log(0.30),
                        maxval=jnp.log(1.80))
    sigmas = jnp.exp(log_sqs)
    betas = jnp.exp(log_bs)
    # Pin the EM anchor to be in the panel exactly.
    phis = phis.at[-1].set(anchor[0])
    sigmas = sigmas.at[-1].set(anchor[1])
    betas = betas.at[-1].set(anchor[2])
    return phis, sigmas, betas


def collect_ffbsi_pairs(ys_2d, phi, sigma_q, beta, key, M):
    """Run BPF + FFBSi at one theta. Returns (M, T) trajectories."""
    bpf_key, ffbsi_key, init_key = jr.split(key, 3)
    x0_parts = jr.normal(init_key, (N_FFBSI_PARTS, D)) * 0.5
    bpf_pars = {"phi": phi, "sigma_q": sigma_q, "beta": beta}
    _, bpf_parts, bpf_log_w, _, _ = bootstrap_pf(
        ys_2d, x0_parts, sv_transition_mean, sv_log_lik,
        sigma_q, bpf_pars, bpf_key,
    )
    trajs = ffbsi_smoother(
        bpf_parts, bpf_log_w, sv_transition_mean,
        sigma_q, bpf_pars, ffbsi_key, M=M,
    )
    return trajs[..., 0]  # (M, T)


def build_meta_dataset(key, ys_2d, anchor):
    """Build the (theta_panel, FFBSi_trajectory) dataset."""
    k_panel, k_ffbsi = jr.split(key)
    phis, sigmas, betas = sample_meta_theta(k_panel, N_META, anchor)

    @jax.jit
    def one_sample(phi, sigma_q, beta, sub_key):
        return collect_ffbsi_pairs(ys_2d, phi, sigma_q, beta, sub_key, M_FFBSI)

    keys = jr.split(k_ffbsi, N_META)
    print(f"  Meta panel: {N_META} theta samples, M={M_FFBSI} FFBSi trajs each")
    print(f"  Generating FFBSi targets ...")
    t0 = time.perf_counter()
    trajs_panel = []
    for i in range(N_META):
        ti = one_sample(phis[i], sigmas[i], betas[i], keys[i])
        ti.block_until_ready()
        trajs_panel.append(ti)
        if (i + 1) % max(1, N_META // 8) == 0:
            print(f"    [{i+1:>3}/{N_META}] phi={float(phis[i]):.3f} "
                  f"sigma_q={float(sigmas[i]):.3f} beta={float(betas[i]):.3f}")
    trajs_panel = jnp.stack(trajs_panel, axis=0)  # (N_META, M_FFBSI, T)
    print(f"  FFBSi panel done in {time.perf_counter() - t0:.1f}s")
    return phis, sigmas, betas, trajs_panel


def train_amortized_flow(key, ys_2d, anchor):
    """Offline meta-training: minimise NLL of FFBSi pairs across the theta panel."""
    k_data, k_init = jr.split(key)
    phis, sigmas, betas, trajs_panel = build_meta_dataset(k_data, ys_2d, anchor)

    net = init_flow_params(k_init, sigma_q_anchor=anchor[1])
    # Global-norm clip + adam: per-element clipping cannot bound the total grad
    # norm and let earlier runs explode into the unbounded-b regime even after
    # the flow itself was made stable.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(TRAIN_LR),
    )
    opt_state = optimizer.init(net)

    @jax.jit
    def nll_one_theta(net, phi, sigma_q, beta, trajs):
        """trajs: (M, T). Returns scalar NLL for one theta."""
        theta = pack_theta(phi, sigma_q, beta)
        theta_enc = encode_theta(net, theta)
        h_seq = compute_gru_hiddens(net, theta_enc, ys_2d[:, 0])  # (T, H_GRU)

        def per_pair_logq(x_prev_s, x_t_s, h_t):
            abks = flow_readout(net, x_prev_s, h_t, theta_enc)
            return flow_log_density(x_t_s, abks)

        def per_traj(traj):
            return jax.vmap(per_pair_logq)(traj[:-1], traj[1:], h_seq[1:])

        log_q_all = jax.vmap(per_traj)(trajs)  # (M, T-1)
        return -jnp.mean(log_q_all)

    @jax.jit
    def loss_batch(net, idx, phis, sigmas, betas, trajs_panel):
        def f(i):
            return nll_one_theta(
                net, phis[i], sigmas[i], betas[i], trajs_panel[i]
            )
        return jnp.mean(jax.vmap(f)(idx))

    @jax.jit
    def train_step(net, opt_state, idx, phis, sigmas, betas, trajs_panel):
        loss, grads = jax.value_and_grad(loss_batch)(
            net, idx, phis, sigmas, betas, trajs_panel
        )
        # Drop a non-finite gradient batch entirely rather than feeding NaNs
        # into Adam's running stats and poisoning the rest of training.
        finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(g))
            for g in jax.tree_util.tree_leaves(grads)
        ]))
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(finite, g, jnp.zeros_like(g)), grads
        )
        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = optax.apply_updates(net, updates)
        return net, opt_state, loss

    print(f"\nMeta-training conditional flow ({N_TRAIN_STEPS} steps, "
          f"batch={TRAIN_BATCH})")
    t0 = time.perf_counter()
    losses = []
    for step in range(N_TRAIN_STEPS):
        key, sub = jr.split(key)
        idx = jr.choice(sub, N_META, (TRAIN_BATCH,), replace=False)
        net, opt_state, loss = train_step(
            net, opt_state, idx, phis, sigmas, betas, trajs_panel
        )
        losses.append(float(loss))
        if (step + 1) % 100 == 0:
            print(f"  [{step+1:>5}/{N_TRAIN_STEPS}] NLL = {float(loss):.4f}")
    print(f"  meta-training done in {time.perf_counter() - t0:.1f}s")
    return net, np.array(losses)


# Stochvol M-step (closed form, per-timestep weighted)
def stochvol_m_step(x_prev_path, x_curr_path, ys_flat, w_step):
    """
    Per-timestep importance-weighted M-step.

    x_prev_path, x_curr_path : (N, T)
    ys_flat                  : (T,)
    w_step                   : (N, T) per-timestep softmax weights — each
                               column normalised across particles.  Decoupling
                               the timesteps avoids the path-degeneracy
                               collapse of softmax(sum_t log_w_t) at long T.
    """
    Ex_p_p = jnp.einsum("nt,nt->t", w_step, x_prev_path ** 2)
    Ex_c_c = jnp.einsum("nt,nt->t", w_step, x_curr_path ** 2)
    Ex_cp  = jnp.einsum("nt,nt->t", w_step, x_curr_path * x_prev_path)
    E_emx  = jnp.einsum("nt,nt->t", w_step, jnp.exp(-x_curr_path))

    s_xp = jnp.sum(Ex_cp)
    s_pp = jnp.sum(Ex_p_p) + 1e-8
    s_xx = jnp.sum(Ex_c_c)

    phi_new = s_xp / s_pp
    res = s_xx - 2.0 * phi_new * s_xp + phi_new**2 * s_pp
    sigma_q_sq = jnp.maximum(res / float(x_curr_path.shape[1]), 1e-4)
    sigma_q_new = jnp.sqrt(sigma_q_sq)

    beta_sq = jnp.mean(ys_flat**2 * E_emx)
    beta_new = jnp.sqrt(jnp.maximum(beta_sq, 1e-4))

    # Clamp to reasonable range
    phi_new = jnp.clip(phi_new, -0.999, 0.999)
    sigma_q_new = jnp.clip(sigma_q_new, 0.02, 2.0)
    beta_new = jnp.clip(beta_new, 0.02, 5.0)
    return phi_new, sigma_q_new, beta_new


# EM E-step variants
def em_step_twisted_flow(net, phi, sigma_q, beta, ys_2d, x0s, noise, key):
    """One EM iteration with the theta-conditional flow + parallel twisted PF."""
    tape, theta_enc = build_flow_tape(net, phi, sigma_q, beta, ys_2d, noise)
    pars = {
        "phi": phi, "sigma_q": sigma_q, "beta": beta,
        "flow_net": net, "theta_enc": theta_enc, "key": key,
    }
    states, iters = tpf_flow.run_parallel(x0s, tape, pars, max_iter=100)
    pars_w = {k: v for k, v in pars.items() if k != "key"}
    log_w = tpf_flow.compute_log_weights(x0s, states, tape, pars_w)  # (N, T)

    # Per-timestep softmax — decouples each t from the cumulative log-weight
    # collapse.  When the flow approximates the smoothing marginal, the
    # incremental log-weight log_w[:, t] is small and per-step weights stay
    # near uniform → low-variance M-step estimator.
    w_step = jax.nn.softmax(log_w, axis=0)              # (N, T)
    ess_step = 1.0 / jnp.sum(w_step ** 2, axis=0)        # (T,)
    ess_mean = jnp.mean(ess_step)

    full = jnp.concatenate([x0s[:, None, :], states], axis=1)  # (N, T+1, 1)
    x_prev_path = full[:, :-1, 0]  # (N, T)
    x_curr_path = full[:, 1:, 0]   # (N, T)

    phi_n, sq_n, b_n = stochvol_m_step(x_prev_path, x_curr_path, ys_2d[:, 0], w_step)
    return phi_n, sq_n, b_n, ess_mean, iters[0]


def em_step_bpf(phi, sigma_q, beta, ys_2d, key, N_bpf=N):
    """Bootstrap-PF-only EM iteration (filter-mean plug-in M-step)."""
    init_key, bpf_key = jr.split(key)
    x0_parts = jr.normal(init_key, (N_bpf, D)) * 0.5
    bpf_pars = {"phi": phi, "sigma_q": sigma_q, "beta": beta}
    filt_means, _, _, _, _ = bootstrap_pf(
        ys_2d, x0_parts, sv_transition_mean, sv_log_lik,
        sigma_q, bpf_pars, bpf_key,
    )
    fm = filt_means[:, 0]                                     # (T,)
    x_prev = jnp.concatenate([jnp.array([0.0]), fm[:-1]])     # (T,)
    x_curr = fm
    s_xp = jnp.sum(x_curr * x_prev)
    s_pp = jnp.sum(x_prev**2) + 1e-8
    s_xx = jnp.sum(x_curr**2)
    phi_n = jnp.clip(s_xp / s_pp, -0.999, 0.999)
    res = s_xx - 2.0 * phi_n * s_xp + phi_n**2 * s_pp
    sq_n = jnp.clip(jnp.sqrt(res / float(T)), 0.02, 2.0)
    beta_sq = jnp.mean(ys_2d[:, 0] ** 2 * jnp.exp(-fm))
    b_n = jnp.clip(jnp.sqrt(jnp.maximum(beta_sq, 1e-4)), 0.02, 5.0)
    return phi_n, sq_n, b_n


def em_step_ffbsi(phi, sigma_q, beta, ys_2d, key, M=M_FFBSI, N_bpf=N_FFBSI_PARTS):
    """BPF + FFBSi EM iteration (uniform-weight M-step on smoothed trajectories)."""
    bpf_key, ffbsi_key, init_key = jr.split(key, 3)
    x0_parts = jr.normal(init_key, (N_bpf, D)) * 0.5
    bpf_pars = {"phi": phi, "sigma_q": sigma_q, "beta": beta}
    _, bpf_parts, bpf_log_w, _, _ = bootstrap_pf(
        ys_2d, x0_parts, sv_transition_mean, sv_log_lik,
        sigma_q, bpf_pars, bpf_key,
    )
    trajs = ffbsi_smoother(
        bpf_parts, bpf_log_w, sv_transition_mean,
        sigma_q, bpf_pars, ffbsi_key, M=M,
    )                                                          # (M, T, 1)
    xs = trajs[..., 0]                                         # (M, T)
    T_ = xs.shape[1]
    x_prev = jnp.concatenate([jnp.zeros((M, 1)), xs[:, :-1]], axis=1)
    x_curr = xs
    # Uniform per-timestep weights — FFBSi paths are exact draws from the
    # smoothing posterior, so each contributes equally at every t.
    w_step = jnp.ones((M, T_)) / float(M)
    return stochvol_m_step(x_prev, x_curr, ys_2d[:, 0], w_step)


# Run
def main():
    key = jr.PRNGKey(0)
    print(f"Stochastic volatility:  T={T}, N={N}, N_EM={N_EM}")
    print(f"  truth:    phi*={PHI_TRUE}  sigma_q*={SIGMA_Q_TRUE}  beta*={BETA_TRUE}")
    print(f"  EM init:  phi0={PHI_INIT}  sigma_q0={SIGMA_Q_INIT}  beta0={BETA_INIT}")

    key, sim_key = jr.split(key)
    true_xs, ys = simulate_sv(sim_key, T, PHI_TRUE, SIGMA_Q_TRUE, BETA_TRUE)
    ys_2d = ys[:, None]
    print(f"  observation range: [{float(ys.min()):.3f}, {float(ys.max()):.3f}]"
          f"   std={float(ys.std()):.3f}")

    # Meta-training (cached)
    anchor = (0.85, 0.30, 0.80)
    if os.path.exists(CACHE_PATH):
        print(f"\nLoading cached meta-trained flow from {CACHE_PATH}")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        net = jax.tree_util.tree_map(jnp.asarray, cache["flow_net"].item())
        meta_losses = cache["losses"]
    else:
        key, train_key = jr.split(key)
        net, meta_losses = train_amortized_flow(train_key, ys_2d, anchor)
        np.savez(
            CACHE_PATH,
            flow_net=jax.tree_util.tree_map(np.asarray, net),
            losses=meta_losses,
        )
        print(f"  Cached flow + losses -> {CACHE_PATH}")

    # Particle EM with three E-step variants
    key, x0_key, noise_key = jr.split(key, 3)
    x0s = jr.normal(x0_key, (N, D)) * 0.5
    pf_noise = jr.normal(noise_key, (N, T, D))

    history = {
        "twisted_flow": {"phi": [], "sq": [], "beta": [], "time": [],
                          "ess": [], "deer_iters": []},
        "bpf":          {"phi": [], "sq": [], "beta": [], "time": []},
        "ffbsi":        {"phi": [], "sq": [], "beta": [], "time": []},
    }

    @partial(jax.jit, static_argnums=())
    def jit_twisted_step(net, phi, sigma_q, beta, key):
        return em_step_twisted_flow(net, phi, sigma_q, beta, ys_2d, x0s, pf_noise, key)

    jit_bpf_step = jax.jit(em_step_bpf)
    jit_ffbsi_step = jax.jit(em_step_ffbsi)

    print(f"\n{'='*72}")
    print(f"EM: theta-conditional flow twist + parallel DEER smoother")
    print(f"{'='*72}")
    phi, sq, b = PHI_INIT, SIGMA_Q_INIT, BETA_INIT
    # Warmup compile
    key, k = jr.split(key)
    _ = jit_twisted_step(net, phi, sq, b, k)
    jax.block_until_ready(_[0])
    for it in range(N_EM):
        key, k = jr.split(key)
        t0 = time.perf_counter()
        phi_n, sq_n, b_n, ess, iters = jit_twisted_step(net, phi, sq, b, k)
        jax.block_until_ready(phi_n)
        dt = time.perf_counter() - t0
        history["twisted_flow"]["phi"].append(float(phi_n))
        history["twisted_flow"]["sq"].append(float(sq_n))
        history["twisted_flow"]["beta"].append(float(b_n))
        history["twisted_flow"]["time"].append(dt)
        history["twisted_flow"]["ess"].append(float(ess))
        history["twisted_flow"]["deer_iters"].append(int(iters))
        if (it + 1) % 5 == 0 or it == 0:
            print(f"  [tw-flow] it={it+1:>3}  phi={float(phi_n):.3f}  "
                  f"sq={float(sq_n):.3f}  beta={float(b_n):.3f}  "
                  f"ESS={float(ess):.1f}/{N}  iters={int(iters)}  "
                  f"t={dt:.3f}s")
        phi, sq, b = phi_n, sq_n, b_n

    print(f"\n{'='*72}")
    print(f"EM: bootstrap PF only (filter-mean plug-in baseline)")
    print(f"{'='*72}")
    phi, sq, b = PHI_INIT, SIGMA_Q_INIT, BETA_INIT
    key, k = jr.split(key)
    _ = jit_bpf_step(phi, sq, b, ys_2d, k)
    jax.block_until_ready(_[0])
    for it in range(N_EM):
        key, k = jr.split(key)
        t0 = time.perf_counter()
        phi_n, sq_n, b_n = jit_bpf_step(phi, sq, b, ys_2d, k)
        jax.block_until_ready(phi_n)
        dt = time.perf_counter() - t0
        history["bpf"]["phi"].append(float(phi_n))
        history["bpf"]["sq"].append(float(sq_n))
        history["bpf"]["beta"].append(float(b_n))
        history["bpf"]["time"].append(dt)
        if (it + 1) % 5 == 0 or it == 0:
            print(f"  [bpf]     it={it+1:>3}  phi={float(phi_n):.3f}  "
                  f"sq={float(sq_n):.3f}  beta={float(b_n):.3f}  "
                  f"t={dt:.3f}s")
        phi, sq, b = phi_n, sq_n, b_n

    print(f"\n{'='*72}")
    print(f"EM: bootstrap PF + FFBSi (gold-standard sequential baseline)")
    print(f"{'='*72}")
    phi, sq, b = PHI_INIT, SIGMA_Q_INIT, BETA_INIT
    key, k = jr.split(key)
    _ = jit_ffbsi_step(phi, sq, b, ys_2d, k)
    jax.block_until_ready(_[0])
    for it in range(N_EM):
        key, k = jr.split(key)
        t0 = time.perf_counter()
        phi_n, sq_n, b_n = jit_ffbsi_step(phi, sq, b, ys_2d, k)
        jax.block_until_ready(phi_n)
        dt = time.perf_counter() - t0
        history["ffbsi"]["phi"].append(float(phi_n))
        history["ffbsi"]["sq"].append(float(sq_n))
        history["ffbsi"]["beta"].append(float(b_n))
        history["ffbsi"]["time"].append(dt)
        if (it + 1) % 5 == 0 or it == 0:
            print(f"  [ffbsi]   it={it+1:>3}  phi={float(phi_n):.3f}  "
                  f"sq={float(sq_n):.3f}  beta={float(b_n):.3f}  "
                  f"t={dt:.3f}s")
        phi, sq, b = phi_n, sq_n, b_n

    # Plot
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "figures"))
    import paper_style as ps  # noqa: E402
    ps.apply()

    fig, axes = plt.subplots(2, 3, figsize=(ps.COL_FULL, 5.0),
                             constrained_layout=True)
    iters = np.arange(1, N_EM + 1)

    methods = [
        ("twisted_flow", "twisted-flow EM",     ps.PALETTE["par_alt"]),
        ("ffbsi",        "FFBSi EM (gold)",      ps.PALETTE["ffbsi"]),
        ("bpf",          "bootstrap-PF EM",      ps.PALETTE["bpf"]),
    ]
    truths = [PHI_TRUE, SIGMA_Q_TRUE, BETA_TRUE]
    inits = [PHI_INIT, SIGMA_Q_INIT, BETA_INIT]
    keys = ["phi", "sq", "beta"]
    titles = [r"$\phi$", r"$\sigma_q$", r"$\beta$"]

    for col, (k_, ttl, truth, init) in enumerate(zip(keys, titles, truths, inits)):
        ax = axes[0, col]
        ax.axhline(truth, color="k", lw=0.8, ls="--",
                   label=f"truth = {truth:.2f}")
        for mk, mlab, mc in methods:
            arr = np.array([init] + history[mk][k_])
            ax.plot(np.arange(len(arr)), arr, color=mc, lw=1.1,
                    label=mlab, marker="o", ms=2.5)
        ax.set_xlabel("EM iteration")
        ax.set_ylabel(ttl)
        ax.set_title(f"{ttl} recovery")
        if col == 0:
            ax.legend(fontsize=6.5, loc="best", framealpha=0.9)

    # Wall-clock per EM iter
    ax = axes[1, 0]
    for mk, mlab, mc in methods:
        ax.plot(iters, history[mk]["time"], color=mc, lw=1.1, marker="o",
                ms=2.5, label=mlab)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("time / iter (s)")
    ax.set_yscale("log")
    ax.set_title("Wall-clock per EM iteration")
    ax.legend(fontsize=6.0, loc="best", framealpha=0.9)

    # ESS for the flow E-step
    ax = axes[1, 1]
    ax.plot(iters, history["twisted_flow"]["ess"],
            color=ps.PALETTE["par_alt"], lw=1.1, marker="o", ms=2.5,
            label="path ESS")
    ax.axhline(N, color="k", lw=0.6, ls=":", label=f"N = {N}")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("path ESS")
    ax.set_title("Twisted-flow path ESS over EM")
    ax.legend(fontsize=6.5)

    # Meta-training NLL
    ax = axes[1, 2]
    ax.plot(np.arange(len(meta_losses)), meta_losses,
            color=ps.PALETTE["par_alt"], lw=1.0)
    ax.set_xlabel("Meta-train step")
    ax.set_ylabel("NLL")
    ax.set_title("Offline meta-training")

    cum_tw = np.sum(history["twisted_flow"]["time"])
    cum_ff = np.sum(history["ffbsi"]["time"])
    speedup = cum_ff / cum_tw if cum_tw > 0 else float("nan")
    fig.suptitle(
        f"Particle EM with $\\theta$-conditional flow twist  "
        f"(stochvol, $T{{=}}{T}$, $N{{=}}{N}$).  "
        f"End-to-end speedup vs FFBSi-EM: $\\mathbf{{{speedup:.1f}\\times}}$",
        fontsize=10, y=1.03,
    )
    out_png = "particle_em_stochvol.png"
    out_svg = "particle_em_stochvol.svg"
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.close(fig)
    print(f"\nSaved {out_png} and {out_svg}")
    print(f"End-to-end speedup vs FFBSi EM: {speedup:.2f}x  "
          f"(twisted-flow {cum_tw:.1f}s vs FFBSi {cum_ff:.1f}s)")


if __name__ == "__main__":
    main()
