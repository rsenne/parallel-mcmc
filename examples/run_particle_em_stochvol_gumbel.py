"""
run_particle_em_stochvol_gumbel.py

Particle EM for stochastic volatility with the *windowed Gumbel-STE* twisted
PF as the E-step, compared head-to-head against the no-resampling twisted
smoother E-step (run_particle_em_stochvol.py), bootstrap-PF plug-in, and
BPF+FFBSi gold standard.

Why the Gumbel E-step
---------------------
The no-resampling twisted smoother weights whole trajectories by importance
sampling: with an imperfect twist the weights degenerate as T grows, which
run_particle_em_stochvol.py works around with a per-timestep softmax hack.
Windowed Gumbel-STE resampling interpolates between the two failure modes:
resampling every K steps bounds weight degeneracy (weights reset at each
boundary) while keeping ancestral paths diverse (only T/K resample events),
and its log_Z is the standard unbiased SMC estimate.

E-step estimator: trajectories within window w are weighted by the softmax
(over particles) of the summed within-window log-weights — the SMC
filter-smoother restricted to a window, i.e. fixed-lag smoothing with lag
up to K. Cross-boundary transition pairs (x_{K-1}, x_K) are recovered from
the post-resampling window-start particles, which are exact parent values.

Reuses the theta-conditional flow twist, model, M-step, and baselines from
run_particle_em_stochvol.py (imported as a module; its meta-flow cache is
shared).
"""

import os
import sys
import time

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_particle_em_stochvol as base  # noqa: E402

from src.gumbel_particle_filter import TwistedGumbelParticleFilter  # noqa: E402

T = base.T
N = base.N
N_EM = base.N_EM
D = base.D

K = int(os.environ.get("SV_K", 250))
TAU = float(os.environ.get("SV_TAU", 1.0))
MAX_ITER = int(os.environ.get("SV_MAX_ITER", 100))
DAMP = float(os.environ.get("SV_DAMP", 1.0))
# "window": within-window filter-smoother weights (principled).
# "step":   per-timestep incremental softmax (matches the old script's hack).
W_MODE = os.environ.get("SV_W_MODE", "window")

tgpf_flow = TwistedGumbelParticleFilter(
    base.flow_step_fn, base.flow_log_weight_fn, D=D, tau=TAU
)


def em_step_twisted_gumbel(net, phi, sigma_q, beta, ys_2d, x0s, tape_key,
                           deer_key):
    """One EM iteration: windowed Gumbel-STE twisted PF E-step.

    tape_key is held fixed across EM iterations (fixed transition noise and
    Gumbel draws, mirroring the fixed pf_noise of the no-resampling script);
    deer_key varies per iteration (Hutchinson probes only).
    """
    theta = base.pack_theta(phi, sigma_q, beta)
    theta_enc = base.encode_theta(net, theta)
    h_seq = base.compute_gru_hiddens(net, theta_enc, ys_2d[:, 0])  # (T, H)

    tape = tgpf_flow.make_tape(ys_2d, {"h": h_seq}, tape_key, N=N, K=K)
    pars = {
        "phi": phi, "sigma_q": sigma_q, "beta": beta,
        "flow_net": net, "theta_enc": theta_enc, "key": deer_key,
    }
    states, iters, log_Z, ess_w, log_ws, win_x0s = tgpf_flow.run_parallel(
        x0s, tape, pars, max_iter=MAX_ITER, damp_factor=DAMP,
        return_weights=True,
    )
    # states: (N, T, D), log_ws: (N, T), win_x0s: (num_windows, N, D)

    num_windows = T // K
    if W_MODE == "window":
        # Filter-smoother weights per window: one weight per trajectory
        # segment, from the summed within-window log-weights.
        lw_win = log_ws.reshape(N, num_windows, K).sum(axis=-1)     # (N, W)
        w_win = jax.nn.softmax(lw_win, axis=0)                      # (N, W)
        w_step = jnp.repeat(w_win, K, axis=1)                       # (N, T)
    else:
        w_step = jax.nn.softmax(log_ws, axis=0)                     # (N, T)
    ess_step = 1.0 / jnp.sum(w_step**2, axis=0)                     # (T,)
    ess_mean = jnp.mean(ess_step)

    # Transition pairs. Within a window x_prev is the previous state; at the
    # first step of each window it is the post-resampling window-start
    # particle (the exact parent value).
    states_w = states.reshape(N, num_windows, K, D)
    x0_w = jnp.transpose(win_x0s, (1, 0, 2))[:, :, None, :]         # (N, W, 1, D)
    x_prev_w = jnp.concatenate([x0_w, states_w[:, :, :-1, :]], axis=2)
    x_prev_path = x_prev_w.reshape(N, T)
    x_curr_path = states[:, :, 0]                                   # (N, T)

    phi_n, sq_n, b_n = base.stochvol_m_step(
        x_prev_path, x_curr_path, ys_2d[:, 0], w_step
    )
    return phi_n, sq_n, b_n, ess_mean, jnp.max(iters), log_Z


def run_em_loop(label, step_fn, key, history):
    # jnp scalars from the start so the warmup trace matches the loop
    # (passing Python floats first and traced outputs later recompiles).
    phi = jnp.float32(base.PHI_INIT)
    sq = jnp.float32(base.SIGMA_Q_INIT)
    b = jnp.float32(base.BETA_INIT)
    key, k = jr.split(key)
    _ = step_fn(phi, sq, b, k)
    jax.block_until_ready(_[0])
    for it in range(N_EM):
        key, k = jr.split(key)
        t0 = time.perf_counter()
        out = step_fn(phi, sq, b, k)
        jax.block_until_ready(out[0])
        dt = time.perf_counter() - t0
        phi_n, sq_n, b_n = out[0], out[1], out[2]
        history["phi"].append(float(phi_n))
        history["sq"].append(float(sq_n))
        history["beta"].append(float(b_n))
        history["time"].append(dt)
        if len(out) > 3:
            history.setdefault("ess", []).append(float(out[3]))
            history.setdefault("deer_iters", []).append(int(out[4]))
        if len(out) > 5:
            history.setdefault("log_Z", []).append(float(out[5]))
        if (it + 1) % 5 == 0 or it == 0:
            extra = ""
            if len(out) > 3:
                extra = f"  ESS={float(out[3]):.1f}/{N}  iters={int(out[4])}"
            if len(out) > 5:
                extra += f"  logZ={float(out[5]):.1f}"
            print(f"  [{label}] it={it+1:>3}  phi={float(phi_n):.3f}  "
                  f"sq={float(sq_n):.3f}  beta={float(b_n):.3f}{extra}  "
                  f"t={dt:.3f}s")
        phi, sq, b = phi_n, sq_n, b_n
    return key


def main():
    key = jr.PRNGKey(0)
    print(f"Stochvol Gumbel-EM:  T={T}, N={N}, K={K} ({T//K} windows), "
          f"N_EM={N_EM}, w_mode={W_MODE}, damp={DAMP}")
    print(f"  truth:    phi*={base.PHI_TRUE}  sigma_q*={base.SIGMA_Q_TRUE}  "
          f"beta*={base.BETA_TRUE}")
    print(f"  EM init:  phi0={base.PHI_INIT}  sigma_q0={base.SIGMA_Q_INIT}  "
          f"beta0={base.BETA_INIT}")

    key, sim_key = jr.split(key)
    true_xs, ys = base.simulate_sv(
        sim_key, T, base.PHI_TRUE, base.SIGMA_Q_TRUE, base.BETA_TRUE
    )
    ys_2d = ys[:, None]

    if not os.path.exists(base.CACHE_PATH):
        raise SystemExit(
            f"Meta-flow cache not found at {base.CACHE_PATH}; run "
            f"run_particle_em_stochvol.py first to train it."
        )
    print(f"\nLoading cached meta-trained flow from {base.CACHE_PATH}")
    cache = np.load(base.CACHE_PATH, allow_pickle=True)
    net = jax.tree_util.tree_map(jnp.asarray, cache["flow_net"].item())

    key, x0_key, noise_key, tape_key = jr.split(key, 4)
    x0s = jr.normal(x0_key, (N, D)) * 0.5
    pf_noise = jr.normal(noise_key, (N, T, D))   # for the no-resample E-step

    history = {m: {"phi": [], "sq": [], "beta": [], "time": []}
               for m in ["gumbel", "twisted_flow", "bpf", "ffbsi"]}

    jit_gumbel = jax.jit(
        lambda phi, sq, b, k: em_step_twisted_gumbel(
            net, phi, sq, b, ys_2d, x0s, tape_key, k)
    )
    jit_flow = jax.jit(
        lambda phi, sq, b, k: base.em_step_twisted_flow(
            net, phi, sq, b, ys_2d, x0s, pf_noise, k)
    )
    jit_bpf = jax.jit(
        lambda phi, sq, b, k: base.em_step_bpf(phi, sq, b, ys_2d, k)
    )
    jit_ffbsi = jax.jit(
        lambda phi, sq, b, k: base.em_step_ffbsi(phi, sq, b, ys_2d, k)
    )

    methods_env = os.environ.get(
        "SV_METHODS", "gumbel,twisted_flow,bpf,ffbsi").split(",")
    all_steps = {"gumbel": jit_gumbel, "twisted_flow": jit_flow,
                 "bpf": jit_bpf, "ffbsi": jit_ffbsi}
    for label in methods_env:
        print(f"\n{'='*72}\nEM: {label}\n{'='*72}")
        key = run_em_loop(label, all_steps[label], key, history[label])

    # Summary table
    print(f"\n{'='*84}")
    print(f"{'method':>14}  {'phi':>7}  {'sigma_q':>8}  {'beta':>7}  "
          f"{'t/iter (s)':>11}  {'total (s)':>10}")
    print("-" * 84)
    print(f"{'truth':>14}  {base.PHI_TRUE:>7.3f}  {base.SIGMA_Q_TRUE:>8.3f}  "
          f"{base.BETA_TRUE:>7.3f}")
    for m in ["gumbel", "twisted_flow", "bpf", "ffbsi"]:
        h = history[m]
        if not h["phi"]:
            continue
        print(f"{m:>14}  {h['phi'][-1]:>7.3f}  {h['sq'][-1]:>8.3f}  "
              f"{h['beta'][-1]:>7.3f}  {np.mean(h['time']):>11.3f}  "
              f"{np.sum(h['time']):>10.1f}")
    if "ess" in history["gumbel"]:
        print(f"\n  gumbel   final ESS={history['gumbel']['ess'][-1]:.1f}/{N}"
              f"  max DEER iters={max(history['gumbel']['deer_iters'])}")

    if not all(history[m]["phi"] for m in ["gumbel", "twisted_flow",
                                           "bpf", "ffbsi"]):
        print("\n(some methods skipped via SV_METHODS — no plot)")
        return

    # Plot
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "figures"))
    import paper_style as ps  # noqa: E402
    ps.apply()

    fig, axes = plt.subplots(2, 3, figsize=(ps.COL_FULL, 5.0),
                             constrained_layout=True)
    methods = [
        ("gumbel",       "Gumbel-window EM",  ps.PALETTE["par"]),
        ("twisted_flow", "no-resample EM",    ps.PALETTE["par_alt"]),
        ("ffbsi",        "FFBSi EM (gold)",   ps.PALETTE["ffbsi"]),
        ("bpf",          "bootstrap-PF EM",   ps.PALETTE["bpf"]),
    ]
    truths = [base.PHI_TRUE, base.SIGMA_Q_TRUE, base.BETA_TRUE]
    inits = [base.PHI_INIT, base.SIGMA_Q_INIT, base.BETA_INIT]
    param_keys = ["phi", "sq", "beta"]
    titles = [r"$\phi$", r"$\sigma_q$", r"$\beta$"]

    for col, (k_, ttl, truth, init) in enumerate(
            zip(param_keys, titles, truths, inits)):
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
            ax.legend(fontsize=6.0, loc="best", framealpha=0.9)

    it_axis = np.arange(1, N_EM + 1)
    ax = axes[1, 0]
    for mk, mlab, mc in methods:
        ax.plot(it_axis, history[mk]["time"], color=mc, lw=1.1, marker="o",
                ms=2.5, label=mlab)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("time / iter (s)")
    ax.set_yscale("log")
    ax.set_title("Wall-clock per EM iteration")
    ax.legend(fontsize=6.0, loc="best", framealpha=0.9)

    ax = axes[1, 1]
    for mk, mlab, mc in [("gumbel", "Gumbel-window", ps.PALETTE["par"]),
                         ("twisted_flow", "no-resample", ps.PALETTE["par_alt"])]:
        if "ess" in history[mk]:
            ax.plot(it_axis, history[mk]["ess"], color=mc, lw=1.1,
                    marker="o", ms=2.5, label=mlab)
    ax.axhline(N, color="k", lw=0.6, ls=":", label=f"N = {N}")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("mean ESS")
    ax.set_title("E-step ESS over EM")
    ax.legend(fontsize=6.5)

    ax = axes[1, 2]
    if "log_Z" in history["gumbel"]:
        ax.plot(it_axis, history["gumbel"]["log_Z"],
                color=ps.PALETTE["par"], lw=1.1, marker="o", ms=2.5)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel(r"$\log \hat{Z}(\theta_k)$")
    ax.set_title("Unbiased marginal-lik. estimate (Gumbel)")

    cum_g = np.sum(history["gumbel"]["time"])
    cum_ff = np.sum(history["ffbsi"]["time"])
    fig.suptitle(
        f"Particle EM: windowed Gumbel-STE twisted PF E-step  "
        f"(stochvol, $T{{=}}{T}$, $N{{=}}{N}$, $K{{=}}{K}$).  "
        f"vs FFBSi-EM: $\\mathbf{{{cum_ff/cum_g:.1f}\\times}}$",
        fontsize=10, y=1.03,
    )
    out = "particle_em_stochvol_gumbel"
    plt.savefig(out + ".png")
    plt.savefig(out + ".svg")
    plt.close(fig)
    print(f"\nSaved {out}.png/.svg")
    print(f"End-to-end vs FFBSi EM: {cum_ff/cum_g:.2f}x  "
          f"(gumbel {cum_g:.1f}s vs FFBSi {cum_ff:.1f}s)")


if __name__ == "__main__":
    main()
