"""
gumbel_particle_filter.py

Parallel-in-time particle filter with hard-forward / soft-Jacobian
Gumbel resampling, parallelised across time via quasi-DEER.

Key idea
--------
A standard particle filter is non-differentiable because the multinomial
resampling step makes ancestor indices discrete:

    a_t^i ~ Categorical(softmax(log_w_{t-1}^{1:N})).

We sample those ancestors exactly via the Gumbel-max trick:

    a_t^i = argmax_j ( log_w^j + g_{t,i,j} ),    g_{t,i,j} ~ Gumbel(0,1).

For the Jacobian used by DEER we replace the hard argmax with a
straight-through Gumbel-Softmax:

    soft_{i,j} = softmax( (log_w + g_i) / tau )_j                  (N, N)
    ste        = soft + stop_grad( one_hot(argmax) - soft )         (N, N)
    X_new      = X_pred @ ste.T                                     (D, N)

In the forward pass ``ste`` is exactly the hard one-hot ancestor matrix,
so the recursion remains an ordinary particle filter with the same
randomness. In the backward / JVP pass the Jacobian flows through
``soft`` so DEER receives a usable local linearisation through
resampling.

The marginal-likelihood estimator therefore retains the usual finite-N
unbiasedness whenever the DEER fixed-point solve converges to the
sequential trajectory.

Shapes / conventions
--------------------
  D — latent state dimension
  N — number of particles
  T — number of timesteps

The "state" carried by DEER at each timestep is the (D*N,) flattened
post-resampling particle cloud. Pre-drawn randomness per step:

  - transition noise  : (D, N)
  - Gumbel(0,1) noise : (N, N)   one Gumbel vector per child particle

so the full tape is (observations, noise, gumbel) with leading dim T.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from src import deer, qdeer


def sample_gumbel(key: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Draw i.i.d. Gumbel(0,1) noise via inverse-CDF of Uniform(eps, 1).

    A small epsilon avoids -inf when the uniform sample is exactly 0.
    """
    eps = jnp.finfo(jnp.float32).tiny
    u = jr.uniform(key, shape, minval=eps, maxval=1.0)
    return -jnp.log(-jnp.log(u))


def gumbel_ste_resample(
    X: jnp.ndarray,
    log_w: jnp.ndarray,
    gumbel: jnp.ndarray,
    tau: float,
) -> jnp.ndarray:
    """Hard-forward, soft-Jacobian Gumbel-Softmax resampling.

    Parameters
    ----------
    X      : (D, N) particle matrix (columns are particles).
    log_w  : (N,)   unnormalised log-weights for the source particles.
    gumbel : (N, N) pre-drawn Gumbel(0,1) noise — row i is the noise for
             the i-th child particle (one categorical draw per child).
    tau    : softmax temperature for the surrogate Jacobian. Smaller =
             closer to the hard one-hot (sharper, but stiffer gradients);
             larger = smoother gradients but worse local linearisation.

    Returns
    -------
    X_new : (D, N) resampled particles. Equal in value to ``X[:, a]``
            where ``a = argmax_j (log_w + g_i)`` (exact categorical draw),
            but differentiates as ``X @ soft.T``.
    """
    log_a = log_w - jax.scipy.special.logsumexp(log_w)
    logits = log_a[None, :] + gumbel                          # (N, N)
    hard_idx = jnp.argmax(logits, axis=-1)                    # (N,)
    N = log_w.shape[0]
    hard_oh = jax.nn.one_hot(hard_idx, N, dtype=X.dtype)      # (N, N)
    soft = jax.nn.softmax(logits / tau, axis=-1)              # (N, N)
    ste = soft + jax.lax.stop_gradient(hard_oh - soft)        # (N, N)
    return X @ ste.T                                           # (D, N)


class GumbelParallelParticleFilter:
    """Particle filter parallelised in time via quasi-DEER with
    hard-forward Gumbel-max resampling.

    The forward pass is an exact bootstrap particle filter using the
    pre-drawn Gumbel noise to sample ancestors. The Jacobian used by
    DEER flows through a Gumbel-Softmax STE relaxation, giving a
    well-defined local linearisation through the resampling step.

    Parameters
    ----------
    transition_fn : (X: (D,N), noise: (D,N), params) -> X_pred: (D,N)
        Reparameterised transition. Must be differentiable in X.
    log_lik_fn    : (X_pred: (D,N), obs, params) -> log_w: (N,)
        Per-particle log-likelihood of the observation.
    D, N          : latent dimension and particle count.
    tau           : Gumbel-Softmax temperature for the JVP surrogate
                    (default 0.5). Only affects the DEER Jacobian, not
                    the forward trajectory.
    """

    def __init__(
        self,
        transition_fn: Callable,
        log_lik_fn: Callable,
        D: int,
        N: int,
        tau: float = 0.5,
    ):
        self.transition_fn = transition_fn
        self.log_lik_fn = log_lik_fn
        self.D = D
        self.N = N
        self.tau = tau

    def make_tape(
        self,
        observations: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Pre-draw all per-step randomness for the full horizon.

        Returns
        -------
        tape = (observations, noise, gumbel)
            observations : (T, *obs_shape)
            noise        : (T, D, N)   — transition noise
            gumbel       : (T, N, N)   — Gumbel(0,1) for resampling
        """
        T = observations.shape[0]
        k_noise, k_gumbel = jr.split(key)
        noise = jr.normal(k_noise, (T, self.D, self.N))
        gumbel = sample_gumbel(k_gumbel, (T, self.N, self.N))
        return (observations, noise, gumbel)

    def pf_step(
        self,
        state: jnp.ndarray,
        driver: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        params: dict,
    ) -> jnp.ndarray:
        """One particle-filter step in DEER form.

        Signature matches qdeer.seq1d: (state, driver, params) -> state.

        state  : (D*N,) flattened post-resample particles at t-1.
        driver : (obs_t, noise_t (D,N), gumbel_t (N,N))
        """
        obs, noise, gumbel = driver
        X = state.reshape(self.D, self.N)
        X_pred = self.transition_fn(X, noise, params)         # (D, N)
        log_w = self.log_lik_fn(X_pred, obs, params)          # (N,)
        tau = params.get("tau", self.tau)
        X_new = gumbel_ste_resample(X_pred, log_w, gumbel, tau)
        return X_new.ravel()

    def run_sequential(
        self,
        s0: jnp.ndarray,
        tape: Tuple,
        params: dict,
    ) -> jnp.ndarray:
        """Sequential bootstrap PF via jax.lax.scan. Returns states (T, D*N)."""
        def _scan(state, driver):
            next_state = self.pf_step(state, driver, params)
            return next_state, next_state
        _, states = jax.lax.scan(_scan, s0, tape)
        return states

    def run_parallel(
        self,
        s0: jnp.ndarray,
        tape: Tuple,
        params: dict,
        yinit_guess: Optional[jnp.ndarray] = None,
        max_iter: int = 100,
        solver: str = "full",
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Parallel-in-time PF via DEER.

        Resampling couples all particles, so the per-step Jacobian is
        dense. The default ``solver='full'`` uses full (D*N, D*N) DEER
        — fine when D*N is modest (≲ a few hundred). For larger systems
        use ``'quasi'`` (Hutchinson-diagonal) or ``'rank1'`` (Hutchinson
        rank-1) at the cost of slower convergence.

        Parameters
        ----------
        s0          : (D*N,) initial flattened particles.
        tape        : output of make_tape.
        params      : dict, must contain 'key' (used by quasi/rank1
                      Hutchinson estimators; harmless for 'full').
        yinit_guess : (T, D*N) DEER warm-start; defaults to zeros.
        max_iter    : maximum DEER Newton iterations.
        solver      : 'full' | 'quasi' | 'rank1'.
        **kwargs    : forwarded to the underlying seq1d call (e.g.
                      clip_val, damp_factor, full_trace, num_samples).

        Returns
        -------
        states, samp_iters
            states     : (T, D*N) — converged DEER trajectory.
            samp_iters : scalar — number of DEER iterations taken.
        """
        if solver == "full":
            states, samp_iters = deer.seq1d(
                self.pf_step,
                s0,
                tape,
                params,
                yinit_guess=yinit_guess,
                max_iter=max_iter,
                quasi=False,
                **kwargs,
            )
        elif solver == "quasi":
            states, samp_iters = qdeer.seq1d(
                self.pf_step,
                s0,
                tape,
                params,
                yinit_guess=yinit_guess,
                max_iter=max_iter,
                **kwargs,
            )
        elif solver == "rank1":
            states, samp_iters = qdeer.seq1d_rank1(
                self.pf_step,
                s0,
                tape,
                params,
                yinit_guess=yinit_guess,
                max_iter=max_iter,
                **kwargs,
            )
        else:
            raise ValueError(
                f"unknown solver={solver!r}; expected 'full', 'quasi', or 'rank1'"
            )
        return states, samp_iters

    def log_marginal_likelihood(
        self,
        s0: jnp.ndarray,
        states: jnp.ndarray,
        tape: Tuple,
        params: dict,
    ) -> jnp.ndarray:
        """Standard unbiased log-marginal-likelihood estimator.

        log Z_T ≈ sum_t  log( (1/N) sum_i exp(log_w_t^i) )

        where log_w_t^i is the incremental log-weight at time t computed
        from the *pre-resample* predicted particles X_pred_t.

        ``states`` is the full trajectory of post-resample particles
        produced by run_sequential or run_parallel (T, D*N).
        """
        observations, noise, _ = tape
        T = observations.shape[0]
        N = self.N

        prev_states = jnp.concatenate([s0[None, :], states[:-1]], axis=0)  # (T, D*N)

        def step_logZ(prev_state, obs_t, noise_t):
            X_prev = prev_state.reshape(self.D, N)
            X_pred = self.transition_fn(X_prev, noise_t, params)
            log_w = self.log_lik_fn(X_pred, obs_t, params)
            return jax.scipy.special.logsumexp(log_w) - jnp.log(float(N))

        per_step = jax.vmap(step_logZ)(prev_states, observations, noise)  # (T,)
        return jnp.sum(per_step)


class TwistedGumbelParticleFilter:
    """Twisted PF with windowed Gumbel-STE resampling, parallelised
    per-window via DEER.

    The motivation is structural. A standard PF resamples every step,
    making the genealogy a tightly coupled chain that the DEER soft
    Jacobian linearises only one flip at a time (iters scale ~linearly
    with T — see logs/gumbel_pf_scaling.log).

    With a good twist, each particle's trajectory is a smooth
    deterministic chain (the twisted_step_fn), so DEER can solve K
    steps per particle in O(log K) parallel depth with O(1) iters when
    the chain is well-conditioned. Resampling is restricted to fixed
    window boundaries every K steps, where one Gumbel-STE op splices
    particles across N — the only place the dense, ancestor-coupling
    Jacobian appears.

    For T = num_windows * K, total work is roughly:
        O(num_windows * (deer_iters * log K + N^2))   vs.  O(T)
    so the speedup depends on (a) K large enough that ancestor flips
    are rare (twists keep ESS high) and (b) num_windows small enough
    that the serial outer loop is cheap.

    Parameters
    ----------
    twisted_step_fn : (state: (D,), driver, params) -> next_state: (D,)
        Deterministic twisted transition for ONE particle. driver =
        (obs_t, noise_t, twist_params_t). Same convention as
        TwistedParallelParticleFilter.
    log_weight_fn : (x_prev: (D,), x_next: (D,), obs_t, twist_t, params) -> scalar
        Incremental log importance weight at one timestep.
    D : int
        Latent dimension per particle.
    tau : float
        Gumbel-Softmax temperature for the DEER Jacobian surrogate at
        resampling checkpoints. tau=1.0 worked well for the dense-PF
        case; near-uniform weights here mean even larger tau is fine.
    """

    def __init__(
        self,
        twisted_step_fn: Callable,
        log_weight_fn: Callable,
        D: int,
        tau: float = 1.0,
    ):
        self.twisted_step_fn = twisted_step_fn
        self.log_weight_fn = log_weight_fn
        self.D = D
        self.tau = tau

    def make_tape(
        self,
        observations: jnp.ndarray,
        twist_params: Any,
        key: jnp.ndarray,
        N: int,
        K: int,
    ) -> Tuple[Any, jnp.ndarray, Any, jnp.ndarray, int, int]:
        """Pre-draw all per-step randomness, reshaped into windows of size K.

        Parameters
        ----------
        observations : (T, *obs_shape)
        twist_params : pytree with leading dim T
        key          : JAX PRNG key
        N            : number of particles
        K            : window size. T must be divisible by K.

        Returns
        -------
        tape = (obs_w, noise_w, twist_w, gumbel_w)
            obs_w    : (num_windows, K, *obs_shape)
            noise_w  : (N, num_windows, K, D)   -- particle-major
            twist_w  : pytree with leading dims (num_windows, K, ...)
            gumbel_w : (num_windows, N, N)      -- one Gumbel matrix per window boundary

        N and K are recoverable from array shapes inside _run, so they
        are not stored in the tape (which lets the tape pass cleanly
        through jax.jit without static-shape tracer issues).
        """
        T = observations.shape[0]
        if T % K != 0:
            raise ValueError(f"T={T} must be divisible by K={K}")
        num_windows = T // K

        k_noise, k_gumbel = jr.split(key)
        noise = jr.normal(k_noise, (N, T, self.D))                    # (N, T, D)
        gumbel = sample_gumbel(k_gumbel, (num_windows, N, N))         # one per boundary

        obs_w = observations.reshape((num_windows, K) + observations.shape[1:])
        noise_w = noise.reshape(N, num_windows, K, self.D)
        twist_w = jax.tree_util.tree_map(
            lambda x: x.reshape((num_windows, K) + x.shape[1:]), twist_params
        )
        return (obs_w, noise_w, twist_w, gumbel)

    def _window_log_weights(
        self,
        x0: jnp.ndarray,
        states: jnp.ndarray,
        win_obs: jnp.ndarray,
        win_twist: Any,
        params: dict,
    ) -> jnp.ndarray:
        """Per-particle, per-step log weights within one window.

        x0     : (D,)   particle state at the window start (post-resample).
        states : (K, D) particle trajectory within the window.
        Returns (K,) log weights.
        """
        x_prevs = jnp.concatenate([x0[None, :], states[:-1]], axis=0)

        def step_w(x_prev, x_next, obs_t, twist_t):
            return self.log_weight_fn(x_prev, x_next, obs_t, twist_t, params)

        return jax.vmap(step_w)(x_prevs, states, win_obs, win_twist)

    def _run_one_particle_window(
        self,
        x0: jnp.ndarray,
        win_obs: jnp.ndarray,
        win_noise: jnp.ndarray,
        win_twist: Any,
        params: dict,
        mode: str,
        max_iter: int,
        deer_key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Solve one particle's K-step twisted chain. Returns (states (K,D), iters)."""
        particle_tape = (win_obs, win_noise, win_twist)

        if mode == "sequential":
            def scan_fn(state, driver):
                next_state = self.twisted_step_fn(state, driver, params)
                return next_state, next_state
            _, states = jax.lax.scan(scan_fn, x0, particle_tape)
            return states, jnp.array(0, dtype=jnp.int32)

        elif mode == "parallel":
            params_i = {**params, "key": deer_key}
            return qdeer.seq1d(
                self.twisted_step_fn, x0, particle_tape, params_i,
                max_iter=max_iter,
            )

        else:
            raise ValueError(f"unknown mode={mode!r}")

    def _run(
        self,
        x0s: jnp.ndarray,
        tape: Tuple,
        params: dict,
        mode: str,
        max_iter: int = 100,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Common windowed driver — sequential or parallel-per-particle."""
        obs_w, noise_w, twist_w, gumbel_w = tape
        num_windows, K = obs_w.shape[0], obs_w.shape[1]
        N = noise_w.shape[0]

        # Per-window DEER keys (only used when mode == 'parallel').
        deer_master = params.get("key", jr.PRNGKey(0))
        deer_keys_outer = jr.split(deer_master, num_windows)
        # For each window we also need per-particle keys.
        params_clean = {k: v for k, v in params.items() if k != "key"}

        def window_step(carry, scan_inputs):
            particles, log_w_acc = carry              # (N, D), (N,)
            win_idx, win_obs, win_twist, win_gumbel, deer_key = scan_inputs

            win_noise_all = noise_w[:, win_idx]       # (N, K, D)

            per_particle_keys = jr.split(deer_key, N)

            def solve_one(x0, noise_i, key_i):
                return self._run_one_particle_window(
                    x0, win_obs, noise_i, win_twist, params_clean,
                    mode=mode, max_iter=max_iter, deer_key=key_i,
                )

            states_w, iters = jax.vmap(solve_one)(particles, win_noise_all,
                                                   per_particle_keys)
            # states_w: (N, K, D)

            # Accumulate log weights across the window.
            def weights_one(x0, states):
                return self._window_log_weights(
                    x0, states, win_obs, win_twist, params_clean
                )
            log_ws_w = jax.vmap(weights_one)(particles, states_w)   # (N, K)
            log_w_window_total = log_w_acc + jnp.sum(log_ws_w, axis=1)

            # log Z increment for this window:
            #   log( (1/N) sum_i exp( log_w_acc_i + sum_t log_w_t_i ) )
            log_Z_inc = (jax.scipy.special.logsumexp(log_w_window_total)
                         - jnp.log(float(N)))

            end_particles = states_w[:, -1, :]                       # (N, D)

            # Resample at the window boundary.
            X = end_particles.T                                      # (D, N)
            X_resamp = gumbel_ste_resample(
                X, log_w_window_total, win_gumbel, params.get("tau", self.tau)
            )
            new_particles = X_resamp.T                               # (N, D)
            new_log_w_acc = jnp.zeros_like(log_w_acc)

            return (new_particles, new_log_w_acc), (states_w, iters, log_Z_inc)

        init = (x0s, jnp.zeros(N, dtype=x0s.dtype))
        scan_inputs = (
            jnp.arange(num_windows),
            obs_w,
            twist_w,
            gumbel_w,
            deer_keys_outer,
        )
        _, (states_chunks, iters_chunks, logZ_per_window) = jax.lax.scan(
            window_step, init, scan_inputs
        )
        # states_chunks: (num_windows, N, K, D) -> (N, T, D)
        states = states_chunks.transpose(1, 0, 2, 3).reshape(
            N, num_windows * K, self.D
        )
        log_Z = jnp.sum(logZ_per_window)
        return states, iters_chunks, log_Z

    def run_sequential(self, x0s, tape, params):
        """Sequential twisted PF with Gumbel-STE checkpoint resampling.

        Returns (states (N, T, D), iters_per_window (zeros), log_Z scalar).
        """
        return self._run(x0s, tape, params, mode="sequential")

    def run_parallel(self, x0s, tape, params, max_iter: int = 100):
        """Parallel-in-time twisted PF: per-window, per-particle DEER.

        Each window of K steps is solved by quasi-DEER, vmapped over
        particles. Resampling is applied serially at window boundaries.

        params must contain 'key' (used to split per-window DEER keys).
        Returns (states (N, T, D), iters_per_window (num_windows, N), log_Z).
        """
        return self._run(x0s, tape, params, mode="parallel",
                         max_iter=max_iter)
