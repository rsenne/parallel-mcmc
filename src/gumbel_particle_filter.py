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
