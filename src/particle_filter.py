"""
particle_filter.py

Differentiable particle filter with OT-based resampling, GPU-compatible
via JAX and integrated with the quasi-DEER parallel framework.

The resampling step uses the regularised Sinkhorn algorithm in log-space,
making the full filter step end-to-end differentiable and compatible with
DEER's Jacobian-linearisation.  By pre-drawing the transition noise into a
"tape" the map  s_t -> s_{t+1}  becomes deterministic, so DEER can solve
all T steps simultaneously in O(log T) parallel depth.

Terminology:
  D  — latent state dimension
  N  — number of particles
  T  — number of timesteps

State vector `s` has shape (D*N,): particles stored column-major so that
  X = s.reshape(D, N)  gives the D×N particle matrix.

Julia prototype: OT.jl / ParticleFilters.jl (user, 2024).
"""

from typing import Callable, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jr

from src import qdeer

def _pairwise_sq_dist(X: jnp.ndarray) -> jnp.ndarray:
    """
    Pairwise squared Euclidean distances for columns of X.

    Parameters
    ----------
    X : (D, N)

    Returns
    -------
    C : (N, N),  C[i, j] = ||X[:,i] - X[:,j]||^2
    """
    sq = jnp.sum(X ** 2, axis=0)                   # (N,)
    C  = sq[:, None] + sq[None, :] - 2.0 * (X.T @ X)
    return jnp.maximum(C, 0.0)                      # clamp float rounding


def log_sinkhorn(
    log_a: jnp.ndarray,
    log_b: jnp.ndarray,
    C: jnp.ndarray,
    eps: float,
    max_iter: int = 100,
) -> jnp.ndarray:
    """
    Log-space Sinkhorn for regularised optimal transport.

    Solves
        min_{P}  <C, P>  +  eps * KL(P || exp(log_a) exp(log_b)^T)
        s.t.     P 1_M = exp(log_a),   P^T 1_N = exp(log_b)

    Uses jax.lax.scan for a static, JIT/AD-compatible computation graph.

    Parameters
    ----------
    log_a    : (N,)   log source marginal
    log_b    : (M,)   log target marginal
    C        : (N, M) cost matrix
    eps      : regularisation strength (> 0)
    max_iter : number of Sinkhorn iterations

    Returns
    -------
    log_P : (N, M) log transport plan
    """
    log_K = -C / eps   # (N, M)

    def step(carry, _):
        log_u, log_v = carry
        log_u = log_a - jax.scipy.special.logsumexp(
            log_K + log_v[None, :], axis=1
        )
        log_v = log_b - jax.scipy.special.logsumexp(
            log_K + log_u[:, None], axis=0
        )
        return (log_u, log_v), None

    init = (jnp.zeros_like(log_a), jnp.zeros_like(log_b))
    (log_u, log_v), _ = jax.lax.scan(step, init, None, length=max_iter)

    return log_u[:, None] + log_K + log_v[None, :]   # (N, M)


def ot_resample(
    X: jnp.ndarray,
    log_w: jnp.ndarray,
    eps: float = 0.1,
    max_iter: int = 100,
) -> jnp.ndarray:
    """
    Differentiable OT-based particle resampling.

    Transports the weighted source distribution  sum_i w_i delta(X[:,i])
    to the uniform distribution over N target locations, where the target
    locations are the result of the transport (i.e. new particles).

    For eps -> 0 this approximates deterministic-OT resampling; large eps
    collapses all particles to their weighted mean.  Intermediate eps gives
    a smooth, fully differentiable interpolation for gradient-based methods.

    Parameters
    ----------
    X       : (D, N)  particle matrix — each column is one particle
    log_w   : (N,)    unnormalised log-weights
    eps     : Sinkhorn regularisation (default 0.1)
    max_iter: Sinkhorn iterations    (default 100)

    Returns
    -------
    Y : (D, N)  resampled particles with approximately uniform weights
    """
    D, N = X.shape

    log_a = log_w - jax.scipy.special.logsumexp(log_w)  # normalise source
    log_b = jnp.full(N, -jnp.log(float(N)))             # uniform target

    C     = _pairwise_sq_dist(X)                          # (N, N)
    log_P = log_sinkhorn(log_a, log_b, C, eps, max_iter)  # (N, N)

    # Y[:, j] = N * sum_i P[i,j] * X[:, i]  =  X @ (N * P)
    return X @ (N * jnp.exp(log_P))                       # (D, N)

class ParallelParticleFilter:
    """
    Differentiable particle filter with OT resampling.

    Supports both sequential execution (jax.lax.scan) and parallel
    execution across time via quasi-DEER (qdeer.seq1d).

    Parameters
    ----------
    transition_fn : callable
        Signature: (X: (D,N), noise: (D,N), params: dict) -> X_pred: (D,N)
        Reparameterised transition — noise is pre-drawn so the map is
        deterministic and differentiable.
    log_lik_fn    : callable
        Signature: (X_pred: (D,N), obs, params: dict) -> log_w: (N,)
        Log-likelihood of the observation given each particle.
    D             : latent state dimension
    N             : number of particles
    ot_eps        : Sinkhorn regularisation strength (default 0.1)
    ot_iters      : Sinkhorn iterations (default 100)
    """

    def __init__(
        self,
        transition_fn: Callable,
        log_lik_fn: Callable,
        D: int,
        N: int,
        ot_eps: float = 0.1,
        ot_iters: int = 100,
    ):
        self.transition_fn = transition_fn
        self.log_lik_fn    = log_lik_fn
        self.D       = D
        self.N       = N
        self.ot_eps  = ot_eps
        self.ot_iters = ot_iters


    def make_tape(
        self,
        observations: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Pre-draw transition noise for all T timesteps.

        The returned tape is a JAX pytree  (observations, noise)  where
        each leaf has first dimension T.  It can be used directly as
        ``xinp`` in qdeer.seq1d or as ``xs`` in jax.lax.scan.

        Parameters
        ----------
        observations : (T, *obs_shape)
        key          : JAX PRNG key

        Returns
        -------
        tape : (observations, noise)  — noise has shape (T, D, N)
        """
        T     = observations.shape[0]
        noise = jr.normal(key, (T, self.D, self.N))
        return (observations, noise)

    def pf_step(
        self,
        state: jnp.ndarray,
        driver: Tuple[jnp.ndarray, jnp.ndarray],
        params: dict,
    ) -> jnp.ndarray:
        """
        One particle filter step.

        Matches the DEER/qdeer convention:  func(state, driver, params) -> next_state

        Parameters
        ----------
        state  : (D*N,)  flattened particle array
        driver : (obs_t, noise_t)
                   obs_t   has shape (*obs_shape,)
                   noise_t has shape (D, N)
        params : dict forwarded to transition_fn and log_lik_fn.
                 Must contain 'key' when used with run_parallel.
                 May contain 'ot_eps' to override the instance default.

        Returns
        -------
        next_state : (D*N,)
        """
        obs, noise = driver
        X      = state.reshape(self.D, self.N)
        X_pred = self.transition_fn(X, noise, params)         # (D, N)
        log_w  = self.log_lik_fn(X_pred, obs, params)         # (N,)
        ot_eps = params.get("ot_eps", self.ot_eps)
        X_new  = ot_resample(X_pred, log_w, ot_eps, self.ot_iters)
        return X_new.ravel()                                   # (D*N,)


    def run_sequential(
        self,
        s0: jnp.ndarray,
        tape: Tuple[jnp.ndarray, jnp.ndarray],
        params: dict,
    ) -> jnp.ndarray:
        """
        Sequential particle filter via jax.lax.scan.

        Parameters
        ----------
        s0     : (D*N,)  initial flattened particles
        tape   : output of make_tape
        params : model parameters dict

        Returns
        -------
        states : (T, D*N)  particle states at each timestep
        """
        def _scan_fn(state, driver):
            next_state = self.pf_step(state, driver, params)
            return next_state, next_state

        _, states = jax.lax.scan(_scan_fn, s0, tape)
        return states

    def run_parallel(
        self,
        s0: jnp.ndarray,
        tape: Tuple[jnp.ndarray, jnp.ndarray],
        params: dict,
        yinit_guess: Optional[jnp.ndarray] = None,
        max_iter: int = 100,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, int]:
        """
        Parallel particle filter via quasi-DEER (O(log T) depth).

        Finds all T filter states simultaneously by iterating the
        Jacobian-linearised associative scan from DEER until convergence.

        Parameters
        ----------
        s0          : (D*N,)    initial flattened particles
        tape        : output of make_tape
        params      : dict, must include 'key' for the Hutchinson estimator
        yinit_guess : (T, D*N) initial guess; defaults to zeros if None
        max_iter    : max quasi-DEER Newton iterations
        **kwargs    : forwarded to qdeer.seq1d  (e.g. clip_val, damp_factor,
                      full_trace)

        Returns
        -------
        states     : (T, D*N)  particle states at each timestep
        samp_iters : number of DEER iterations taken
        """
        states, samp_iters = qdeer.seq1d(
            self.pf_step,
            s0,
            tape,
            params,
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            **kwargs,
        )
        return states, samp_iters
