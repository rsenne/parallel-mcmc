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
import jaxopt
from functools import partial

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


def _sinkhorn_uv(log_a, log_b, log_K, max_iter):
    """Inner Sinkhorn fixed-point iteration in log-space. Returns (u*, v*)."""
    def step(carry, _):
        log_u, log_v = carry
        log_u = log_a - jax.scipy.special.logsumexp(log_K + log_v[None, :], axis=1)
        log_v = log_b - jax.scipy.special.logsumexp(log_K + log_u[:, None], axis=0)
        return (log_u, log_v), None
    init = (jnp.zeros_like(log_a), jnp.zeros_like(log_b))
    (log_u, log_v), _ = jax.lax.scan(step, init, None, length=max_iter)
    return log_u, log_v


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _log_sinkhorn_impl(log_a, log_b, log_K, max_iter):
    """
    Sinkhorn log-transport-plan with implicit differentiation.

    Forward: ordinary Sinkhorn iterations.
    JVP:     iterates the linearized update at the converged (u*, v*) to
             solve the implicit-function-theorem system, giving exact
             gradients at the fixed point with O(1) backward memory.
    VJP:     obtained automatically from the linear JVP rule by JAX's
             transposition, so both jax.jvp (used by DEER) and jax.grad
             (used by MALA) work through this operator.
    """
    log_u, log_v = _sinkhorn_uv(log_a, log_b, log_K, max_iter)
    return log_u[:, None] + log_K + log_v[None, :]


@_log_sinkhorn_impl.defjvp
def _log_sinkhorn_impl_jvp(max_iter, primals, tangents):
    log_a, log_b, log_K = primals
    dlog_a, dlog_b, dlog_K = tangents

    # Forward
    log_u, log_v = _sinkhorn_uv(log_a, log_b, log_K, max_iter)
    log_P = log_u[:, None] + log_K + log_v[None, :]

    # Row- and column-stochastic matrices used in the linearized update.
    # Compute via softmax so they are exactly row/col-stochastic even if
    # the forward Sinkhorn hasn't fully converged — otherwise the
    # linearized iteration can diverge (spectral radius > 1).
    A = jax.nn.softmax(log_K + log_v[None, :], axis=1)   # (N, M) rows sum to 1
    C = jax.nn.softmax(log_K + log_u[:, None], axis=0)   # (N, M) cols sum to 1

    # Direct θ-tangent contributions to the linearized update
    # f_i(v;θ)  = log_a_i  - LSE_j(log_K[i,j] + v_j)   ⇒ df_direct_i = dlog_a_i - sum_l A[i,l]*dlog_K[i,l]
    # g_j(u;θ)  = log_b_j  - LSE_i(log_K[i,j] + u_i)   ⇒ dg_direct_j = dlog_b_j - sum_k C[k,j]*dlog_K[k,j]
    du_direct = dlog_a - jnp.sum(A * dlog_K, axis=1)
    dv_direct = dlog_b - jnp.sum(C * dlog_K, axis=0)

    # Linearized alternating iteration for the implicit-function-theorem
    # system. Because A, C are stochastic by construction (softmax), the
    # iteration contracts at the same rate as Sinkhorn itself
    _LIN_ITERS = 60
    du = jnp.zeros_like(log_a)
    dv = jnp.zeros_like(log_b)
    for _ in range(_LIN_ITERS):
        du = du_direct - A @ dv
        dv = dv_direct - C.T @ du

    dlog_P = du[:, None] + dlog_K + dv[None, :]
    return log_P, dlog_P


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

    Uses implicit differentiation (custom_jvp) so the backward pass does
    NOT unroll the Sinkhorn scan — O(1) memory in max_iter, exact gradient
    at the fixed point, and compatible with both jax.grad (reverse mode)
    and jax.jvp (forward mode, used by DEER).

    Parameters
    ----------
    log_a    : (N,)   log source marginal
    log_b    : (M,)   log target marginal
    C        : (N, M) cost matrix
    eps      : regularisation strength (> 0)
    max_iter : number of Sinkhorn iterations (fwd + linearized bwd)

    Returns
    -------
    log_P : (N, M) log transport plan
    """
    log_K = -C / eps
    return _log_sinkhorn_impl(log_a, log_b, log_K, max_iter)


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

    C     = _pairwise_sq_dist(X)                          
    log_P = log_sinkhorn(log_a, log_b, C, eps, max_iter)  

    # Y[:, j] = N * sum_i P[i,j] * X[:, i]  =  X @ (N * P)
    return X @ (N * jnp.exp(log_P))                     


def ot_resample_ste(
    X: jnp.ndarray,
    log_w: jnp.ndarray,
    eps: float = 0.1,
    max_iter: int = 100,
) -> jnp.ndarray:
    """Stop-gradient straight-through variant of ot_resample.

    Forward is identical. Backward treats the transport plan T as a
    constant, so gradients flow only through the pathwise term X @ T —
    not through the Sinkhorn fixed point. This removes the implicit-JVP
    layer (and its forward-over-reverse Hess·v) that makes DEER converge
    slowly on OT-PF + parallel MALA, at the cost of a biased gradient
    estimator (Corenflos et al., 2021-style STE).
    """
    D, N = X.shape
    log_a = log_w - jax.scipy.special.logsumexp(log_w)
    log_b = jnp.full(N, -jnp.log(float(N)))
    C     = _pairwise_sq_dist(jax.lax.stop_gradient(X))
    log_P = log_sinkhorn(
        jax.lax.stop_gradient(log_a),
        log_b,
        jax.lax.stop_gradient(C),
        eps,
        max_iter,
    )
    T_mat = jax.lax.stop_gradient(N * jnp.exp(log_P))
    return X @ T_mat

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


class TwistedParallelParticleFilter:
    """
    Parallel-in-time particle filter via twisted proposals + per-particle DEER.

    Parameters
    ----------
    twisted_step_fn : callable
        (state: (D,), driver: tuple, params: dict) -> next_state: (D,)
        Deterministic twisted transition for one particle.
        driver = (obs_t, noise_t, twist_params_t).
    log_weight_fn : callable
        (x_prev: (D,), x_next: (D,), obs_t, twist_params_t, params: dict) -> scalar
        Incremental log importance weight at one timestep.
    D : int
        Latent dimension per particle.
    """

    def __init__(
        self,
        twisted_step_fn: Callable,
        log_weight_fn: Callable,
        D: int,
    ):
        self.twisted_step_fn = twisted_step_fn
        self.log_weight_fn = log_weight_fn
        self.D = D

    def make_tape(
        self,
        observations: jnp.ndarray,
        twist_params: Any,
        key: jnp.ndarray,
        N: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
        """
        Pre-draw per-particle noise for all T timesteps.

        Parameters
        ----------
        observations : (T, *obs_shape)
        twist_params : pytree with leading dimension T
        key          : JAX PRNG key
        N            : number of particles

        Returns
        -------
        tape : (observations, noise, twist_params)
               noise has shape (N, T, D)
        """
        T = observations.shape[0]
        noise = jr.normal(key, (N, T, self.D))
        return (observations, noise, twist_params)

    def _step(self, state, driver, params):
        """DEER-compatible step for a single particle."""
        return self.twisted_step_fn(state, driver, params)

    def run_sequential(
        self,
        x0s: jnp.ndarray,
        tape: Tuple,
        params: dict,
    ) -> jnp.ndarray:
        """
        Sequential twisted PF: jax.lax.scan per particle, vmapped over N.

        Parameters
        ----------
        x0s    : (N, D) initial particles
        tape   : output of make_tape
        params : model parameters dict

        Returns
        -------
        states : (N, T, D)
        """
        observations, noise, twist_params = tape

        def run_one(x0, noise_i):
            def _scan_fn(state, driver):
                next_state = self._step(state, driver, params)
                return next_state, next_state
            _, states = jax.lax.scan(
                _scan_fn, x0, (observations, noise_i, twist_params)
            )
            return states

        return jax.vmap(run_one)(x0s, noise)

    def run_parallel(
        self,
        x0s: jnp.ndarray,
        tape: Tuple,
        params: dict,
        max_iter: int = 100,
        basis: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parallel twisted PF: per-particle DEER, vmapped over N.

        Parameters
        ----------
        x0s      : (N, D) initial particles
        tape     : output of make_tape
        params   : dict, must include 'key'
        max_iter : max DEER iterations
        basis    : optional (R, R_inv) with R of shape (D, D).
                   When provided, DEER operates in the rotated coordinate
                   system z = R @ x so that the per-step Jacobian
                   R @ J @ R^{-1} is more diagonal.  Results are returned
                   in the original x-space.
        **kwargs : forwarded to qdeer.seq1d

        Returns
        -------
        states : (N, T, D)
        iters  : (N,) DEER iterations per particle
        """
        observations, noise, twist_params = tape
        N = x0s.shape[0]
        keys = jr.split(params['key'], N)

        if basis is not None:
            R, R_inv = basis
            base_step = self._step
            def deer_step(z, driver, params):
                x = R_inv @ z
                x_next = base_step(x, driver, params)
                return R @ x_next
            x0s_solve = x0s @ R.T
            # Rotate the DEER warm-start into the new coordinate system
            if 'yinit_guess' in kwargs and kwargs['yinit_guess'] is not None:
                kwargs = {**kwargs, 'yinit_guess': kwargs['yinit_guess'] @ R.T}
        else:
            deer_step = self._step
            x0s_solve = x0s

        def solve_one(x0, noise_i, key_i):
            particle_tape = (observations, noise_i, twist_params)
            params_i = {**params, 'key': key_i}
            return qdeer.seq1d(
                deer_step, x0, particle_tape, params_i,
                max_iter=max_iter, **kwargs,
            )

        states, iters = jax.vmap(solve_one)(x0s_solve, noise, keys)

        if basis is not None:
            states = states @ R_inv.T

        return states, iters

    def compute_log_weights(
        self,
        x0s: jnp.ndarray,
        states: jnp.ndarray,
        tape: Tuple,
        params: dict,
    ) -> jnp.ndarray:
        """
        Compute per-particle, per-timestep log importance weights.

        Parameters
        ----------
        x0s    : (N, D)
        states : (N, T, D)
        tape   : output of make_tape
        params : model parameters dict

        Returns
        -------
        log_weights : (N, T)
        """
        observations, noise, twist_params = tape

        def weights_one(x0, trajectory):
            x_prevs = jnp.concatenate([x0[None, :], trajectory[:-1]], axis=0)

            def weight_one_step(x_prev, x_next, obs_t, twist_t):
                return self.log_weight_fn(x_prev, x_next, obs_t, twist_t, params)

            return jax.vmap(weight_one_step)(
                x_prevs, trajectory, observations, twist_params
            )

        return jax.vmap(weights_one)(x0s, states)
