"""
particle_filter.py

Twisted particle filter *without* resampling, parallelised across time by
per-particle quasi-DEER.

Each particle follows a deterministic twisted recurrence once its transition
noise is pre-drawn into a "tape", so the N particle trajectories are
independent D-dimensional nonlinear recurrences that DEER solves in
O(log T) parallel depth. The filter is fully reparameterised, hence exactly
differentiable in model and twist parameters; it is used here to *train*
twists (ELBO / VSMC objective) and as the no-resampling reference for the
windowed Gumbel-STE filter in gumbel_particle_filter.py, which adds
resampling at window boundaries to control weight degeneracy.

Terminology:
  D  — latent state dimension
  N  — number of particles
  T  — number of timesteps
"""

from typing import Callable, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jr

from src import qdeer


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
