"""
stampede.py

STAMPEDE: Simultaneous Twisted Associative Markov-chain Particle
          Estimation via DEER

Doubly-parallel particle MCMC:
  - Outer loop: Parallel MALA (via DEER) using ParallelMALA from samplers.py
  - Inner loop: Twisted particle smoother for p(y|θ) evaluation

The PF likelihood uses fixed randomness (correlated pseudo-marginal MCMC,
Deligiannidis et al. 2018), making the log posterior a smooth deterministic
function of θ suitable for gradient-based MCMC and DEER parallelisation.
"""

from typing import Callable, Any, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from src.particle_filter import TwistedParallelParticleFilter
from src.samplers import ParallelMALA


class STAMPEDE:
    """
    Doubly-parallel particle MCMC.

    Composes ParallelMALA (outer DEER loop over MCMC iterations) with
    TwistedParallelParticleFilter (inner sequential PF for likelihood).

    Parameters
    ----------
    twisted_step_fn : callable
        Per-particle twisted transition: (x, driver, model_params) -> x_next
    pf_log_weight_fn : callable
        Per-particle log weight: (x_prev, x_next, obs, twist, model_params) -> scalar
    log_prior_fn : callable
        Log prior on unconstrained parameters: (theta) -> scalar
    theta_to_params_fn : callable
        Maps unconstrained theta to model params dict: (theta) -> dict
    D_state : int
        Latent state dimension per particle.
    D_theta : int
        Dimension of the unconstrained parameter vector.
    N_particles : int
        Number of particles.
    chain_length : int
        MCMC chain length (K).
    max_deer_iter : int
        Max DEER iterations for outer MCMC loop.
    damp_factor : float
        DEER damping for outer MCMC loop.
    clip_val : float
        Gradient clipping value for DEER.
    """

    def __init__(
        self,
        twisted_step_fn: Callable,
        pf_log_weight_fn: Callable,
        log_prior_fn: Callable,
        theta_to_params_fn: Callable,
        D_state: int,
        D_theta: int,
        N_particles: int,
        chain_length: int,
        max_deer_iter: int = 50,
        damp_factor: float = 0.8,
        clip_val: float = 1.0,
        compute_twists_fn: Callable = None,
    ):
        self.tpf = TwistedParallelParticleFilter(
            twisted_step_fn, pf_log_weight_fn, D=D_state,
        )
        self.log_prior_fn = log_prior_fn
        self.theta_to_params_fn = theta_to_params_fn
        self.D_state = D_state
        self.N = N_particles
        self.compute_twists_fn = compute_twists_fn

        # Build the log posterior that ParallelMALA will target
        def log_posterior(theta, target_params):
            model_params = theta_to_params_fn(theta)
            if compute_twists_fn is not None:
                twist_params = compute_twists_fn(
                    model_params, target_params['observations'])
            else:
                twist_params = target_params['twist_params']
            tape = (
                target_params['observations'],
                target_params['pf_noise'],
                twist_params,
            )
            states = self.tpf.run_sequential(
                target_params['x0s'], tape, model_params,
            )
            log_ws = self.tpf.compute_log_weights(
                target_params['x0s'], states, tape, model_params,
            )
            total = jnp.sum(log_ws, axis=1)                         # (N,)
            log_ml = jax.scipy.special.logsumexp(total) - jnp.log(float(N_particles))
            return log_ml + log_prior_fn(theta)

        self.log_posterior = log_posterior

        # Instantiate ParallelMALA from the existing codebase
        self.sampler = ParallelMALA(
            log_prob=log_posterior,
            dim=D_theta,
            chain_length=chain_length,
            max_iter=max_deer_iter,
            clip_val=clip_val,
            damp_factor=damp_factor,
        )

    def run_sequential(self, key, theta0, params):
        """Sequential PMCMC via jax.lax.scan (baseline)."""
        return self.sampler.run_sequential_mala(key, theta0, params)

    def run_parallel(self, key, theta0, yinit_guess, params):
        """Parallel PMCMC via quasi-DEER."""
        return self.sampler.run_parallel_mala(key, theta0, yinit_guess, params)
