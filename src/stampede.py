"""
stampede.py

STAMPEDE: Simultaneous Twisted Associative Markov-chain Particle
          Estimation via DEER

Doubly-parallel particle MCMC:
  - Outer loop: Parallel MALA (via DEER) for model parameters θ
  - Inner loop: Twisted particle smoother for p(y|θ) evaluation

The PF likelihood uses fixed randomness (correlated pseudo-marginal MCMC,
Deligiannidis et al. 2018), making the log posterior a smooth deterministic
function of θ suitable for gradient-based MCMC and DEER parallelisation.
"""

from typing import Callable, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from src import qdeer
from src.particle_filter import TwistedParallelParticleFilter

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def sigmoid_accept(x):
    """Differentiable approximation to accept/reject."""
    zero = jax.nn.sigmoid(x) - jax.lax.stop_gradient(jax.nn.sigmoid(x))
    return zero + jax.lax.stop_gradient((x > 0))


class STAMPEDE:
    """
    Doubly-parallel particle MCMC.

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
    N_particles : int
        Number of particles.
    chain_length : int
        MCMC chain length (K).
    max_deer_iter : int
        Max DEER iterations for outer MCMC loop.
    damp_factor : float
        DEER damping for outer MCMC loop.
    """

    def __init__(
        self,
        twisted_step_fn: Callable,
        pf_log_weight_fn: Callable,
        log_prior_fn: Callable,
        theta_to_params_fn: Callable,
        D_state: int,
        N_particles: int,
        chain_length: int,
        max_deer_iter: int = 50,
        damp_factor: float = 0.8,
        clip_val: float = 1.0,
    ):
        self.tpf = TwistedParallelParticleFilter(
            twisted_step_fn, pf_log_weight_fn, D=D_state,
        )
        self.log_prior_fn = log_prior_fn
        self.theta_to_params_fn = theta_to_params_fn
        self.D_state = D_state
        self.N = N_particles
        self.K = chain_length
        self.max_deer_iter = max_deer_iter
        self.damp_factor = damp_factor
        self.clip_val = clip_val

        self._log_post_and_grad = jax.value_and_grad(self._log_posterior)

    def _log_posterior(self, theta: jnp.ndarray, target_params: dict) -> float:
        """
        PF-based log posterior:  log p(θ) + log p̂(y|θ).

        target_params must contain:
            observations, pf_noise, twist_params, x0s
        """
        model_params = self.theta_to_params_fn(theta)
        tape = (
            target_params['observations'],
            target_params['pf_noise'],
            target_params['twist_params'],
        )
        states = self.tpf.run_sequential(
            target_params['x0s'], tape, model_params,
        )
        log_ws = self.tpf.compute_log_weights(
            target_params['x0s'], states, tape, model_params,
        )
        total = jnp.sum(log_ws, axis=1)                             # (N,)
        log_ml = jax.scipy.special.logsumexp(total) - jnp.log(float(self.N))
        return log_ml + self.log_prior_fn(theta)

    # ── MALA step (matches ParallelMALA convention) ───────────────────────
    def _mala_step(
        self,
        theta: jnp.ndarray,
        driver: jnp.ndarray,
        params: dict,
    ) -> jnp.ndarray:
        """One MALA step targeting the PF-based posterior."""
        step_size = params['step_size']
        key, prop_key, accept_key = jr.split(driver, 3)

        lp, grad = self._log_post_and_grad(theta, params['target_params'])

        # propose
        theta_star = (theta + step_size * grad
                      + jnp.sqrt(2.0 * step_size) * jr.normal(prop_key, theta.shape))

        # new log-prob + grad
        lp_star, grad_star = self._log_post_and_grad(theta_star, params['target_params'])

        # MH correction (forward/reverse proposal densities)
        scale = jnp.sqrt(2.0 * step_size) * jnp.ones_like(theta)
        log_fwd = tfd.MultivariateNormalDiag(
            loc=theta + step_size * grad, scale_diag=scale,
        ).log_prob(theta_star)
        log_rev = tfd.MultivariateNormalDiag(
            loc=theta_star + step_size * grad_star, scale_diag=scale,
        ).log_prob(theta)

        log_alpha = lp_star + log_rev - lp - log_fwd
        accept = sigmoid_accept(log_alpha - jnp.log(jr.uniform(accept_key)))
        return accept * theta_star + (1.0 - accept) * theta

    # ── Sequential PMCMC ──────────────────────────────────────────────────
    def run_sequential(
        self,
        key: jnp.ndarray,
        theta0: jnp.ndarray,
        params: dict,
    ) -> jnp.ndarray:
        """
        Sequential PMCMC via jax.lax.scan.

        Returns
        -------
        chain : (K, D_theta)
        """
        drivers = jr.split(key, self.K)

        def _scan_fn(theta, driver):
            theta_new = self._mala_step(theta, driver, params)
            return theta_new, theta_new

        _, chain = jax.lax.scan(_scan_fn, theta0, drivers)
        return chain

    # ── Parallel PMCMC (DEER) ─────────────────────────────────────────────
    def run_parallel(
        self,
        key: jnp.ndarray,
        theta0: jnp.ndarray,
        params: dict,
        yinit_guess: Optional[jnp.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, int]:
        """
        Parallel PMCMC via quasi-DEER.

        Returns
        -------
        chain      : (K, D_theta)
        deer_iters : number of outer DEER iterations
        """
        drivers = jr.split(key, self.K)
        chain, iters = qdeer.seq1d(
            self._mala_step,
            theta0,
            drivers,
            params,
            yinit_guess=yinit_guess,
            max_iter=self.max_deer_iter,
            damp_factor=self.damp_factor,
            clip_val=self.clip_val,
            **kwargs,
        )
        return chain, iters
