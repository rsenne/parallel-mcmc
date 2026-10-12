import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from functools import partial

# core DEER + QDEER algorithms
import src
from src import deer, qdeer, windowed_qdeer, elk

def _diag_mvn_logpdf(x, loc, scale_diag):
    """Log-pdf of a diagonal multivariate normal (replaces tfp dependency)."""
    d = x - loc
    return -0.5 * jnp.sum(d**2 / scale_diag**2) - jnp.sum(jnp.log(scale_diag)) - 0.5 * x.shape[0] * jnp.log(2 * jnp.pi)

def sigmoid_accept(x):
    zero = jax.nn.sigmoid(x) - jax.lax.stop_gradient(jax.nn.sigmoid(x)) # zero on fwd pass.
    return zero + jax.lax.stop_gradient((x > 0)) 

class ParallelMALA:
    
    # our constructor
    def __init__(self, log_prob, dim, chain_length, max_iter, alg="quasi",
                 clip_val=1.0, damp_factor=1.0, full_trace=False,
                 basis_transformation=False, window_size=None, sigmasq=1e8,
                 autodiff_mode="reverse"):
        '''
        Args:
            logp - unnormalized log-posterior function that ONLY takes in theta as argument. Use partial.
            dim - how many dimensions is our parameter space for sampling? (added together)
            epsilon - the MALA stepsize.
            alg - parallel solver: "quasi" (quasi-DEER, diagonal Hutchinson + associative scan)
                  or "quasi_elk" (Levenberg-Marquardt damped Newton via scalar Kalman filter).
                  quasi_elk is the sigmasq -> inf limit of quasi; at finite sigmasq it pulls
                  DEER iterates toward the current guess, avoiding the pathological-Jacobian
                  excursions that normally require gradient-norm clipping in the MALA map.
            clip_val - what are we clipping individual gradient entries to in absolute value?
            damp_factor - slightly damping the Jacobian.
            sigmasq - Quasi-ELK emission noise (ignored when alg="quasi"). Large sigmasq ~= DEER.
        '''
        # 1. internalize + get the target log-prob and grad
        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        # autodiff_mode="forward" uses jax.jvp instead of jax.value_and_grad,
        # so the log_prob can contain jax.lax.while_loop (which reverse-mode
        # cannot differentiate).  Cost: D forward passes per gradient instead
        # of one fwd+bwd pair.  Worth it when the forward pass is much
        # cheaper, e.g. parallel-in-time twisted PF inside log_prob.
        if autodiff_mode == "forward":
            def _value_and_grad_fwd(theta, target_params):
                f = lambda t: log_prob(t, target_params)
                eye = jnp.eye(theta.shape[0], dtype=theta.dtype)
                pushforward = lambda v: jax.jvp(f, (theta,), (v,))
                primals, tangents = jax.vmap(pushforward,
                                              in_axes=0,
                                              out_axes=(0, 0))(eye)
                return primals[0], tangents
            self.target_log_prob_and_grad = _value_and_grad_fwd
        else:
            self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self.autodiff_mode = autodiff_mode
        self.max_iter = max_iter
        self.alg = alg
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace
        self.basis_transformation = basis_transformation
        self.window_size = window_size
        self.sigmasq = sigmasq
        if self.window_size is not None:
            assert 1 <= self.window_size <= self.chain_length
        
    def mala_fxn_for_seq(self, state, driver, params):
        step_size = params["step_size"]
        key, *skeys = jr.split(driver, 3)

        # Gradient-norm clip, bounded so DEER's Newton iterates stay in a well-conditioned
        # Jacobian regime. Pass grad_max_norm=None to disable (e.g., when pairing Quasi-ELK
        # damping with the unclipped MALA map); the lambda is then skipped at trace time so
        # autodiff doesn't encounter 0*inf through jnp.minimum.
        grad_max_norm = params.get("grad_max_norm", 50.0)

        if grad_max_norm is None:
            _clip_norm = lambda g: g
        else:
            def _clip_norm(g):
                gnorm = jnp.linalg.norm(g)
                scale = jnp.minimum(1.0, grad_max_norm / (gnorm + 1e-12))
                return g * scale

        logprob_state, grad_state = self.target_log_prob_and_grad(state, params["target_params"])
        grad_state = _clip_norm(grad_state)
        next_state = state + step_size * grad_state
        next_state = next_state + jnp.sqrt(2.0 * step_size) * jr.normal(skeys[0], (state.shape[0],))

        # get new log prob and grad
        logprob_nextstate, grad_nextstate = self.target_log_prob_and_grad(next_state, params["target_params"])
        grad_nextstate = _clip_norm(grad_nextstate)

        # accept / reject
        scale = jnp.sqrt(2.0 * step_size) * jnp.ones_like(state)
        num = logprob_nextstate + _diag_mvn_logpdf(
            state, next_state + step_size * grad_nextstate, scale)
        den = logprob_state + _diag_mvn_logpdf(
            next_state, state + step_size * grad_state, scale)
        g = sigmoid_accept(num-den-jnp.log(jr.uniform(skeys[1])))
        next_state = g*next_state + (1.0-g)*state

        return next_state

    def mala_fxn_for_deer(self, state, driver, params):
        # if transform basis, do initial transformation (assume orthogonal)
        if self.basis_transformation:
            state = params["basis"] @ state 

        # then run seq update
        next_state = self.mala_fxn_for_seq(state, driver, params)

        # transform reverse
        if self.basis_transformation:
            next_state = params["basis"].T @ next_state

        return next_state

    def run_parallel_mala(self, key, initial_state, yinit_guess, params):
        drivers = jr.split(key, (self.chain_length,))

        if self.basis_transformation:
            initial_state = params["basis"].T @ initial_state
            yinit_guess = jnp.einsum('...ij, ...j -> ...i', params["basis"].T, yinit_guess)

        if self.alg == "quasi_elk":
            sigmasq = params.get("sigmasq", self.sigmasq)
            out_states, iters = elk.quasi_elk_seq1d(
                self.mala_fxn_for_deer, initial_state, drivers, params,
                yinit_guess=yinit_guess, sigmasq=sigmasq,
                max_iter=self.max_iter, clip_val=self.clip_val,
                full_trace=self.full_trace, damp_factor=self.damp_factor,
            )
        else:
            out_states, iters = qdeer.seq1d(
                self.mala_fxn_for_deer, initial_state, drivers, params,
                yinit_guess=yinit_guess, max_iter=self.max_iter, clip_val=self.clip_val,
                full_trace=self.full_trace, damp_factor=self.damp_factor
            )

        if self.basis_transformation:
            out_states = jnp.einsum('...ij, ...j -> ...i', params["basis"], out_states)

        return out_states, iters

    def run_parallel_mala_window(self, key, initial_state, yinit_guess, params):
        drivers = jr.split(key, (self.chain_length,))

        if self.basis_transformation:
            initial_state = params["basis"].T @ initial_state 
            yinit_guess = jnp.einsum('...ij, ...j -> ...i', params["basis"].T, yinit_guess)

        out_states, iters = windowed_qdeer.seq1d(
            self.mala_fxn_for_deer, initial_state, drivers, params, self.window_size,
            yinit_guess=yinit_guess, max_iter=self.max_iter, clip_val=self.clip_val,
            full_trace=self.full_trace, damp_factor=self.damp_factor
        )

        if self.basis_transformation:
            out_states = jnp.einsum('...ij, ...j -> ...i', params["basis"], out_states)

        return out_states, iters 

    def run_sequential_mala(self, key, initial_state, params):

        def _fxn_for_scan(state, driver):
            state = self.mala_fxn_for_seq(state, driver, params)
            return state, state 

        drivers = jr.split(key, (self.chain_length,))
        _, out_states = jax.lax.scan(_fxn_for_scan, initial_state, drivers)

        return out_states
