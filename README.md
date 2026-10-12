# Differentiable, parallel-in-time particle filters

Research code for a particle filter that is (i) an exact SMC in its forward
pass, (ii) parallel across time, and (iii) differentiable end-to-end.
Twisted (look-ahead) proposals turn every particle into an independent
nonlinear recurrence that DEER solves in O(log K) parallel depth per window;
resampling happens only at window boundaries using Gumbel-max sampling with
a straight-through softmax Jacobian. The same object gives gradient-based
maximum likelihood and gradient-based (MALA) particle MCMC.

Builds on the DEER / quasi-DEER / ELK solvers from
[lindermanlab/parallel-mcmc](https://github.com/lindermanlab/parallel-mcmc)
(Zoltowski et al., NeurIPS 2025).

See `PAPER_PLAN.md` for the current paper outline, experiment list, and status.

## Layout
```
src/
  gumbel_particle_filter.py   TwistedGumbelParticleFilter (headline method) and the
                              per-step GumbelParallelParticleFilter ablation
  particle_filter.py          TwistedParallelParticleFilter: no-resampling twisted PF,
                              exactly reparameterised (used to train twists)
  samplers.py                 ParallelMALA with frozen-tape, forward-mode gradients
  baselines.py                bootstrap PF, FFBSi smoother
  deer.py, qdeer.py, elk.py, windowed_qdeer.py   parallel-in-time solvers
examples/
  run_twisted_gumbel_pf_*.py          correctness, K-sweeps, Lorenz-63, stochvol
  run_gumbel_pf_*.py                  per-step-resampling ablation
  run_stampede_multichain*.py         multi-chain MALA over the parallel PF
  run_particle_em_*.py                particle EM (stochvol, Poisson LDS)
  run_twisted_pf_{lorenz63,lorenz96,stochvol}.py   model + twist code for older
                                      no-resampling experiments (to be folded into src/models)
scripts/                      SGE submission scripts (BU SCC, L40S GPUs)
```

## Setup
Python >= 3.11, managed with `uv`; install into the project directory.
```
uv sync
uv run python examples/run_twisted_gumbel_pf_lgssm.py
```
Runs on CPU for testing; speedups require a GPU.
