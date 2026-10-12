# ICML 2027 paper plan — differentiable, parallel-in-time particle filters

Status as of 2026-10-12. Deadline: ICML 2027 abstract/paper deadline is not yet
announced; third-party trackers project ~Jan 16 (abstract) / Jan 22 (paper) 2027.
Plan for ~13 working weeks.

## 1. Thesis

A particle filter whose forward pass is an *exact* SMC (unbiased marginal
likelihood, correct resampling law) but which is (i) parallel in time and
(ii) differentiable end-to-end. Twisted proposals make each particle an
independent nonlinear recurrence that DEER solves in O(log K) depth per
window; resampling happens only at window boundaries via Gumbel-max with a
straight-through softmax Jacobian. Three payoffs that VI already has and
plain PFs lack: parallel hardware utilisation, gradient-based MLE, and
gradient-based (MALA) particle MCMC. Unlike VI, the posterior approximation
is a consistent SMC approximation, so it holds up on multimodal / chaotic /
strongly non-Gaussian posteriors.

**Positioning sentence.** Twist learning on log Ẑ *is* variational SMC
(Maddison 2017 FIVO; Naesseth 2018 VSMC; Le 2018 AESMC): our filter is a
parallel-in-time VSMC with an exact-forward, STE-backward resampler. That
framing makes the "as good as VI" claim precise: same objective family, but
the inference engine is SMC rather than a parametric family.

## 2. What is in hand (branch Gumbel_STE_PF)

| Component | File | Status |
|---|---|---|
| Windowed twisted-Gumbel PF (headline) | `src/gumbel_particle_filter.py` `TwistedGumbelParticleFilter` | Done. par == seq to 1e-7, 2 DEER iters/window on LGSSM; 77× at K=2048 T=8192 (L40S) |
| Per-step Gumbel PF (negative ablation) | same file, `GumbelParallelParticleFilter` | Done. iters ∝ T, motivates windowing |
| No-resampling twisted PF (exact reparam; used to train twists) | `src/particle_filter.py` | Done |
| DEER / quasi-DEER / ELK / windowed | `src/deer.py`, `qdeer.py`, `elk.py`, `windowed_qdeer.py` | Upstream, stable |
| Frozen-tape MALA, forward-mode grads (needed because PF has while_loop) | `src/samplers.py` `ParallelMALA(autodiff_mode="forward")` | Done |
| Multi-chain pMCMC on LGSSM / SV (φ only) | `examples/run_stampede_multichain*.py` | Done; SV R-hat unstable at M=8/16 with N=64 |
| Lorenz-63 learned full-cov twist + K-sweep | `examples/run_twisted_gumbel_pf_lorenz63_learned.py` | Done; ELBO-trained twist is *worse* than EKF-init (open problem) |
| SV stress test | `examples/run_twisted_gumbel_pf_stochvol.py` | Done |
| Particle EM with Gumbel E-step (SV) | `examples/run_particle_em_stochvol_gumbel.py` | Done; 3.3× vs FFBSi EM |
| PLDS EM, Lorenz-96 (full-Jacobian DEER) | old no-resampling filter only | Needs port to Gumbel filter |
| Full-Jacobian DEER for coupled D | only inline in `run_twisted_pf_lorenz96.py` | Needs to move into `src/` and into the Gumbel filter |
| VI baseline | — | Missing |
| dSMC baseline (Corenflos et al. 2022, JAX code `AdrienCorenflos/parallel-ps`) | — | Missing |
| Gradient-fidelity experiment | — | Missing (critical) |

Deleted 2026-10-12: OT/Sinkhorn PF, rank-1 DEER, single-chain outer-DEER
STAMPEDE (confirmed negative, 0.08–0.69×), upstream MCMC demos, ParallelHMC,
old figure-composition scripts, old LGSSM-only no-resampling scripts.
Phase-1 numbers survive in `results_consolidation/SUMMARY.md` (gitignored)
and in `git show dee561e:paper/neurips_2026.tex`.

## 3. Experiments

Priority order. "Cost" is engineering time; GPU time is seconds-to-minutes
everywhere.

### E1. Correctness and scaling (LGSSM, Kalman ground truth)  — mostly done
- par vs seq max-abs diff, log Ẑ vs Kalman log-likelihood, filtering RMSE.
- Sweeps: T (128→65k), K (16→T), N (16→2048), D independent vs coupled.
- Per-step Gumbel ablation (iters ∝ T) vs windowed (iters O(1)).
- **Todo**: D-sweep and coupled-D on the Gumbel filter (needs full-Jacobian
  solver inside `TwistedGumbelParticleFilter`); re-run everything through one
  harness that writes CSV + figure. ~1 week.

### E2. Gradient fidelity (LGSSM)  — NEW, critical, cheap
The MLE and twist-learning claims rest on ∇θ log Ẑ through STE resampling
being usable. On LGSSM the exact ∇θ log p(y|θ) is available from the Kalman
filter.
- Compare, across θ grid and seeds: (a) STE windowed-Gumbel gradient,
  (b) stop-gradient / "ignore resampling" gradient (Ścibior & Wood 2021,
  FIVO default), (c) no-resampling twisted PF gradient (exact reparam but
  degenerate weights), (d) finite differences on the frozen tape,
  (e) Kalman truth.
- Report bias and variance vs K, N, twist quality, τ (STE temperature).
- Hypothesis to confirm: with a good twist, resampling is rare and
  low-information, so STE bias is small; bias grows as twist worsens.
- Deliverable: one figure + a paragraph that pre-empts the "biased gradient"
  review. ~1 week.

### E3. Smoothing / filtering quality vs VI and SMC baselines (nonlinear)  — partial
Models: SV (T=1k–10k), Lorenz-63 (D=3), Lorenz-96 (D=10/20/40), PLDS (D=5–20).
Baselines: bootstrap PF, FFBSi, EKF/UKF-RTS, **structured Gaussian VI**
(parallel Kalman-form ELBO; cite Hassan et al. 2022 parallel VI), **dSMC**,
sequential twisted PF.
Metrics: RMSE to truth, posterior NLL / 90% coverage, log Ẑ, wall-clock at
matched N; ESS per window.
- **Todo**: write VI baseline (structured Gaussian, parallel scan), wrap dSMC,
  port Lorenz-96/PLDS to Gumbel filter with full-Jacobian solver, fix
  Lorenz-63 weight degeneracy (ESS/N = 1/N at N=64 is unacceptable; raise N
  or improve twist). ~3 weeks.

### E4. Twist learning = parallel VSMC  — partial, currently negative
- Train twists by Adam on (a) no-resampling ELBO (current), (b) windowed
  Gumbel log Ẑ with STE gradients (new; this is the paper's VSMC objective),
  (c) BIF-target regression (amortised warm start).
- Measure deployment: ESS, RMSE, DEER iters, speedup. Need to show (b) does
  not degrade DEER convergence the way (a) did on Lorenz-63.
- ~2 weeks.

### E5. MLE by autodiff  — partial
- Direct gradient ascent on log Ẑ(θ) with frozen or refreshed tape, Adam;
  compare to EM (FFBSi E-step, Gumbel E-step), and to VI ELBO ascent.
- Models: SV (φ, σ_q, β), PLDS (A, σ_q), Lorenz-63 (σ, ρ, β or just noise
  scales). Report parameter recovery, iterations, wall-clock.
- Existing EM-on-SV results become a baseline row.
- ~2 weeks.

### E6. Gradient-based particle MCMC  — partial
- M vmapped MALA chains, each with frozen-tape parallel PF likelihood; tape
  refreshed between blocks (correlated pseudo-marginal).
- Sample full θ on SV (3 params) and PLDS, not just φ.
- Baselines: PMMH random-walk with bootstrap PF (sequential), MALA with
  sequential twisted PF, PMMH with dSMC likelihood.
- Headline metric: ESS per second at matched GPU; R-hat across chains;
  posterior vs long reference run. Fix the stuck-chain issue (N=64 too
  small; use N≥256 and overdispersed inits).
- Do not revive single-chain DEER-over-MALA (negative result; optional
  appendix sentence).
- ~2 weeks.

### E7. One real dataset  — new
- SV on daily equity/FX returns (classic), or PLDS on neural spike counts.
  Needed for ICML credibility; reuse E5/E6 pipeline. ~1 week.

## 4. Infrastructure before experiments
1. `src/models/{lgssm,stochvol,lorenz63,lorenz96,plds}.py`: simulate,
   twisted_step_fn, log_weight_fn, analytic/BIF twists. Mine from the
   remaining `examples/run_twisted_pf_*.py` and `run_particle_em_*.py`, then
   delete those scripts.
2. Full-Jacobian DEER solver as a `solver=` option in
   `TwistedGumbelParticleFilter` (move `seq1d_fulljac` from the Lorenz-96
   script into `src/deer.py`).
3. `src/baselines.py`: add structured Gaussian VI and a dSMC wrapper.
4. `experiments/` with one script per E-number writing CSV; `figures/`
   regenerated from CSV (keep `examples/figures/paper_style.py`).

## 5. Risks and how to pre-empt them
- **dSMC exists** and is O(log T) with no twist needed. Must compare on
  wall-clock and RMSE, and argue what dSMC lacks: differentiability through
  its forward pass, gradient-based MCMC, and per-window exactness relative
  to a sequential twisted PF.
- **STE gradient bias** (Concrete PF, Younis & Sudderth 2023, is the
  precedent). E2 settles it empirically; theory paragraph: forward law is
  exact so Ẑ is unbiased regardless; only ∇ is biased.
- **Speedup ceiling** is the serial scan over T/K windows; speedups below
  ~5× at K≤64. Present the K trade-off honestly (ESS vs speed).
- **"vmap over chains is trivial"**: the MCMC contribution is gradient
  access plus cheap per-step likelihood, so report ESS/sec vs PMMH, not
  chain-throughput vs one chain.
- **Lorenz-63 twist degeneracy** undermines the "beats bootstrap" claim;
  fix before showing it.

## 6. Timeline (weeks from 2026-10-13)
- W1–2: models refactor, full-Jac solver, E2.
- W3–5: VI + dSMC baselines, E3 (SV, L63, L96, PLDS).
- W6–7: E4 twist learning.
- W8–9: E5 MLE.
- W10–11: E6 pMCMC full-θ, ESS/sec.
- W12: E7 real data, figures.
- W13: writing/buffer. Draft text in Overleaf throughout; the old NeurIPS
  draft's background and method sections are reusable after re-framing.
