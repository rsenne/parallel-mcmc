#!/bin/bash -l
#
# Particle EM for stochastic volatility with theta-conditional NF twist.
# Compares twisted-flow EM (parallel DEER E-step) against bootstrap-PF EM
# and BPF+FFBSi EM (gold standard) on the same simulated data.
#
# Outputs:
#   examples/_em_stochvol_meta.npz     (cached meta-trained flow)
#   particle_em_stochvol.png/.svg      (recovery + wall-clock figure)
#
# Submit:  qsub scripts/qsub_em_stochvol.sh
# Monitor: qstat -u $USER  ;  tail -f logs/em_stochvol.o<JOB_ID>

#$ -N em_stochvol
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=03:00:00
#$ -o logs/em_stochvol.o$JOB_ID
#$ -e logs/em_stochvol.e$JOB_ID
#$ -j n
#$ -P ssmsvi

echo "=========================================="
echo "Job ID:    $JOB_ID"
echo "Host:      $(hostname)"
echo "Date:      $(date)"
echo "=========================================="

module load python3/3.11
module load cuda/12.8

cd /projectnb/ssmsvi/rsenne/parallel-mcmc || exit 1

# JAX 0.9.x ships its own CUDA 12.9 libraries via the nvidia-* wheels in
# .venv; the system 'cuda/12.8' module ships 12.8 libs that don't ABI-match
# the bundled jax-cuda12-plugin (cuSPARSE in particular).  Prepend the
# bundled paths so they take precedence -> JAX finds its libs and uses GPU.
NVIDIA_LIB=$(echo .venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH}"

# Surface the GPU/CPU choice in the log so silent fallback is noticeable.
PYTHONUNBUFFERED=1 uv run python -c \
  "import jax; print('JAX devices:', jax.devices())"

# Full-scale config — defaults are set inside run_particle_em_stochvol.py.
# Override here with `export SV_FOO=...` if needed.
echo "Config defaults from examples/run_particle_em_stochvol.py:"
echo "  T=5000 N=256 N_EM=60  flow=2-layer stacked tanh, H_GRU=64 H_MLP=128"
echo "  meta: N_META=32 M_FFBSI=128 N_TRAIN_STEPS=3000 batch=16"

PYTHONUNBUFFERED=1 uv run python -u examples/run_particle_em_stochvol.py

echo "=========================================="
echo "Finished:  $(date)"
echo "=========================================="
