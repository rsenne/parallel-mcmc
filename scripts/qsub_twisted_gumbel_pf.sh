#!/bin/bash -l
#
# Twisted-Gumbel parallel particle filter: LGSSM correctness demo +
# wall-clock scaling sweep.
#
# Submit:  qsub scripts/qsub_twisted_gumbel_pf.sh
# Monitor: qstat -u $USER  ;  tail -f logs/twisted_gumbel_pf.o<JOB_ID>

#$ -N twisted_gumbel_pf
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=01:00:00
#$ -o logs/twisted_gumbel_pf.o$JOB_ID
#$ -e logs/twisted_gumbel_pf.e$JOB_ID
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

# JAX 0.9.x bundles its own CUDA via the nvidia-* wheels; prepend them
# so they take precedence over the system cuda/12.8 module.
NVIDIA_LIB=$(echo .venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH}"

PYTHONUNBUFFERED=1 uv run python -c \
  "import jax; print('JAX devices:', jax.devices())"

echo
echo "--- LGSSM correctness demo (T=1024) ---"
PYTHONUNBUFFERED=1 uv run python -u examples/run_twisted_gumbel_pf_lgssm.py

echo
echo "--- Wall-clock scaling sweep ---"
PYTHONUNBUFFERED=1 uv run python -u examples/run_twisted_gumbel_pf_scaling.py

echo
echo "--- Stochastic-volatility stress test (T=1024) ---"
PYTHONUNBUFFERED=1 uv run python -u examples/run_twisted_gumbel_pf_stochvol.py

echo "=========================================="
echo "Finished:  $(date)"
echo "=========================================="
