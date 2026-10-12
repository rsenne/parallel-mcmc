#!/bin/bash -l
#
# STAMPEDE multi-chain: vmap M independent MALA chains.
# Each chain uses sequential MALA with parallel PF inside log_prob.
#
# Submit:  qsub scripts/qsub_stampede_multichain.sh

#$ -N stampede_multichain
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=01:00:00
#$ -o logs/stampede_multichain.o$JOB_ID
#$ -e logs/stampede_multichain.e$JOB_ID
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

NVIDIA_LIB=$(echo .venv/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH}"

PYTHONUNBUFFERED=1 uv run python -c "import jax; print('JAX devices:', jax.devices())"
echo
PYTHONUNBUFFERED=1 uv run python -u examples/run_stampede_multichain.py

echo "=========================================="
echo "Finished:  $(date)"
echo "=========================================="
