#!/bin/bash -l
#
# Wall-clock scaling sweep for Gumbel-STE parallel particle filter.
#
# Submit:  qsub scripts/qsub_gumbel_pf_scaling.sh
# Monitor: qstat -u $USER  ;  tail -f logs/gumbel_pf_scaling.o<JOB_ID>

#$ -N gumbel_pf_scaling
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=01:00:00
#$ -o logs/gumbel_pf_scaling.o$JOB_ID
#$ -e logs/gumbel_pf_scaling.e$JOB_ID
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

PYTHONUNBUFFERED=1 uv run python -c \
  "import jax; print('JAX devices:', jax.devices())"

PYTHONUNBUFFERED=1 uv run python -u examples/run_gumbel_pf_scaling.py

echo "=========================================="
echo "Finished:  $(date)"
echo "=========================================="
