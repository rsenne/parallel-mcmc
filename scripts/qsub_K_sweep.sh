#!/bin/bash -l
#
# Window-size (K) sweep for twisted-Gumbel parallel PF.
# LGSSM (T=8192) and Lorenz63 (T=4096), single L40S.
#
# Submit:  qsub scripts/qsub_K_sweep.sh
# Monitor: qstat -u $USER ; tail -f logs/K_sweep.o<JOB_ID>

#$ -N tgpf_K_sweep
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=01:00:00
#$ -o logs/K_sweep.o$JOB_ID
#$ -e logs/K_sweep.e$JOB_ID
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
echo "================ LGSSM K-sweep (T=8192) ================"
PYTHONUNBUFFERED=1 uv run python -u examples/run_twisted_gumbel_pf_K_sweep_lgssm.py

echo
echo "================ Lorenz63 K-sweep (T=4096) ================"
PYTHONUNBUFFERED=1 uv run python -u examples/run_twisted_gumbel_pf_K_sweep_lorenz63.py

echo "=========================================="
echo "Finished:  $(date)"
echo "=========================================="
