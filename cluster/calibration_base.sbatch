#!/usr/bin/env bash
# Calibration study base runs: task i runs run_calibration_base.TASKS[i] (seeds 12, 34, 56). All computation is
# on CPU; the GPU is requested only because the queue requires --gres=gpu:1. Default: all 15 tasks.
#
#   sbatch cluster/calibration_base.sbatch                  # all 15 tasks
#   sbatch --array=6-14%2 cluster/calibration_base.sbatch   # the same-culture tasks only
#
#SBATCH --job-name=calbase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-14%2
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs"

"${PYTHON[@]}" -m src.experiments.run_calibration_base --task "${SLURM_ARRAY_TASK_ID}" \
    --jobs "${SLURM_CPUS_PER_TASK}" --skip-existing
