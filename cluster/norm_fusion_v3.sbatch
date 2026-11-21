#!/usr/bin/env bash
# NormFusion v3: task i runs setting i of run_norm_fusion_v3.SETTINGS (0..10), all 5 seeds one after another,
# with the fits of each stage spread over the task's CPUs. Everything computes on CPU; the GPU is requested only
# because the queue requires --gres=gpu:1. At most 2 tasks run at once (QOS: 32 CPUs, 2 GPUs per user).
#
#   sbatch cluster/norm_fusion_v3.sbatch
#
#SBATCH --job-name=nf3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-10%2
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs"

"${PYTHON[@]}" -m src.experiments.run_norm_fusion_v3 --task "${SLURM_ARRAY_TASK_ID}" \
    --jobs "${SLURM_CPUS_PER_TASK}" --skip-existing
