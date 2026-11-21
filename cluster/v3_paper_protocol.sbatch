#!/usr/bin/env bash
# NormFusion v3 under the papers' test-label-selected protocol: task i runs run_v3_paper_protocol.TASKS[i]
# (seeds 12, 34, 56). All computation is on CPU; the GPU is requested only because the queue requires
# --gres=gpu:1.
#
#   sbatch cluster/v3_paper_protocol.sbatch
#
#SBATCH --job-name=v3paper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs"

"${PYTHON[@]}" -m src.experiments.run_v3_paper_protocol --task "${SLURM_ARRAY_TASK_ID}" \
    --jobs "${SLURM_CPUS_PER_TASK}"
