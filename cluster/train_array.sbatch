#!/usr/bin/env bash
# SLURM array job: task i runs manifest line i for all 15 held-out subjects.
#
#   sbatch --array=1-$(grep -cv '^\s*\(#\|$\)' cluster/manifest.txt) cluster/train_array.sbatch
#
# Partition set for the IITGN Singularity cluster (no account needed).
#SBATCH --job-name=scvae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
# IITGN Singularity user guide: partition "gpu" (QOS 32c2g5dw: 2 GPUs / 32 cores per user, 5 days).
#SBATCH --partition=gpu
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-cluster/manifest.txt}"
GPU=1
source cluster/env.sh

ARGS="$(grep -v '^\s*\(#\|$\)' "$MANIFEST" | sed -n "${SLURM_ARRAY_TASK_ID}p")"
if [ -z "$ARGS" ]; then
  echo "No manifest line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 1
fi
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${ARGS}"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
"${PYTHON[@]}" -m src.experiments.train_scvae_reference \
    --protocol strict --all-subjects --skip-existing --device cuda \
    --epochs 300 --log-every 100 ${ARGS}
