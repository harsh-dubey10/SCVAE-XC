#!/usr/bin/env bash
# DCCA-AM cross-culture re-run: task i runs line i of cluster/dcca_sweep.txt
# (all architectures 0-20, 100 epochs; finished architectures are skipped).
#
#   sbatch --array=1-$(grep -cv '^\s*\(#\|$\)' cluster/dcca_sweep.txt) cluster/dcca_array.sbatch
#
#SBATCH --job-name=dcca
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-cluster/dcca_sweep.txt}"
GPU=1
source cluster/env.sh

ARGS="$(grep -v '^\s*\(#\|$\)' "$MANIFEST" | sed -n "${SLURM_ARRAY_TASK_ID}p")"
if [ -z "$ARGS" ]; then
  echo "No manifest line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 1
fi
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${ARGS}"

# shellcheck disable=SC2086
"${PYTHON[@]}" -m src.experiments.train_dcca_am \
    --archs 0-20 --epochs 100 --device "${DEVICE:-cuda}" --skip-existing ${ARGS}
