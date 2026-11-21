#!/usr/bin/env bash
# SCVAE-XC within-SEED runs (run_scvae_xc_within.py) on one GPU, 4 runs side by side.
# Task 1: cross-subject session 3; task 2: cross-session sessions 1-2; task 3: cross-session session 3;
# task 4: cross-subject sessions 1-2.
#
#   sbatch --array=1-3 cluster/scvae_xc_within.sbatch
#
#SBATCH --job-name=xcwithin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh
case "${SLURM_ARRAY_TASK_ID}" in
  1) ARGS=(--settings cross_subject --sessions 3) ;;
  2) ARGS=(--settings cross_session --sessions 1 2) ;;
  3) ARGS=(--settings cross_session --sessions 3) ;;
  4) ARGS=(--settings cross_subject --sessions 1 2) ;;
esac
"${PYTHON[@]}" -m src.experiments.run_scvae_xc_within --jobs 4 --device cuda "${ARGS[@]}"
