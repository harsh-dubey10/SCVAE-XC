#!/bin/bash
# Single GPU job (admin template layout): trains all held-out subjects for one
# configuration. Extra arguments go to src.experiments.train_scvae_reference, e.g.
#
#   sbatch cluster/run_job.sh --session 1 --seed 12
#   sbatch cluster/run_job.sh --split pooled --data-dir data/processed/SEED_GER
#
# Default when no arguments are given: --session 1. For sweeps use train_array.sbatch.
#SBATCH --job-name=scvae_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=30:00:00
#SBATCH --output=results/logs/slurm/%x_%j.log
#SBATCH --error=results/logs/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs/slurm

echo "Running on node: $(hostname)"
nvidia-smi

GPU=1
source cluster/env.sh

[ $# -ge 1 ] || set -- --session 1
"${PYTHON[@]}" -m src.experiments.train_scvae_reference \
    --protocol strict --all-subjects --skip-existing --device cuda "$@"
