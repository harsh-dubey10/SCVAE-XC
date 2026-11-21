#!/usr/bin/env bash
# SCVAE-XC + v3 combination: task i caches v3's fair probabilities (5 seeds) for run_xc_v3_combination.ORDER[i]
# (0-2 two cultures, 3-8 one culture). CPU only; the GPU is requested because the queue requires --gres=gpu:1.
#
#   sbatch cluster/xc_v3_combination.sbatch
#
#SBATCH --job-name=xcv3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-8%2
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"${PYTHON[@]}" -m src.experiments.run_xc_v3_combination --task "${SLURM_ARRAY_TASK_ID}" --jobs "${SLURM_CPUS_PER_TASK}"
