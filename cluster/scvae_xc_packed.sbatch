#!/usr/bin/env bash
# SCVAE-XC trainer (src/experiments/train_scvae_xc.py), PACK manifest lines side by side on one GPU, like
# train_packed.sbatch. Each line logs to results/logs/slurm/<job>_<task>_line<k>.log.
#
#   sbatch --export=ALL,MANIFEST=cluster/scvae_xc_stage2.txt,PACK=6 --array=1-9 cluster/scvae_xc_packed.sbatch
#
#SBATCH --job-name=scvaexc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:?set MANIFEST}"
PACK="${PACK:-6}"
source cluster/env.sh
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$PACK} / PACK ))
export OMP_NUM_THREADS=$(( THREADS > 0 ? THREADS : 1 ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mapfile -t LINES < <(grep -v '^\s*\(#\|$\)' "$MANIFEST")
FIRST=$(( (SLURM_ARRAY_TASK_ID - 1) * PACK ))
LOG_PREFIX="results/logs/slurm/${SLURM_JOB_NAME:-scvaexc}_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
PIDS=()
for (( k = FIRST; k < FIRST + PACK && k < ${#LINES[@]}; k++ )); do
  echo "Task ${SLURM_ARRAY_TASK_ID} line $(( k + 1 )) on $(hostname): ${LINES[$k]}"
  # shellcheck disable=SC2086
  "${PYTHON[@]}" -m src.experiments.train_scvae_xc --device cuda ${LINES[$k]} > "${LOG_PREFIX}_line$(( k + 1 )).log" 2>&1 &
  PIDS+=("$!")
done
STATUS=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "Line $(( FIRST + i + 1 )) failed" >&2; STATUS=1; }
done
exit "$STATUS"
