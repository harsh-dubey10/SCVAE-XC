#!/usr/bin/env bash
# SLURM array job like train_array.sbatch, but each array task runs PACK manifest
# lines side by side on its one GPU (SCVAE uses a fraction of an L40S). The QOS
# allows 2 GPUs / 32 cores per user, so 2 tasks x PACK lines run at once.
#
#   N=$(grep -cv '^\s*\(#\|$\)' cluster/scvae_improve_films.txt)
#   sbatch --export=ALL,MANIFEST=cluster/scvae_improve_films.txt,PACK=4 \
#       --array=1-$(( (N + 3) / 4 )) cluster/train_packed.sbatch
#
# Each line logs to results/logs/slurm/<job>_<task>_line<k>.log (pulled by sync.sh).
#SBATCH --job-name=scvaep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-cluster/manifest.txt}"
PACK="${PACK:-4}"
DEVICE="${DEVICE:-cuda}"
GPU=1
source cluster/env.sh

# Share the allocated cores between the lines instead of each grabbing all of them.
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$PACK} / PACK ))
export OMP_NUM_THREADS=$(( THREADS > 0 ? THREADS : 1 ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mapfile -t LINES < <(grep -v '^\s*\(#\|$\)' "$MANIFEST")
FIRST=$(( (SLURM_ARRAY_TASK_ID - 1) * PACK ))
if [ "$FIRST" -ge "${#LINES[@]}" ]; then
  echo "No manifest lines for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 1
fi
LOG_PREFIX="results/logs/slurm/${SLURM_JOB_NAME:-scvaep}_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$(dirname "$LOG_PREFIX")"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

PIDS=()
for (( k = FIRST; k < FIRST + PACK && k < ${#LINES[@]}; k++ )); do
  ARGS="${LINES[$k]}"
  echo "Task ${SLURM_ARRAY_TASK_ID} line $(( k + 1 )) on $(hostname): ${ARGS}"
  # shellcheck disable=SC2086
  "${PYTHON[@]}" -m src.experiments.train_scvae_reference \
      --protocol strict --all-subjects --skip-existing --device "$DEVICE" \
      --epochs 300 --log-every 100 ${ARGS} > "${LOG_PREFIX}_line$(( k + 1 )).log" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "Line $(( FIRST + i + 1 )) failed; see ${LOG_PREFIX}_line$(( FIRST + i + 1 )).log" >&2
    STATUS=1
  fi
done
exit "$STATUS"
