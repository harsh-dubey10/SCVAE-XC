#!/usr/bin/env bash
# Strict sweep: standard vs uncertainty-weighted MMD, all with source-only
# BatchNorm recalibration. One fold per job; JOBS folds run concurrently.
set -euo pipefail
JOBS="${JOBS:-3}"
PY="${PY:-./.venv/bin/python}"
LOG_DIR=results/logs/bn_sweep
mkdir -p "$LOG_DIR"

for alignment in standard_mmd entropy_weighted_mmd posterior_weighted_mmd combined_uncertainty_mmd; do
  for session in 1 2 3; do
    for seed in 12 34 56; do
      for subject in $(seq 1 15); do
        echo "$alignment $session $seed $subject"
      done
    done
  done
done | xargs -P "$JOBS" -L 1 sh -c '
  log="'"$LOG_DIR"'/$0__s$1__seed$2__subj$3.log"
  if [ -f "$log" ] && grep -q "^Saved checkpoint" "$log"; then exit 0; fi
  '"$PY"' -m src.experiments.train_scvae_reference \
    --protocol strict --alignment "$0" --bn-recalibration source \
    --session "$1" --seed "$2" --test-subject "$3" --epochs 300 --log-every 100 \
    > "$log" 2>&1 || echo "FAILED $0 $1 $2 $3" >> "'"$LOG_DIR"'/failures.txt"
'
echo "sweep finished"
