#!/usr/bin/env bash
# Mechanism test for strict-fold collapse: no post-hoc BN recalibration.
#   released           : released training (control; same-sweep rerun)
#   target_bn_off      : target passes do not update BN running stats
#   lr_cosine          : cosine learning-rate decay (weights stop drifting)
# Folds: the 10 collapsed + 9 healthy folds of the frozen 4 s baseline.
set -euo pipefail
JOBS="${JOBS:-3}"
PY="${PY:-./.venv/bin/python}"
LOG_DIR=results/logs/bn_mechanism
mkdir -p "$LOG_DIR"

for variant in released target_bn_off lr_cosine; do
  grep -v '^#' src/experiments/bn_mechanism_folds.txt | while read -r session seed subject _; do
    echo "$variant $session $seed $subject"
  done
done | xargs -P "$JOBS" -L 1 sh -c '
  case "$0" in
    released) extra="" ;;
    target_bn_off) extra="--target-bn-update off" ;;
    lr_cosine) extra="--lr-schedule cosine" ;;
  esac
  log="'"$LOG_DIR"'/$0__s$1__seed$2__subj$3.log"
  if [ -f "$log" ] && grep -q "^Saved checkpoint" "$log"; then exit 0; fi
  '"$PY"' -m src.experiments.train_scvae_reference \
    --protocol strict $extra \
    --session "$1" --seed "$2" --test-subject "$3" --epochs 300 --log-every 100 \
    --output-dir results/bn_mechanism --checkpoint-dir checkpoints/bn_mechanism \
    > "$log" 2>&1 || echo "FAILED $0 $1 $2 $3" >> "'"$LOG_DIR"'/failures.txt"
'
echo "sweep finished"
