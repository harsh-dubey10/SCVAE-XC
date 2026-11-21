#!/usr/bin/env bash
# Run cluster manifest lines on this machine with the same fixed arguments as
# train_array.sbatch. Folds with an existing result JSON are skipped, so an
# interrupted run resumes and the cluster later skips folds finished here.
#
#   cluster/run_local.sh cluster/crossculture_sweep.txt "--seed 12" \
#       "target_culture" "source --subject-norm" "source --seed" "bn-recalibration target "
#
# Arg 1: manifest. Arg 2: only lines containing this text. Remaining args:
# priority patterns; lines matching the first pattern run first, and so on
# (unmatched lines run last).
set -euo pipefail
MANIFEST=$1; FILTER=$2; shift 2
PY="${PY:-./.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
LOG_DIR=results/logs/local_runs
mkdir -p "$LOG_DIR"

lines=$(grep -v '^\s*\(#\|$\)' "$MANIFEST" | grep -F -- "$FILTER")
ordered=""
for pattern in "$@"; do
  matched=$(grep -F -- "$pattern" <<< "$lines" || true)
  ordered+="$matched"$'\n'
  lines=$(grep -v -F -- "$pattern" <<< "$lines" || true)
done
ordered+="$lines"

total=$(grep -c . <<< "$ordered" || true)
i=0
while IFS= read -r args; do
  [ -z "$args" ] && continue
  i=$((i + 1))
  log="$LOG_DIR/$(echo "$args" | tr -cs 'A-Za-z0-9' '_' | cut -c1-150).log"
  echo "[$(date +%H:%M)] task $i/$total: $args"
  # shellcheck disable=SC2086
  "$PY" -m src.experiments.train_scvae_reference \
      --protocol strict --all-subjects --skip-existing --device "$DEVICE" \
      --epochs 300 --log-every 100 $args >> "$log" 2>&1 \
    || echo "[$(date +%H:%M)] FAILED task $i: $args (see $log)"
done <<< "$ordered"
echo "[$(date +%H:%M)] all $total tasks finished"
