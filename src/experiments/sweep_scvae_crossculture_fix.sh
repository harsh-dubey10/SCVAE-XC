#!/usr/bin/env bash
# Which SCVAE-Net settings transfer across cultures? The released recipe
# (MMD 0.1, 300 epochs) swaps emotions across cultures, so this grid varies
# the MMD weight and training length on all six cross-culture directions,
# with per-recording normalisation and source BN recalibration.
#
# Settings for a target culture must be chosen from the transfers between the
# OTHER two cultures only (see select_crossculture_config.py); reading the
# target culture's own accuracies here would be test-label selection.
#
# Results: results/scvae_fix/ep<E>/<target>/<run>/session_all/seed_<S>/
set -euo pipefail
PY="${PY:-./.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-12}"
EPOCHS="${EPOCHS:-30 100}"
MMD_WEIGHTS="${MMD_WEIGHTS:-0 0.01 0.1}"
CULTURES="SEED_CHN_MM SEED_GER SEED_FRA"
LOG_DIR=results/logs/scvae_fix
mkdir -p "$LOG_DIR"

for epochs in $EPOCHS; do
  for mmd in $MMD_WEIGHTS; do
    for target in $CULTURES; do
      for source in $CULTURES; do
        [ "$source" = "$target" ] && continue
        log="$LOG_DIR/ep${epochs}__mmd${mmd}__${source}_to_${target}__seed${SEED}.log"
        echo "[$(date +%H:%M)] epochs $epochs mmd $mmd $source -> $target"
        "$PY" -m src.experiments.train_scvae_reference --protocol strict --split cross \
          --source-data-dir "data/processed/$source" --data-dir "data/processed/$target" \
          --all-subjects --skip-existing --device "$DEVICE" --seed "$SEED" \
          --epochs "$epochs" --log-every "$epochs" --mmd-weight "$mmd" \
          --bn-recalibration source --subject-norm recording \
          --output-dir "results/scvae_fix/ep$epochs/$target" \
          --checkpoint-dir "checkpoints/scvae_fix/ep$epochs/$target" > "$log" 2>&1 \
          || echo "[$(date +%H:%M)] FAILED: see $log"
      done
    done
  done
done
echo "[$(date +%H:%M)] grid finished"
