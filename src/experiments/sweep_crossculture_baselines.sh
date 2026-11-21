#!/usr/bin/env bash
# Strict SVM and MLP baselines on Chinese, German and French EEG in the two
# subject-independent settings of Liu et al. (J. Neural Eng. 2022):
#   pooled : leave one subject out within a culture, all sessions (their ICSI)
#   cross  : train on one culture, test every subject of another (their CCSI)
# Chinese is the 12-subject multimodal subset (SEED_CHN_MM) the paper used;
# CULTURES="SEED_CHN SEED_GER SEED_FRA" gives the 15-subject SEED check.
# NORM=recording standardises each recording with its own statistics (label-free).
# WINDOW=1s uses the 1-second features (the paper's EEG-only setting).
# MODALITY=_EYE or _EEG_EYE uses eye-movement or EEG+eye features (4 s only).
# MODELS="svm mlp" by default; add linsvm for the paper's linear SVM, mlp_st for
# normalise-then-adapt self-training. SEEDS="12" limits the seeds.
# Finished jobs leave a .complete marker and are skipped on reruns; an
# interrupted job has no marker and is rerun from scratch.
# Results: results/crossculture/<target>/<model>[__subjnorm]{__pooled,__from_<source>}/
set -euo pipefail
NORM="${NORM:-none}"
CULTURES="${CULTURES:-SEED_CHN_MM SEED_GER SEED_FRA}"
MODELS="${MODELS:-svm mlp}"
WIN=$([ "${WINDOW:-4s}" = 1s ] && echo "_1s" || echo "")
MOD="${MODALITY:-}"
PY="${PY:-./.venv/bin/python}"
DATA=data/processed
OUT=results/crossculture
LOG_DIR=results/logs/crossculture
mkdir -p "$LOG_DIR"
NORM_TAG=$([ "$NORM" = none ] && echo "" || echo "__subjnorm")

run() {  # run <model> <target> <run-suffix> <log-name> <extra args...>
  local model=$1 target=$2 suffix=$3 log=$4; shift 4
  local dir="$OUT/$target/$model$NORM_TAG$suffix"
  if [ -f "$dir/.complete" ]; then
    echo "skip $model$NORM_TAG$suffix -> $target"; return
  fi
  rm -rf "$dir"
  # shellcheck disable=SC2086
  "$PY" -m src.baselines.strict_baselines --model "$model" --subject-norm "$NORM" ${SEEDS:+--seeds $SEEDS} \
    --data-dir "$DATA/$target" --output-root "$OUT/$target" "$@" > "$LOG_DIR/$log.log" 2>&1
  touch "$dir/.complete"
  echo "done $model$NORM_TAG$suffix -> $target"
}

for model in $MODELS; do
  for culture in $CULTURES; do
    target="$culture$MOD$WIN"
    run "$model" "$target" "__pooled" "${model}__${NORM}__${target}__pooled" --split pooled
    for other in $CULTURES; do
      [ "$other" = "$culture" ] && continue
      source="$other$MOD$WIN"
      run "$model" "$target" "__from_$source" "${model}__${NORM}__${source}_to_${target}" \
        --split cross --source-data-dir "$DATA/$source"
    done
  done
done
