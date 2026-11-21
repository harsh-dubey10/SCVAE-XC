#!/usr/bin/env bash
# Sync the project with the cluster over SSH.
#
#   cluster/sync.sh push   # code + processed data + raw DE features (leakage audit) -> cluster
#   cluster/sync.sh pull   # results (JSON/CSV) and SLURM logs (.out/.log/.err) <- cluster
#
# CLUSTER is an SSH alias from ~/.ssh/config (HostName <cluster-host>, Port <port>, as in the
# user guide), or pass it directly: CLUSTER=<username>@<cluster-host> cluster/sync.sh push.
# REMOTE_DIR is the project path
# on the cluster (relative paths are relative to the remote home directory).
set -euo pipefail
CLUSTER="${CLUSTER:-singularity}"   # SSH alias for the cluster
REMOTE_DIR="${REMOTE_DIR:-EEG-Emotion}"
cd "$(dirname "$0")/.."

case "${1:-}" in
  push)
    ssh "$CLUSTER" "mkdir -p '$REMOTE_DIR/results/logs/slurm'"
    rsync -az \
      --exclude="*.sif" --exclude="__pycache__/" --exclude=".venv*/" \
      --include='src/***' --include='cluster/***' --include='README.md' \
      --exclude='data/processed/SEED_CHN' \
      --include='data/' --include='data/processed/***' \
      --include='data/raw/' --include='data/raw/*/' \
      --include='data/raw/*/02-EEG-DE-feature*/***' \
      --include='results/' --include='results/reference_scvae/***' \
      --include='results/crossculture/' --include='results/crossculture/liu_leakage_audit*.json' \
      --exclude='*' \
      ./ "$CLUSTER:$REMOTE_DIR/"
    ;;
  pull)
    # Checkpoints stay on the cluster; fetch them explicitly if needed.
    # --update: never replace a file that is newer here (e.g. tables rebuilt locally).
    rsync -az --update \
      --include='*/' --include='*.json' --include='*.csv' --include='*.npz' --include='*.out' --include='*.log' --include='*.err' \
      --exclude='*' \
      "$CLUSTER:$REMOTE_DIR/results/" results/
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 2
    ;;
esac
