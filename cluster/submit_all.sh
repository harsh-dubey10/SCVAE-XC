#!/usr/bin/env bash
# Submit every pending experiment to SLURM in one go. Run on the cluster from
# the project root after `cluster/sync.sh push` and building cluster/scvae.sif.
#
#   cluster/submit_all.sh          # submit everything
#   DRY_RUN=1 cluster/submit_all.sh  # print the sbatch commands only
#
# GPU arrays (train_array.sbatch skips finished folds, so resubmitting is safe):
#   paper_sweep.txt  54 tasks  SCVAE-Net within/across cultures, 3 seeds
#   main_sweep.txt   54 tasks  SEED BN-staleness headline rows, 3 seeds
# IITGN Singularity allows 2 GPUs per user, so at most 2 GPU tasks run at a time;
# the rest wait in the queue. Pull results often: home files untouched for 21
# days are deleted.
# CPU jobs (cpu_task.sbatch; the audit skips settings already in its JSON):
#   leakage audit, 1 s windows + full C grid (closest to Liu et al.)
#   leakage audit, 4 s windows + full C grid
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs/slurm

count() { grep -cv '^\s*\(#\|$\)' "$1"; }
submit() {
  echo "+ sbatch $*"
  [ -n "${DRY_RUN:-}" ] || sbatch "$@"
}

submit --export=ALL,MANIFEST=cluster/paper_sweep.txt --array=1-"$(count cluster/paper_sweep.txt)" cluster/train_array.sbatch
submit --export=ALL,MANIFEST=cluster/main_sweep.txt --array=1-"$(count cluster/main_sweep.txt)" cluster/train_array.sbatch
submit --job-name=audit1s cluster/cpu_task.sbatch src.baselines.liu_leakage_audit \
  --window 1s --grid full --cultures SEED_CHN_MM SEED_GER SEED_FRA
submit --job-name=audit4s cluster/cpu_task.sbatch src.baselines.liu_leakage_audit \
  --window 4s --grid full --cultures SEED_CHN_MM SEED_GER SEED_FRA

echo "Check the queue with: squeue -u \$USER"
