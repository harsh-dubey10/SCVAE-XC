#!/usr/bin/env bash
# SLURM CPU job: runs one Python module with its arguments, e.g. the strict
# baselines or the leakage audit, which need many CPU cores and no GPU.
#
#   sbatch cluster/cpu_task.sbatch src.baselines.liu_leakage_audit --cultures SEED_CHN_MM SEED_GER SEED_FRA
#   sbatch cluster/cpu_task.sbatch src.baselines.strict_baselines --model svm --split pooled \
#       --subject-norm recording --data-dir data/processed/SEED_GER --output-root results/crossculture/SEED_GER
#
#SBATCH --job-name=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/slurm/%x_%j.out
#SBATCH --error=results/logs/slurm/%x_%j.err
# Partition "gpu" (the only one in the user guide). Its QOS rejects jobs without a GPU
# (QOSMinGRES), so this CPU job also holds one of the 2 GPUs a user may use.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source cluster/env.sh
[ $# -ge 1 ] || { echo "usage: sbatch $0 <python.module> [args...]" >&2; exit 2; }

echo "CPU job on $(hostname) with ${SLURM_CPUS_PER_TASK:-?} cores: $*"
"${PYTHON[@]}" -m "$@"
