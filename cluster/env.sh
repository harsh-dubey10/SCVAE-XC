# Sourced by the SLURM scripts: sets PYTHON (an array) to the interpreter to use.
# Preference: the conda env "scvae" (the IITGN Singularity guide's supported way:
# module load anaconda3-2024), then a container image, then .venv-cluster.
# GPU=1 adds --nv when a container is used.
# module/conda scripts often fail under `set -u`; relax it while loading.
case $- in *u*) _env_restore_u=1; set +u ;; *) _env_restore_u= ;; esac
CONDA_ENV="${CONDA_ENV:-scvae}"
IMAGE="${IMAGE:-cluster/scvae.sif}"
VENV="${VENV:-.venv-cluster}"
PYTHON=()

if command -v module >/dev/null 2>&1; then
  module load anaconda3-2024 >/dev/null 2>&1 || true
fi
CONDA_SH=/apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda activate "$CONDA_ENV"
    PYTHON=(python)
  fi
fi

if [ ${#PYTHON[@]} -eq 0 ]; then
  RUNTIME="$(command -v singularity || command -v apptainer || true)"
  if [ -n "$RUNTIME" ] && [ -f "$IMAGE" ]; then
    PYTHON=("$RUNTIME" exec ${GPU:+--nv} --bind "$PWD" --pwd "$PWD" "$IMAGE" python)
  elif [ -x "$VENV/bin/python" ]; then
    PYTHON=("$VENV/bin/python")
  fi
fi

[ -n "$_env_restore_u" ] && set -u
if [ ${#PYTHON[@]} -eq 0 ]; then
  echo "No Python environment found: run 'sbatch cluster/setup_env.sbatch' first." >&2
  exit 1
fi
echo "Python: ${PYTHON[*]}"
