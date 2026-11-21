#!/usr/bin/env bash
# Container-free fallback: create .venv-cluster with CUDA PyTorch. On IITGN
# Singularity run it through cluster/setup_env.sbatch (no login-node work). Run from the repo root. Load a Python module first if the cluster uses
# environment modules (e.g. `module load python/3.11`).
set -euo pipefail
VENV="${VENV:-.venv-cluster}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
# cu124 wheels run on L40S with any driver >= 550.
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124
"$VENV/bin/pip" install "numpy<3" scipy scikit-learn
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
