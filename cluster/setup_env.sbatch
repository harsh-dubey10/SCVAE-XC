#!/usr/bin/env bash
# One-off environment setup for the IITGN Singularity cluster, run as a job
# (login-node work is not allowed):
#
#   sbatch cluster/setup_env.sbatch
#
# Creates the conda env "scvae" with CUDA PyTorch, NumPy, SciPy and
# scikit-learn using the cluster's anaconda3-2024 module, then checks that
# PyTorch sees the GPU. Logs: results/logs/slurm/setup_<jobid>.out and .err.
# If compute nodes have no internet access the pip step fails; then run the
# same commands on the login node (installing packages is not a compute job).
#
#SBATCH --job-name=setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/slurm/%x_%j.out
#SBATCH --error=results/logs/slurm/%x_%j.err
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "Setup on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx scvae; then
  conda create -y -n scvae python=3.11
fi
conda activate scvae
pip install --upgrade pip
# cu124 wheels run on the L40S with any driver >= 550.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "numpy<3" scipy scikit-learn joblib
python -c "import torch, sklearn, scipy; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
