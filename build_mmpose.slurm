#!/bin/bash --login
#SBATCH --job-name=build_mmpose
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/users/dbartaula/logs/build_mmpose_%j.log
#SBATCH --error=/users/dbartaula/logs/build_mmpose_%j_error.log

set -euo pipefail
trap 'ec=$?; echo "[ERROR] Command failed (exit ${ec}) at line ${LINENO}: ${BASH_COMMAND}"; exit ${ec}' ERR

# ---------------------------------------------------------------------------
# Build and install MMPose stack in a conda environment.
#
# Why source build:
# - On some Torch/CUDA combinations, prebuilt mmcv wheels are unavailable.
#
# Defaults target your current cluster usage but can be overridden:
#   sbatch --export=ALL,ENV_NAME=vision,CUDA_MODULE=cuda/12.6 build_mmpose.slurm
#
# Optional overrides:
#   ENV_NAME=vision
#   CUDA_MODULE=cuda/12.6
#   CUDA_HOME=/path/to/cuda
#   TORCH_CUDA_ARCH_LIST=9.0
#   MAX_JOBS=32
#   MMCV_VERSION=2.2.0
# ---------------------------------------------------------------------------

export TOKENIZERS_PARALLELISM=false

ENV_NAME="${ENV_NAME:-vision}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-32}}"
MMCV_VERSION="${MMCV_VERSION:-2.2.0}"

source /users/dbartaula/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Load CUDA toolkit if module system exists.
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load "$CUDA_MODULE"
fi

# Auto-detect CUDA_HOME if not provided.
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  else
    echo "[ERROR] nvcc not found and CUDA_HOME is not set."
    echo "        Set CUDA_HOME explicitly via sbatch --export=ALL,CUDA_HOME=/path/to/cuda"
    exit 1
  fi
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export MAX_JOBS

echo "================================================================="
echo "Building MMPose stack"
echo "  Host                : $(hostname)"
echo "  Conda env           : ${CONDA_DEFAULT_ENV:-<none>}"
echo "  Python              : $(which python)"
echo "  CUDA module         : $CUDA_MODULE"
echo "  CUDA_HOME           : $CUDA_HOME"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS            : $MAX_JOBS"
echo "  MMCV_VERSION        : $MMCV_VERSION"
echo "================================================================="

python -m pip install -U pip
python -m pip install -U "setuptools<81" wheel packaging cython ninja cmake
python -m pip install -U openmim mmengine

# Build mmcv from source against the current torch+cuda environment.
python -m pip install --no-build-isolation -v "mmcv==${MMCV_VERSION}"

# Install matching OpenMMLab runtime packages.
python -m pip install "mmdet>=3.1.0" "mmpose>=1.3.0"

python - <<'PY'
import os
import torch
import mmcv
import mmengine
import mmdet
import mmpose

print("[VERIFY] torch:", torch.__version__, "cuda:", torch.version.cuda)
print("[VERIFY] mmcv:", mmcv.__version__)
print("[VERIFY] mmengine:", mmengine.__version__)
print("[VERIFY] mmdet:", mmdet.__version__)
print("[VERIFY] mmpose:", mmpose.__version__)
print("[VERIFY] CUDA_HOME:", os.environ.get("CUDA_HOME"))
PY

echo "================================================================="
echo "MMPose stack installed successfully."
echo "================================================================="
