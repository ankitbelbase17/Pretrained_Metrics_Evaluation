#!/bin/bash --login
#SBATCH --job-name=build_onnxruntime
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/users/dbartaula/logs/build_onnxruntime_%j.log
#SBATCH --error=/users/dbartaula/logs/build_onnxruntime_%j_error.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Build ONNX Runtime from source and install wheel into torch26_env.
#
# Defaults:
#   - CPU build (USE_CUDA=0)
#   - Version v1.22.0
#
# Optional GPU build:
#   sbatch --export=ALL,USE_CUDA=1,CUDA_HOME=/usr/local/cuda build_onnxruntime.slurm
# ---------------------------------------------------------------------------

export TOKENIZERS_PARALLELISM=false

source /users/dbartaula/miniforge3/etc/profile.d/conda.sh
conda activate torch26_env

ORT_VERSION="${ORT_VERSION:-v1.22.0}"
USE_CUDA="${USE_CUDA:-0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
WORKDIR="${WORKDIR:-/iopsstor/scratch/cscs/dbartaula/build_onnxruntime}"
SRC_DIR="$WORKDIR/onnxruntime"
WHEEL_DIR="$WORKDIR/wheels"

echo "================================================================="
echo "Building ONNX Runtime"
echo "  ORT_VERSION : $ORT_VERSION"
echo "  USE_CUDA    : $USE_CUDA"
echo "  CUDA_HOME   : $CUDA_HOME"
echo "  WORKDIR     : $WORKDIR"
echo "================================================================="

mkdir -p "$WORKDIR" "$WHEEL_DIR"

python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install --upgrade cmake ninja

if [ ! -d "$SRC_DIR/.git" ]; then
  git clone --recursive https://github.com/microsoft/onnxruntime.git "$SRC_DIR"
fi

cd "$SRC_DIR"
git fetch --all --tags
git checkout "$ORT_VERSION"
git submodule sync --recursive
git submodule update --init --recursive

COMMON_ARGS=(
  --config Release
  --parallel
  --build_wheel
  --skip_tests
  --cmake_generator Ninja
)

if [ "$USE_CUDA" = "1" ]; then
  echo "[INFO] CUDA build enabled."
  ./build.sh "${COMMON_ARGS[@]}" --use_cuda --cuda_home "$CUDA_HOME"
else
  echo "[INFO] CPU build enabled."
  ./build.sh "${COMMON_ARGS[@]}"
fi

# Pick newest built wheel from known output roots.
FOUND_WHEEL="$(find build -type f -name "onnxruntime*.whl" | sort | tail -n 1 || true)"
if [ -z "$FOUND_WHEEL" ]; then
  echo "[ERROR] No ONNX Runtime wheel found under build/."
  exit 1
fi

cp -f "$FOUND_WHEEL" "$WHEEL_DIR/"
python -m pip install --force-reinstall "$WHEEL_DIR/$(basename "$FOUND_WHEEL")"

echo "================================================================="
echo "Installed wheel:"
echo "  $WHEEL_DIR/$(basename "$FOUND_WHEEL")"
python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"
echo "SUCCESS"
echo "================================================================="

