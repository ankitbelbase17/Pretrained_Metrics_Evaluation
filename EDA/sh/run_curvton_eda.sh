#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SLURM job script – CURVTON EDA Pipeline (multi-GPU)
# Cluster: clariden
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   sbatch EDA/sh/run_curvton_eda.sh
#
# ─── SLURM directives ────────────────────────────────────────────────────────
#SBATCH --job-name=curvton_eda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/curvton_eda_%j.log
#SBATCH --error=logs/curvton_eda_%j.err

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_PATH="/cluster/home/dbartaula/datasets/dataset_ultimate"
OUT_DIR="figures/curvton"
CACHE_DIR="eda_cache/curvton"
SAMPLE_RATIO=1.0
BATCH_SIZE=32
NUM_GPUS=4
CONDA_ENV="CURVTON7"

# ─── Setup ───────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs

echo "════════════════════════════════════════════════════════════════════════"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Date:      $(date)"
echo "════════════════════════════════════════════════════════════════════════"

# Activate conda environment
source activate ${CONDA_ENV} 2>/dev/null \
    || conda activate ${CONDA_ENV} 2>/dev/null \
    || { echo "Failed to activate conda env ${CONDA_ENV}"; exit 1; }

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"

# ─── Run ─────────────────────────────────────────────────────────────────────
python EDA/run_curvton_eda.py \
    --base_path "${BASE_PATH}" \
    --out_dir   "${OUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --sample_ratio ${SAMPLE_RATIO} \
    --batch_size   ${BATCH_SIZE} \
    --num_gpus     ${NUM_GPUS} \
    --difficulties easy medium hard

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Finished at $(date)"
echo "════════════════════════════════════════════════════════════════════════"
