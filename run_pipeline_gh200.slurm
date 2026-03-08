#!/bin/bash --login
#SBATCH --job-name=curvton_pipeline
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/users/dbartaula/logs/pipeline_%j.log
#SBATCH --error=/users/dbartaula/logs/pipeline_%j_error.log

# ══════════════════════════════════════════════════════════════════════════════
# Unified Pipeline for CURVTON: Pretrained Metrics + EDA
# Hardware: 1 Node x 4 GH200 GPUs (4 GPUs total)
# ══════════════════════════════════════════════════════════════════════════════

# ── WARNING: YOU MUST ACTIVATE YOUR PYTHON ENVIRONMENT HERE ──
source /iopsstor/scratch/cscs/dbartaula/miniforge3/etc/profile.d/conda.sh
conda activate CURVTON7
# ─────────────────────────────────────────────────────────────

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# ── USER MUST CONFIGURE THIS ──────────────────────────────────────────────────
BASE_PATH="/path/to/dataset_ultimate"  # Path to the actual image dataset folder
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

# Dynamically get the master node for torchrun
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

echo "================================================================="
echo "Starting Unified Pipeline (Metrics -> EDA)"
echo "Master Node: $MASTER_ADDR"
echo "Nodes: $SLURM_NNODES, GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Base Path: $BASE_PATH"
echo "================================================================="

# 1. Pretrained Metrics (Computes over Ratios 10%, 20%, 40%, 50%, 100%)
echo " "
echo "================================================================="
echo "[1/2] Launching Distributed Pretrained Metric Computation..."
echo "================================================================="
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    pretrained_metrics/compute_curvton_metrics.py \
    --base_path "$BASE_PATH" \
    --multi_ratio \
    --ratios 0.1 0.2 0.4 0.5 1.0 \
    --batch_size 40 \
    --num_workers 16

if [ $? -ne 0 ]; then
    echo "Error: Pretrained Metric Computation failed! Exiting."
    exit 1
fi


# 2. EDA Pipeline (Generates plots & umaps across splits and ratios)
echo " "
echo "================================================================="
echo "[2/2] Launching Distributed EDA Feature Extraction and Plotting..."
echo "================================================================="
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    EDA/run_curvton_eda.py \
    --base_path "$BASE_PATH" \
    --out_dir "figures/curvton" \
    --cache_dir "eda_cache/curvton" \
    --sample_ratio 0.25 \
    --batch_size 40 \
    --num_workers 16

if [ $? -ne 0 ]; then
    echo "Error: EDA Pipeline failed!"
    exit 1
fi

echo "================================================================="
echo "Pipeline Execution Completed Successfully!"
echo "Outputs:"
echo " - Metrics JSONs: results/unified/"
echo " - EDA Plots    : figures/curvton/"
echo "================================================================="
