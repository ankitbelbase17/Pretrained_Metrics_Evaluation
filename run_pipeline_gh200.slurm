#!/bin/bash --login
#SBATCH --job-name=curvton_pipeline
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
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
BASE_PATH="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3/dataset_ultimate"
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

# 0. Pretrained Metrics for All Datasets (DressCode, VITON-HD, StreetTryOn, etc.)
echo " "
echo "================================================================="
echo "[0/2] Launching Pretrained Metrics for All Datasets in YAML Config..."
echo "================================================================="
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    pretrained_metrics/compute_pretrained_metrics.py \
    --config configs/pretrained_metrics_datasets.yaml \
    --batch_size 40 \
    --num_workers 16

if [ $? -ne 0 ]; then
    echo "Error: Pretrained Metrics (All Datasets) failed! Exiting."
    exit 1
fi

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




# 2. EDA Pipeline (25% split for dataset_ultimate, test sets for others, and overlapped comparison plots)
echo " "
echo "================================================================="
echo "[2/2] Launching EDA for dataset_ultimate (25% split)..."
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
    echo "Error: EDA Pipeline (dataset_ultimate) failed!"
    exit 1
fi

echo " "
echo "================================================================="
echo "[2/3] Launching EDA for VITON-HD test set..."
echo "================================================================="
srun python EDA/run_eda.py \
    --dataset vitonhd \
    --root ./zalando-hd-resized/test \
    --batch_size 40 \
    --cache_dir "eda_cache/vitonhd" \
    --out_dir "figures/vitonhd"

echo " "
echo "================================================================="
echo "[3/3] Launching EDA for DressCode test set..."
echo "================================================================="
srun python EDA/run_eda.py \
    --dataset dresscode \
    --root ./dresscode/test \
    --batch_size 40 \
    --cache_dir "eda_cache/dresscode" \
    --out_dir "figures/dresscode"

echo " "
echo "================================================================="
echo "[4/4] Launching EDA for StreetTryOn test set..."
echo "================================================================="
srun python EDA/run_eda.py \
    --dataset street_tryon \
    --root ./street_tryon/test \
    --batch_size 40 \
    --cache_dir "eda_cache/street_tryon" \
    --out_dir "figures/street_tryon"

# Overlapped comparison plots (using cached features)
echo " "
echo "================================================================="
echo "[5/5] Generating overlapped comparison plots..."
echo "================================================================="
python EDA/run_eda.py \
    --figs_only \
    --cache_dir "eda_cache/curvton" "eda_cache/vitonhd" "eda_cache/dresscode" "eda_cache/street_tryon" \
    --labels "dataset_ultimate" "vitonhd" "dresscode" "street_tryon" \
    --out_dir "figures/comparison"

echo "================================================================="
echo "Pipeline Execution Completed Successfully!"
echo "Outputs:"
echo " - Metrics JSONs: results/unified/"
echo " - EDA Plots    : figures/curvton/"
echo "================================================================="
