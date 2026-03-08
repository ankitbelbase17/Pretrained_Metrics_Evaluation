#!/bin/bash
#SBATCH --job-name=curvton_metrics
#SBATCH --output=logs/curvton_metrics_%j.out
#SBATCH --error=logs/curvton_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4            # Request 4 GH200 GPUs
#SBATCH --cpus-per-task=32           # 32 CPU cores per node to feed GPUs
#SBATCH --mem=300G                   # Allocate plenty of RAM for dataset loading
#SBATCH --time=07:00:00              # 7 hours walltime
#SBATCH --partition=gpu              # Replace with your HPC's GH200 partition name

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_PATH="/path/to/dataset"         # <-- CHANGE THIS TO YOUR DATASET PATH
OUT_DIR="metrics_output/curvton"

# Load modules (modify according to your HPC environment)
# module load python/3.10
# module load cuda/12.2

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# Create log directory just in case it doesn't exist
mkdir -p logs

echo "Starting CURVTON metrics evaluation on multi-GH200"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs requested: $SLURM_GPUS_ON_NODE"
echo "=========================================================="

# 40 batch size per GPU * 4 GPUs = 160 effective batch size
BATCH_SIZE=40
NUM_WORKERS=16  # Slightly reduced per-GPU workers so 4 x 16 doesn't overwhelm the node

echo "Starting multi-GPU computation using torchrun..."
echo "Computing metrics for ratios: 10%, 20%, 30%, 40%, 100%"

srun torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE \
    pretrained_metrics/compute_curvton_metrics.py \
    --base_path "$BASE_PATH" \
    --out_dir "$OUT_DIR" \
    --multi_ratio \
    --ratios 0.1 0.2 0.3 0.4 1.0 \
    --difficulties easy medium hard \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS

echo "=========================================================="
echo "Job completed at: $(date)"
