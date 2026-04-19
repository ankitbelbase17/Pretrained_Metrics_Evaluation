#!/bin/bash --login
#SBATCH --job-name=curvton_unified
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/users/dbartaula/logs/pipeline_unified_%j.log
#SBATCH --error=/users/dbartaula/logs/pipeline_unified_%j_error.log

# ══════════════════════════════════════════════════════════════════════════════
# Unified Pipeline for CURVTON: Pretrained Metrics + EDA + Comparison
# Hardware: 1 Node x 4 GH200 GPUs
# ══════════════════════════════════════════════════════════════════════════════


# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Dynamically get the master node for torchrun
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

echo "================================================================="
echo "Starting Unified Orchestrator Pipeline (Metrics -> EDA -> Radar)"
echo "Master Node: $MASTER_ADDR"
echo "Nodes: $SLURM_NNODES, GPUs per node: $SLURM_GPUS_ON_NODE"
echo "================================================================="

# Launch the unified Python orchestrator.
# --gpus 4 tells run.py to automatically use torchrun for DataParallel processing
# across the heavy deep learning models.
srun python3 run.py --gpus $SLURM_GPUS_ON_NODE

if [ $? -ne 0 ]; then
    echo "Error: Unified Pipeline failed! Check the logs above."
    exit 1
fi

echo "================================================================="
echo "Pipeline Execution Completed Successfully!"
echo "Outputs have been saved to your project directory (plots/ and JSONs)."
echo "================================================================="
