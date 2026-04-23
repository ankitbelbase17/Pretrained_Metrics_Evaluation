#!/bin/bash --login
#SBATCH --job-name=curvton_unified
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/users/dbartaula/logs/pipeline_unified_%j.log
#SBATCH --error=/users/dbartaula/logs/pipeline_unified_%j_error.log

# ══════════════════════════════════════════════════════════════════════════════
# Unified Pipeline for CURVTON: Pretrained Metrics + EDA + Comparison
# Hardware: 1 Node x 4 GH200 GPUs
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
trap 'ec=$?; echo "[ERROR] Command failed (exit ${ec}) at line ${LINENO}: ${BASH_COMMAND}"; exit ${ec}' ERR


# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Activate the torch26_env environment
source /users/dbartaula/miniforge3/etc/profile.d/conda.sh
conda activate torch26_env

# Ensure relative paths resolve inside this repository.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "run.py" ]; then
    echo "Error: run.py not found in $SCRIPT_DIR"
    exit 1
fi

# Dynamically get the master node for torchrun
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

echo "================================================================="
echo "Starting Unified Orchestrator Pipeline (Metrics -> EDA -> Radar)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "Master Node: $MASTER_ADDR"
echo "Nodes: $SLURM_NNODES, GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Working directory: $(pwd)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "Python path: $(which python)"
echo "================================================================="

# Launch the unified Python orchestrator.
# --gpus 4 tells run.py to automatically use torchrun for DataParallel processing
# across the heavy deep learning models.
srun python -u run.py --gpus $SLURM_GPUS_ON_NODE --output_dir "$SCRIPT_DIR/assets"

echo "================================================================="
echo "Pipeline Execution Completed Successfully!"
echo "Outputs have been saved to your project directory (plots/ and JSONs)."
echo "================================================================="
