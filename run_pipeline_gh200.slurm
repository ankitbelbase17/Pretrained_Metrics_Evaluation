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
#SBATCH --output=logs/pipeline_unified_%j.log
#SBATCH --error=logs/pipeline_unified_%j_error.log

# ══════════════════════════════════════════════════════════════════════════════
# Unified Pipeline for CURVTON: Pretrained Metrics + EDA + Comparison
# Hardware: 1 Node x 4 GH200 GPUs
# ══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

# Ensure relative paths resolve inside this repository.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
OUT_DIR="$SCRIPT_DIR/assets"
mkdir -p "$LOG_DIR" "$OUT_DIR"


# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Activate the torch26_env environment
source /users/dbartaula/miniforge3/etc/profile.d/conda.sh
conda activate torch26_env

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
# run.py now handles non-halting phase execution and prints consolidated reports at the end.
echo "[INFO] Launching run.py with non-halting phase execution..."
srun python -u "$SCRIPT_DIR/run.py" \
    --gpus "$SLURM_GPUS_ON_NODE" \
    --output_dir "$OUT_DIR"
RUN_EXIT=$?

echo "================================================================="
if [ "$RUN_EXIT" -eq 0 ]; then
    echo "Pipeline execution completed successfully."
else
    echo "Pipeline execution finished with failures (exit code: $RUN_EXIT)."
    echo "run.py still completed all phases/checks and printed final consolidated report."
fi
echo "Outputs directory: $OUT_DIR"
echo "================================================================="
exit "$RUN_EXIT"
