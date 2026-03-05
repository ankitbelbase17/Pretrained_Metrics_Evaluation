#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# CURVTON Dataset EDA and Metrics Computation Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Computes:
# 1. EDA plots for Easy, Medium, Hard splits (individual + overlapped)
# 2. Pretrained metrics for all difficulty levels
# 3. Multi-ratio analysis (10%, 20%, 30%, 40%, 50%, 100%)
#
# Usage:
#   sbatch compute_curvton.sh           # Full run
#   bash compute_curvton.sh --test      # Test set only
#   bash compute_curvton.sh --quick     # Quick test (10% sample)

#SBATCH --job-name=curvton_eda
#SBATCH --output=logs/curvton_eda_%j.out
#SBATCH --error=logs/curvton_eda_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Dataset paths
TRAIN_PATH="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3/dataset_ultimate"
TEST_PATH="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3/dataset_ultimate_test"

# Output directories
EDA_OUT="figures/curvton"
METRICS_OUT="metrics_output/curvton"
CACHE_DIR="eda_cache/curvton"

# Sample ratios for multi-ratio analysis
RATIOS="0.1 0.2 0.3 0.4 0.5 1.0"

# ═══════════════════════════════════════════════════════════════════════════════
# Parse arguments
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH=$TRAIN_PATH
SUFFIX=""
SAMPLE_RATIO="1.0"
MULTI_RATIO=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test) 
            DATASET_PATH=$TEST_PATH
            SUFFIX="_test"
            ;;
        --quick)
            SAMPLE_RATIO="0.1"
            ;;
        --multi-ratio)
            MULTI_RATIO=true
            ;;
        --ratio)
            SAMPLE_RATIO="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Update output directories with suffix
EDA_OUT="${EDA_OUT}${SUFFIX}"
METRICS_OUT="${METRICS_OUT}${SUFFIX}"

# ═══════════════════════════════════════════════════════════════════════════════
# Setup environment
# ═══════════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "CURVTON Dataset Analysis Pipeline"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Dataset:       $DATASET_PATH"
echo "  EDA Output:    $EDA_OUT"
echo "  Metrics Output: $METRICS_OUT"
echo "  Sample Ratio:  $SAMPLE_RATIO"
echo "  Multi-Ratio:   $MULTI_RATIO"
echo "═══════════════════════════════════════════════════════════════════════════════"

# Create output directories
mkdir -p "$EDA_OUT" "$METRICS_OUT" "$CACHE_DIR" "logs"

# Activate conda environment (adjust as needed)
# source activate tryon_eval

# ═══════════════════════════════════════════════════════════════════════════════
# Run EDA Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║ Step 1: Running EDA Pipeline                                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"

if [ "$MULTI_RATIO" = true ]; then
    python EDA/run_curvton_eda.py \
        --base_path "$DATASET_PATH" \
        --out_dir "$EDA_OUT" \
        --cache_dir "$CACHE_DIR" \
        --multi_ratio
else
    python EDA/run_curvton_eda.py \
        --base_path "$DATASET_PATH" \
        --out_dir "$EDA_OUT" \
        --cache_dir "$CACHE_DIR" \
        --sample_ratio "$SAMPLE_RATIO"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Run Pretrained Metrics Computation
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║ Step 2: Computing Pretrained Metrics                                          ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"

if [ "$MULTI_RATIO" = true ]; then
    python pretrained_metrics/compute_curvton_metrics.py \
        --base_path "$DATASET_PATH" \
        --out_dir "$METRICS_OUT" \
        --multi_ratio
else
    python pretrained_metrics/compute_curvton_metrics.py \
        --base_path "$DATASET_PATH" \
        --out_dir "$METRICS_OUT" \
        --sample_ratio "$SAMPLE_RATIO"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "CURVTON Analysis Complete!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "EDA Figures:"
ls -la "$EDA_OUT"/*.pdf 2>/dev/null || echo "  (check subdirectories)"
echo ""
echo "Metrics:"
ls -la "$METRICS_OUT"/*.json 2>/dev/null || echo "  (no metrics files found)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
