#!/bin/bash
#SBATCH --job-name=metric_tests
#SBATCH --output=logs/metric_tests_%j.out
#SBATCH --error=logs/metric_tests_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1            # Single GPU for testing
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --account=a168
#SBATCH --partition=normal

# =============================================================================
# Run all per-metric standalone smoke-tests (M1–M9)
#
# Each test:
#   1. Loads the pretrained model(s) for that metric
#   2. Runs update + compute on a few batches
#   3. Prints PASSED / FAILED with computed values
#
# Usage:
#   sbatch run_all_metric_tests.slurm
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────
CURVTON_ROOT="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test/hard"
MAX_BATCHES=2          # Quick smoke-test (2 batches = 32 images)
BATCH_SIZE=16
DEVICE="cuda"

mkdir -p logs

echo "============================================================"
echo "  Per-Metric Smoke-Tests — $(date)"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  MAX_BATCHES=${MAX_BATCHES}  BATCH_SIZE=${BATCH_SIZE}"
echo "============================================================"
echo ""

PASS=0
FAIL=0

run_test() {
    local label="$1"
    local script="$2"
    echo "──────────────────────────────────────────────────────────────"
    echo "  Running: ${label}"
    echo "──────────────────────────────────────────────────────────────"
    python "${script}" \
        --device "${DEVICE}" \
        --curvton_root "${CURVTON_ROOT}" \
        --max_batches "${MAX_BATCHES}" \
        --batch_size "${BATCH_SIZE}" \
        --verbose
    if [ $? -eq 0 ]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# ── Run each metric ──────────────────────────────────────────────────────────
run_test "M1 Pose (KeypointRCNN)"                     test_m1_pose.py
run_test "M2 Occlusion (Mask2Former/SegFormer)"        test_m2_occlusion.py
run_test "M3 Background (DeepLabV3 + DETR)"            test_m3_background.py
run_test "M4 Illumination (signal processing)"         test_m4_illumination.py
run_test "M5 Body Shape (HMR2.0 / ViT)"               test_m5_body_shape.py
run_test "M6 Appearance (ArcFace / open_clip)"         test_appearance.py
run_test "M7 Garment Texture (open_clip / ViT)"        test_m7_garment_texture.py
run_test "M8 VAE Latent (SD VAE)"                      test_m8_vae_latent.py
run_test "M9 Camera Angle (HMR2.0 / KeypointRCNN)"    test_m9_camera_angle.py

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL))
echo "============================================================"
echo "  SUMMARY:  ${PASS}/${TOTAL} PASSED,  ${FAIL}/${TOTAL} FAILED"
echo "  Completed at: $(date)"
echo "============================================================"

if [ ${FAIL} -gt 0 ]; then
    exit 1
fi
exit 0
