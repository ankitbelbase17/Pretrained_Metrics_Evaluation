#!/bin/bash
# compute_viton.sh
# Run from the project root: bash pretrained_metrics/sh/compute_viton.sh

BATCH_SIZE=${1:-16}

python pretrained_metrics/compute_pretrained_metrics.py \
    --dataset viton \
    --root ./VITON \
    --batch_size "$BATCH_SIZE" \
    --output_dir "./results/viton"
