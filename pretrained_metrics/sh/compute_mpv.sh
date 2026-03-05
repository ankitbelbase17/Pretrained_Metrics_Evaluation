#!/bin/bash
# compute_mpv.sh
# Run from the project root: bash pretrained_metrics/sh/compute_mpv.sh

BATCH_SIZE=${1:-16}

python pretrained_metrics/compute_pretrained_metrics.py \
    --dataset mpv \
    --root ./MPV \
    --batch_size "$BATCH_SIZE" \
    --output_dir "./results/mpv"
