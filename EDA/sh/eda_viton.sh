#!/bin/bash
# eda_viton.sh
# Run from the project root: bash EDA/sh/eda_viton.sh

BATCH_SIZE=${1:-16}

python EDA/run_eda.py \
    --dataset viton \
    --root ./VITON \
    --batch_size "$BATCH_SIZE" \
    --cache_dir "./eda_cache" \
    --out_dir "./figures/viton"
