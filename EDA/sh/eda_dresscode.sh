#!/bin/bash
# eda_dresscode.sh
# Run from the project root: bash EDA/sh/eda_dresscode.sh

BATCH_SIZE=${1:-16}

python EDA/run_eda.py \
    --dataset dresscode \
    --root ./dresscode \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --cache_dir "./eda_cache_anish" \
    --out_dir "./figures_anish/dresscode"
