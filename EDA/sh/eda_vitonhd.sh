#!/bin/bash
# eda_vitonhd.sh
# Run from the project root: bash EDA/sh/eda_vitonhd.sh

BATCH_SIZE=${1:-16}

python EDA/run_eda.py \
    --dataset vitonhd \
    --root ./zalando-hd-resized \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --cache_dir "./eda_cache_anish" \
    --out_dir "./figures_anish/vitonhd"
