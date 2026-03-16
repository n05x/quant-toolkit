#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model qwen3_5_122b \
    --export-dir ./output/Qwen3.5-122B-A10B-NVFP4 \
    --calib-config configs/calib_qwen35_122b.toml \
    --save-quantiles ./output/Qwen3.5-122B-A10B-NVFP4/quantile_data.json \
    --floor-amaxes
