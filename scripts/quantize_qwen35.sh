#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model qwen3_5_moe \
    --export-dir ./output/Qwen3.5-397B-A17B-NVFP4 \
    --calib-config configs/calib_qwen35.toml \
    --cpu-capacity 200GiB \
    --streaming \
    --floor-amaxes
