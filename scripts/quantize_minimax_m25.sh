#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model minimax_m25 \
    --export-dir ./output/MiniMax-M2.5-NVFP4 \
    --calib-config configs/calib_minimax_m25.toml \
    --streaming \
    --floor-amaxes
