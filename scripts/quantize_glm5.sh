#!/bin/bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

export SAFETENSORS_FAST_GPU=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python quantize.py \
    --model glm5 \
    --export-dir ./output/GLM-5-NVFP4 \
    --calib-config configs/calib_glm5.toml \
    --cpu-capacity 120GiB \
    --floor-amaxes
