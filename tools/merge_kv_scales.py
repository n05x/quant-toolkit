#!/usr/bin/env python3
"""Merge per-rank KV amax files into a single kv_scales.safetensors.

Takes the element-wise max of amax values across ranks and converts to
FP8 E4M3 scales (amax / 448).  The output file uses key names that
SGLang's weight loader expects.

Usage:
    python merge_kv_scales.py \
        --input kv_scales_rank0.safetensors kv_scales_rank1.safetensors \
        --output ./output/MyModel/kv_scales.safetensors
"""

import argparse

import torch
from safetensors.torch import load_file, save_file

FP8_E4M3_MAX = 448.0

parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs="+", required=True, help="Per-rank amax files.")
parser.add_argument("--output", required=True, help="Output scales safetensors file.")
parser.add_argument("--margin", type=float, default=1.1,
                    help="Safety margin multiplier on amax (default: 1.1 = 10%%).")
args = parser.parse_args()

# Load all rank files and collect amax per key.
merged = {}
for path in args.input:
    data = load_file(path)
    for key, val in data.items():
        if key in merged:
            merged[key] = torch.max(merged[key], val)
        else:
            merged[key] = val.clone()

# Convert amax -> scale and rename keys.
print(f"Applying {args.margin:.0%} safety margin to amax values.")
scales = {}
for key, amax in sorted(merged.items()):
    am = amax.item() * args.margin
    sc = am / FP8_E4M3_MAX if am > 0 else 1.0

    if key.endswith(".k_amax"):
        scale_key = key.replace(".k_amax", ".self_attn.k_scale")
    elif key.endswith(".v_amax"):
        scale_key = key.replace(".v_amax", ".self_attn.v_scale")
    else:
        print(f"  skipping unknown key: {key}")
        continue

    scales[scale_key] = torch.tensor(sc, dtype=torch.float32)
    print(f"  {scale_key}: amax={am:.3f} scale={sc:.6f}")

save_file(scales, args.output)
print(f"\nSaved {len(scales)} scales to {args.output}")
