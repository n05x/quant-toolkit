"""Merge calibration amax files and update checkpoint input_scale values.

Takes N amax safetensor files (from --save-amax runs), merges them via
element-wise max, converts to input_scale, and updates the checkpoint's
input_scales.safetensors in-place (taking max with existing scales).

Requires the checkpoint to have been restructured with restructure_scales.py
(or exported with the updated export_hf.py) so that all input_scale tensors
live in a single input_scales.safetensors file.

Usage:
    python merge_amax.py \
        --checkpoint ./output/GLM-5-NVFP4 \
        --amax amax_coding.safetensors amax_broad.safetensors [...]

    # Dry run (show what would change, don't write):
    python merge_amax.py --checkpoint ./output/GLM-5-NVFP4 \
        --amax amax_broad.safetensors --dry-run
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SCALE_SHARD_NAME = "input_scales.safetensors"

# NVFP4: input_scale = amax / (maxbound * 448.0) = amax / (6.0 * 448.0)
NVFP4_SCALE_DIVISOR = 6.0 * 448.0  # 2688.0

# Matches the _QuantFusedExperts module tree layout:
#   experts.gate_proj.0.input_quantizer  (projection-first)
# and remaps to checkpoint layout:
#   experts.0.gate_proj.input_scale      (expert-first)
_EXPERT_PROJ_RE = re.compile(
    r"^(.*\.experts)\.(gate_proj|up_proj|down_proj)\.(\d+)\.(.+)$"
)


def _amax_key_to_checkpoint_key(amax_key):
    """Convert an amax module-tree key to the corresponding checkpoint key.

    Module tree:  model.layers.0.mlp.experts.gate_proj.0.input_quantizer
    Checkpoint:   model.layers.0.mlp.experts.0.gate_proj.input_scale
    """
    # Only process input_quantizer keys (weight scales are deterministic)
    if not amax_key.endswith(".input_quantizer"):
        return None

    # Replace suffix
    key = amax_key.replace(".input_quantizer", ".input_scale")

    # Remap expert layout: projection-first -> expert-first
    m = _EXPERT_PROJ_RE.match(key)
    if m:
        prefix, proj, idx, suffix = m.groups()
        key = f"{prefix}.{idx}.{proj}.{suffix}"

    return key


def load_and_merge_amaxes(amax_paths):
    """Load N amax files and merge via element-wise max."""
    merged = {}

    for path in amax_paths:
        print(f"Loading {path}...")
        amaxes = load_file(path)
        print(f"  {len(amaxes)} keys")

        for key, val in amaxes.items():
            val = val.float().squeeze()  # saved as 1-d, squeeze to scalar
            if key in merged:
                merged[key] = torch.max(merged[key], val)
            else:
                merged[key] = val

    print(f"\nMerged: {len(merged)} unique amax keys across {len(amax_paths)} files")
    return merged


def _floor_sparse_amaxes(amaxes):
    """Floor outlier expert amaxes at median/10 of their peer group.

    A sparsely-calibrated expert (activated on a few samples) is more
    dangerous than an uncalibrated one: it has a tight scale fitted to
    those few samples, and will clip on out-of-distribution inputs → NaN.

    For each (layer, projection) group, experts with amax < median/10 are
    pulled up to median/10. This catches dangerous outliers without
    flattening the natural per-expert variation.
    """
    # GLM-5 (projection-first): experts.gate_proj.0.input_quantizer
    # MiniMax (expert-first):   experts.0.w1.input_quantizer
    expert_patterns = [
        re.compile(r"^(.*\.experts)\.(gate_proj|up_proj|down_proj)\.(\d+)\.input_quantizer$"),
        re.compile(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.input_quantizer$"),
    ]
    groups = defaultdict(dict)
    key_templates = {}  # group_key -> format string for reconstructing keys

    for key, val in amaxes.items():
        m = expert_patterns[0].match(key)
        if m:
            group_key = (m.group(1), m.group(2))
            expert_idx = int(m.group(3))
            groups[group_key][expert_idx] = val
            key_templates[group_key] = "{prefix}.{proj}.{idx}.input_quantizer"
            continue
        m = expert_patterns[1].match(key)
        if m:
            group_key = (m.group(1), m.group(3))
            expert_idx = int(m.group(2))
            groups[group_key][expert_idx] = val
            key_templates[group_key] = "{prefix}.{idx}.{proj}.input_quantizer"

    floored = 0
    for (prefix, proj), experts in groups.items():
        calibrated = sorted(v for v in experts.values() if v > 0)
        if not calibrated:
            continue
        median = calibrated[len(calibrated) // 2]
        threshold = median / 10
        tmpl = key_templates[(prefix, proj)]
        for idx, val in experts.items():
            if val < threshold:
                key = tmpl.format(prefix=prefix, proj=proj, idx=idx)
                amaxes[key] = threshold
                floored += 1

    print(f"Floored {floored} expert amaxes to median/10 "
          f"({len(groups)} groups)")
    return amaxes


def convert_to_input_scales(amaxes):
    """Filter to input_quantizer keys only, convert amax -> input_scale,
    and remap keys to checkpoint format."""
    scales = {}
    skipped = 0

    for amax_key, amax_val in amaxes.items():
        ckpt_key = _amax_key_to_checkpoint_key(amax_key)
        if ckpt_key is None:
            skipped += 1
            continue

        scale = amax_val / NVFP4_SCALE_DIVISOR

        # Zero amax means the expert was never activated in ANY calibration
        # run AND flooring didn't help (entire group uncalibrated).
        # Don't overwrite the existing checkpoint scale with zero.
        if scale <= 0:
            skipped += 1
            continue

        scales[ckpt_key] = scale

    print(f"Converted {len(scales)} input amaxes to scales "
          f"(skipped {skipped} non-input/zero keys)")
    return scales


def _floor_checkpoint_scales(scales):
    """Floor outlier expert input_scales directly in checkpoint key format.

    Same logic as _floor_sparse_amaxes but operates on checkpoint-format
    input_scale keys and values, so it can be used without any amax files.
    """
    expert_patterns = [
        re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.input_scale$"),
        re.compile(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.input_scale$"),
    ]
    groups = defaultdict(dict)

    for key, val in scales.items():
        for pat in expert_patterns:
            m = pat.match(key)
            if m:
                group_key = (m.group(1), m.group(3))
                expert_idx = int(m.group(2))
                groups[group_key][expert_idx] = (key, val.float().squeeze())
                break

    floored = 0
    for (prefix, proj), experts in groups.items():
        vals = sorted(v for _, v in experts.values() if v > 0)
        if not vals:
            continue
        median = vals[len(vals) // 2]
        threshold = median / 10
        for idx, (key, val) in experts.items():
            if val < threshold:
                scales[key] = threshold.reshape(scales[key].shape)
                floored += 1

    print(f"Floored {floored} expert scales to median/10 "
          f"({len(groups)} groups)")
    return scales


def update_checkpoint(checkpoint_dir, new_scales, dry_run=False):
    """Update input_scale values in the checkpoint's input_scales.safetensors.

    For each key, takes max(existing_scale, new_scale).
    """
    checkpoint_dir = Path(checkpoint_dir)
    scale_path = checkpoint_dir / SCALE_SHARD_NAME

    if not scale_path.is_file():
        raise FileNotFoundError(
            f"{scale_path} not found. Run restructure_scales.py first."
        )

    existing = load_file(str(scale_path))
    print(f"\nLoaded {len(existing)} scales from {SCALE_SHARD_NAME}")

    missing_keys = [k for k in new_scales if k not in existing]
    if missing_keys:
        print(f"WARNING: {len(missing_keys)} keys not in {SCALE_SHARD_NAME}:")
        for k in missing_keys[:10]:
            print(f"  {k}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")

    updated = 0
    unchanged = 0

    for key, new_scale in new_scales.items():
        if key not in existing:
            continue
        old_scale = existing[key].float().squeeze()
        final_scale = torch.max(old_scale, new_scale)

        if final_scale > old_scale:
            existing[key] = final_scale.reshape(existing[key].shape)
            updated += 1
        else:
            unchanged += 1

    if updated > 0 and not dry_run:
        save_file(existing, str(scale_path))

    print(f"\nTotal: {updated} scales updated, {unchanged} unchanged")
    if dry_run and updated > 0:
        print("(dry run — no files were modified)")

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Merge amax files and update checkpoint input_scale values."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the quantized checkpoint directory")
    parser.add_argument("--amax", nargs="*", default=[],
                        help="One or more amax safetensors files to merge (omit to just floor existing scales)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files")
    args = parser.parse_args()

    # Validate inputs
    for path in args.amax:
        if not os.path.isfile(path):
            parser.error(f"Amax file not found: {path}")
    scale_path = Path(args.checkpoint) / SCALE_SHARD_NAME
    if not scale_path.is_file():
        parser.error(
            f"{scale_path} not found. Run restructure_scales.py first."
        )

    if args.amax:
        # Full pipeline: merge amax files, floor, convert, update checkpoint.
        merged_amaxes = load_and_merge_amaxes(args.amax)
        merged_amaxes = _floor_sparse_amaxes(merged_amaxes)
        new_scales = convert_to_input_scales(merged_amaxes)
        updated = update_checkpoint(args.checkpoint, new_scales, dry_run=args.dry_run)

        if updated > 0 and not args.dry_run:
            print(f"\nCheckpoint updated successfully.")
        elif updated == 0:
            print(f"\nNo scales needed updating (all existing scales >= new scales).")
    else:
        # No amax files: just floor existing checkpoint scales in-place.
        print("No amax files provided — flooring existing checkpoint scales only.")
        existing = load_file(str(scale_path))
        print(f"Loaded {len(existing)} scales from {SCALE_SHARD_NAME}")
        existing = _floor_checkpoint_scales(existing)
        if not args.dry_run:
            save_file(existing, str(scale_path))
            print(f"\nCheckpoint updated successfully.")
        else:
            print("(dry run — no files were modified)")


if __name__ == "__main__":
    main()
