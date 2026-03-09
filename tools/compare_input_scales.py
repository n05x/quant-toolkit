#!/usr/bin/env python3
"""Compare input_scale tensors between two NVFP4 checkpoints.

Analyzes calibration thoroughness by comparing per-expert activation scales.
A more thorough calibration generally produces:
  - Fewer zero/degenerate scales (uncalibrated experts)
  - Higher entropy in the scale distribution (more experts with distinct values)
  - Fewer outlier ratios between the two checkpoints

Usage:
    python tools/compare_input_scales.py \
        --ours ./output/Qwen3.5-397B-A17B-NVFP4 \
        --theirs /path/to/other/Qwen3.5-397B-A17B-NVFP4
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def load_input_scales(model_dir: Path) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]

    # Group scale keys by shard for single-pass loading.
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for k in wm:
        if k.endswith(".input_scale"):
            shard_to_keys[wm[k]].append(k)

    scales = {}
    for shard_name, keys in sorted(shard_to_keys.items()):
        print(f"    {shard_name}: {len(keys)} scales...")
        with safe_open(str(model_dir / shard_name), framework="pt") as f:
            for key in keys:
                scales[key] = f.get_tensor(key).float().cpu()
    return scales


def classify_key(key: str) -> tuple[str, str, int, str]:
    """Returns (layer, expert_type, expert_idx, proj)."""
    import re
    m = re.match(
        r".*layers\.(\d+)\.mlp\.(shared_expert|experts)\.?(\d*)\.(gate_proj|up_proj|down_proj)\.input_scale",
        key,
    )
    if not m:
        return ("?", "?", -1, "?")
    layer = m.group(1)
    expert_type = m.group(2)
    expert_idx = int(m.group(3)) if m.group(3) else 0
    proj = m.group(4)
    return (layer, expert_type, expert_idx, proj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--theirs", required=True, type=Path)
    args = parser.parse_args()

    print(f"Loading scales from: {args.ours}")
    ours = load_input_scales(args.ours)
    print(f"  {len(ours)} input_scale tensors")

    print(f"Loading scales from: {args.theirs}")
    theirs = load_input_scales(args.theirs)
    print(f"  {len(theirs)} input_scale tensors")

    common = sorted(set(ours.keys()) & set(theirs.keys()))
    only_ours = set(ours.keys()) - set(theirs.keys())
    only_theirs = set(theirs.keys()) - set(ours.keys())
    print(f"\nCommon keys: {len(common)}")
    if only_ours:
        print(f"Only in ours: {len(only_ours)}")
    if only_theirs:
        print(f"Only in theirs: {len(only_theirs)}")

    # --- Per-key comparison ---
    ratios = []
    zero_ours = 0
    zero_theirs = 0
    zero_both = 0
    diffs = []

    for key in common:
        a = ours[key].item() if ours[key].numel() == 1 else ours[key].abs().max().item()
        b = theirs[key].item() if theirs[key].numel() == 1 else theirs[key].abs().max().item()

        az = a == 0 or math.isnan(a)
        bz = b == 0 or math.isnan(b)

        if az and bz:
            zero_both += 1
        elif az:
            zero_ours += 1
        elif bz:
            zero_theirs += 1
        else:
            r = a / b
            ratios.append(r)
            diffs.append(abs(a - b))

    print(f"\n{'='*60}")
    print("DEGENERATE SCALES (zero or NaN)")
    print(f"{'='*60}")
    print(f"  Zero in ours only:   {zero_ours}")
    print(f"  Zero in theirs only: {zero_theirs}")
    print(f"  Zero in both:        {zero_both}")

    if not ratios:
        print("\nNo valid scale pairs to compare.")
        return

    ratios_t = torch.tensor(ratios)
    diffs_t = torch.tensor(diffs)

    print(f"\n{'='*60}")
    print(f"RATIO (ours / theirs) across {len(ratios)} valid pairs")
    print(f"{'='*60}")
    print(f"  Mean:   {ratios_t.mean():.4f}")
    print(f"  Median: {ratios_t.median():.4f}")
    print(f"  Std:    {ratios_t.std():.4f}")
    print(f"  Min:    {ratios_t.min():.4f}")
    print(f"  Max:    {ratios_t.max():.4f}")
    print(f"  P5:     {ratios_t.quantile(0.05):.4f}")
    print(f"  P95:    {ratios_t.quantile(0.95):.4f}")

    # How many are significantly different?
    far_off = (ratios_t < 0.5) | (ratios_t > 2.0)
    print(f"\n  |ratio| > 2x:  {far_off.sum().item()} ({far_off.float().mean()*100:.2f}%)")
    very_far = (ratios_t < 0.1) | (ratios_t > 10.0)
    print(f"  |ratio| > 10x: {very_far.sum().item()} ({very_far.float().mean()*100:.2f}%)")

    print(f"\n{'='*60}")
    print("ABSOLUTE DIFFERENCE")
    print(f"{'='*60}")
    print(f"  Mean:   {diffs_t.mean():.6f}")
    print(f"  Median: {diffs_t.median():.6f}")
    print(f"  P95:    {diffs_t.quantile(0.95):.6f}")
    print(f"  Max:    {diffs_t.max():.6f}")

    # --- Per-layer analysis ---
    print(f"\n{'='*60}")
    print("PER-LAYER SUMMARY (routed experts only)")
    print(f"{'='*60}")

    layer_stats = defaultdict(lambda: {"ratios": [], "zero_ours": 0, "zero_theirs": 0})
    for key in common:
        layer, etype, eidx, proj = classify_key(key)
        if etype != "experts":
            continue
        a = ours[key].item() if ours[key].numel() == 1 else ours[key].abs().max().item()
        b = theirs[key].item() if theirs[key].numel() == 1 else theirs[key].abs().max().item()
        az = a == 0 or math.isnan(a)
        bz = b == 0 or math.isnan(b)
        if az and not bz:
            layer_stats[layer]["zero_ours"] += 1
        elif bz and not az:
            layer_stats[layer]["zero_theirs"] += 1
        elif not az and not bz:
            layer_stats[layer]["ratios"].append(a / b)

    print(f"  {'Layer':>5}  {'Median R':>9}  {'Std R':>7}  {'Zero Us':>7}  {'Zero Them':>9}")
    for layer in sorted(layer_stats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        s = layer_stats[layer]
        rt = torch.tensor(s["ratios"]) if s["ratios"] else torch.tensor([0.0])
        print(f"  {layer:>5}  {rt.median():.4f}     {rt.std():.4f}  {s['zero_ours']:>7}  {s['zero_theirs']:>9}")

    # --- Entropy comparison (scale diversity) ---
    print(f"\n{'='*60}")
    print("SCALE DIVERSITY (entropy of log-binned scales per layer)")
    print(f"{'='*60}")

    def binned_entropy(values, n_bins=50):
        if len(values) < 2:
            return 0.0
        t = torch.tensor(values)
        t = t[t > 0]
        if len(t) < 2:
            return 0.0
        log_t = torch.log10(t)
        counts = torch.histc(log_t, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * probs.log2()).sum().item()

    layer_scales_ours = defaultdict(list)
    layer_scales_theirs = defaultdict(list)
    for key in common:
        layer, etype, _, _ = classify_key(key)
        if etype != "experts":
            continue
        a = ours[key].item() if ours[key].numel() == 1 else ours[key].abs().max().item()
        b = theirs[key].item() if theirs[key].numel() == 1 else theirs[key].abs().max().item()
        layer_scales_ours[layer].append(a)
        layer_scales_theirs[layer].append(b)

    print(f"  {'Layer':>5}  {'Ours H':>7}  {'Theirs H':>9}  {'Delta':>7}")
    total_h_ours = 0
    total_h_theirs = 0
    n_layers = 0
    for layer in sorted(layer_scales_ours.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        h_ours = binned_entropy(layer_scales_ours[layer])
        h_theirs = binned_entropy(layer_scales_theirs[layer])
        delta = h_ours - h_theirs
        print(f"  {layer:>5}  {h_ours:.3f}    {h_theirs:.3f}      {delta:+.3f}")
        total_h_ours += h_ours
        total_h_theirs += h_theirs
        n_layers += 1

    if n_layers:
        print(f"\n  Average entropy — Ours: {total_h_ours/n_layers:.3f}, "
              f"Theirs: {total_h_theirs/n_layers:.3f}, "
              f"Delta: {(total_h_ours-total_h_theirs)/n_layers:+.3f}")
        print(f"  (Higher entropy = more diverse/differentiated expert scales)")


if __name__ == "__main__":
    main()
