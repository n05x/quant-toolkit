"""Replace quantized shared expert weights with BF16 originals from the source model.

Copies the NVFP4 checkpoint, then for each layer's shared_expert:
  - Removes the FP4-packed weight, weight_scale, weight_scale_2, and input_scale
  - Inserts the original BF16 weight from the source checkpoint
  - Updates the safetensors index accordingly
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source BF16 model directory")
    parser.add_argument("--nvfp4", required=True, help="NVFP4 checkpoint directory")
    parser.add_argument("--out", required=True, help="Output directory for patched checkpoint")
    args = parser.parse_args()

    src_dir = Path(args.src)
    nvfp4_dir = Path(args.nvfp4)
    out_dir = Path(args.out)

    # Copy the entire checkpoint.
    if out_dir.exists():
        print(f"Output directory {out_dir} already exists, aborting.")
        return
    print(f"Copying {nvfp4_dir} -> {out_dir} ...")
    shutil.copytree(nvfp4_dir, out_dir, symlinks=False)

    # Load both indices.
    with open(out_dir / "model.safetensors.index.json") as f:
        nvfp4_index = json.load(f)
    with open(src_dir / "model.safetensors.index.json") as f:
        src_index = json.load(f)

    wm = nvfp4_index["weight_map"]
    src_wm = src_index["weight_map"]

    # Identify shared expert keys to replace.
    shared_re = re.compile(
        r"model\.language_model\.layers\.(\d+)\.mlp\.shared_expert\."
        r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)"
    )

    keys_to_remove = []
    bf16_keys = {}  # key -> source shard
    for k in list(wm.keys()):
        m = shared_re.match(k)
        if not m:
            continue
        layer, proj, suffix = m.group(1), m.group(2), m.group(3)
        if suffix == "weight":
            # This is the FP4-packed weight; replace with BF16.
            src_key = k  # Same key name in source.
            if src_key in src_wm:
                bf16_keys[src_key] = src_wm[src_key]
            keys_to_remove.append(k)
        else:
            # weight_scale, weight_scale_2, input_scale — remove entirely.
            keys_to_remove.append(k)

    print(f"Shared expert keys to remove from NVFP4: {len(keys_to_remove)}")
    print(f"BF16 weights to copy from source: {len(bf16_keys)}")

    # Figure out which NVFP4 shards are affected (need rewriting).
    affected_shards = set()
    for k in keys_to_remove:
        affected_shards.add(wm[k])

    # Remove shared expert keys from weight_map.
    for k in keys_to_remove:
        del wm[k]

    # Load BF16 weights grouped by source shard.
    src_by_shard = {}
    for k, shard in bf16_keys.items():
        src_by_shard.setdefault(shard, []).append(k)

    bf16_tensors = {}
    for shard, keys in src_by_shard.items():
        shard_path = src_dir / shard
        print(f"  Loading {len(keys)} BF16 tensors from {shard}")
        from safetensors import safe_open
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for k in keys:
                bf16_tensors[k] = f.get_tensor(k)

    # Save BF16 shared expert weights into a new shard.
    new_shard = "model-shared-expert-bf16.safetensors"
    print(f"Saving {len(bf16_tensors)} BF16 tensors to {new_shard}")
    save_file(bf16_tensors, str(out_dir / new_shard))
    for k in bf16_tensors:
        wm[k] = new_shard

    # Rewrite affected NVFP4 shards (remove shared expert tensors).
    for shard in sorted(affected_shards):
        if shard == "model-inputscales.safetensors":
            # Rewrite input scales without shared expert entries.
            shard_path = out_dir / shard
            data = load_file(str(shard_path))
            removed = 0
            for k in list(data.keys()):
                if shared_re.match(k):
                    del data[k]
                    removed += 1
            print(f"  Rewriting {shard}: removed {removed} shared expert input scales")
            save_file(data, str(shard_path))
        else:
            shard_path = out_dir / shard
            data = load_file(str(shard_path))
            removed = 0
            for k in list(data.keys()):
                if shared_re.match(k):
                    del data[k]
                    removed += 1
            if removed > 0:
                print(f"  Rewriting {shard}: removed {removed} shared expert tensors")
                save_file(data, str(shard_path))

    # Update metadata.
    total = sum(t.numel() for t in bf16_tensors.values())
    nvfp4_index["metadata"]["total_size"] = nvfp4_index["metadata"].get("total_size", 0)
    print(f"  Added {total:,} BF16 params for shared experts")

    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(nvfp4_index, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
