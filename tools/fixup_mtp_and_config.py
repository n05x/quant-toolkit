#!/usr/bin/env python3
"""Merge MTP weights from the source checkpoint and fix config.json.

Patches an already-exported NVFP4 checkpoint that is missing the MTP
(multi-token prediction) weights and/or has an empty architectures field.

Usage:
    python tools/fixup_mtp_and_config.py \
        --export-dir ./output/Qwen3.5-397B-A17B-NVFP4 \
        --source-model Qwen/Qwen3.5-397B-A17B
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--source-model", required=True,
                        help="HF model ID or local path to the unquantized source.")
    parser.add_argument("--mtp-prefix", default="mtp.",
                        help="Key prefix for MTP weights in the source checkpoint.")
    args = parser.parse_args()

    export_dir = args.export_dir

    # Resolve source directory.
    source_dir = Path(args.source_model)
    if not source_dir.is_dir():
        from huggingface_hub import snapshot_download
        source_dir = Path(snapshot_download(
            args.source_model,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=True,
        ))

    # Load both index files.
    with open(export_dir / "model.safetensors.index.json") as f:
        our_index = json.load(f)
    our_wm = our_index["weight_map"]

    with open(source_dir / "model.safetensors.index.json") as f:
        src_wm = json.load(f)["weight_map"]

    mtp_keys = [k for k in src_wm if k.startswith(args.mtp_prefix)]
    already_present = [k for k in mtp_keys if k in our_wm]
    needed = [k for k in mtp_keys if k not in our_wm]

    print(f"Source has {len(mtp_keys)} MTP keys, "
          f"{len(already_present)} already present, {len(needed)} to merge.")

    if needed:
        src_shards = sorted(set(src_wm[k] for k in needed))
        mtp_shard: dict[str, torch.Tensor] = {}
        for src_shard_name in src_shards:
            src_data = load_file(str(source_dir / src_shard_name))
            for key in needed:
                if src_wm[key] == src_shard_name and key in src_data:
                    mtp_shard[key] = src_data[key].contiguous()

        shard_name = "model-mtp.safetensors"
        save_file(mtp_shard, str(export_dir / shard_name))
        for key in mtp_shard:
            our_wm[key] = shard_name

        total_bytes = sum(t.numel() * t.element_size() for t in mtp_shard.values())
        print(f"Wrote {len(mtp_shard)} MTP tensors to {shard_name} "
              f"({total_bytes / 1e9:.2f} GB)")

    # Update index.
    our_index["weight_map"] = dict(sorted(our_wm.items()))
    our_index["metadata"]["total_size"] = sum(
        (export_dir / f).stat().st_size for f in set(our_wm.values())
    )
    with open(export_dir / "model.safetensors.index.json", "w") as f:
        json.dump(our_index, f, indent=2)
    print(f"Updated index: {len(our_wm)} total keys.")

    # Fix config.json architectures.
    config_path = export_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    changed = False
    if not config.get("architectures"):
        arch = config.get("model_type", "")
        # Derive from model_type heuristic; fallback to what the source has.
        src_config_path = source_dir / "config.json"
        if src_config_path.exists():
            with open(src_config_path) as f:
                src_config = json.load(f)
            if src_config.get("architectures"):
                config["architectures"] = src_config["architectures"]
                changed = True

    if config.pop("auto_map", None) is not None:
        changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Fixed config.json: architectures={config.get('architectures')}")
    else:
        print(f"config.json OK: architectures={config.get('architectures')}")

    # Copy any missing metadata files from source.
    import shutil
    copy_patterns = [
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "special_tokens_map.json", "tokenizer.model",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "merges.txt", "vocab.json",
    ]
    copied = []
    for name in copy_patterns:
        src = source_dir / name
        dst = args.export_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    if copied:
        print(f"Copied missing metadata: {', '.join(copied)}")


if __name__ == "__main__":
    main()
