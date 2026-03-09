#!/usr/bin/env python3
"""Dequantize FP8 checkpoint to bfloat16 by processing safetensors files directly."""
import argparse
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def resolve_model_dir(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to its local cache directory."""
    path = snapshot_download(model_id, local_files_only=True)
    return Path(path)


def process_single_file(args):
    """Process a single safetensors file - designed to run in parallel."""
    file_name, weight_names, model_dir, output_dir = args
    block_size = (128, 128)

    try:
        file_path = Path(model_dir) / file_name

        tensors = {}
        with safe_open(file_path, framework="pt") as f:
            for weight_name in weight_names:
                tensors[weight_name] = f.get_tensor(weight_name)

        processed = {}
        fp8_dequantized = 0

        for weight_name, tensor in tensors.items():
            if tensor is None:
                continue

            if tensor.dtype == torch.float8_e4m3fn and weight_name.endswith('.weight'):
                scale_name = weight_name.replace('.weight', '.weight_scale_inv')

                if scale_name in tensors:
                    fp8_dequantized += 1

                    weight_fp32 = tensor.to(torch.float32)
                    scale_inv = tensors[scale_name]

                    block_m, block_n = block_size
                    scale_expanded = scale_inv.repeat_interleave(block_m, dim=0)
                    scale_expanded = scale_expanded.repeat_interleave(block_n, dim=1)
                    scale_expanded = scale_expanded[:weight_fp32.shape[0], :weight_fp32.shape[1]]

                    # Despite the name "weight_scale_inv", these are actually scales (not inverses).
                    # The Triton kernel multiplies by them, so we do too.
                    dequant = (weight_fp32 * scale_expanded).to(torch.bfloat16)
                    processed[weight_name] = dequant

                    if fp8_dequantized <= 3:
                        print(f"    {weight_name}:")
                        print(f"      FP8 weight: min={weight_fp32.min():.2f}, max={weight_fp32.max():.2f}")
                        print(f"      Scale: min={scale_inv.min():.6f}, max={scale_inv.max():.6f}")
                        print(f"      Dequantized: min={dequant.min():.2f}, max={dequant.max():.2f}, mean={dequant.mean():.4f}, std={dequant.std():.4f}")

                    tensors[scale_name] = None
                else:
                    processed[weight_name] = tensor.to(torch.bfloat16)
            elif weight_name.endswith('.weight_scale_inv'):
                pass
            else:
                if tensor.dtype == torch.float8_e4m3fn:
                    processed[weight_name] = tensor.to(torch.bfloat16)
                else:
                    processed[weight_name] = tensor

        output_file = Path(output_dir) / file_name
        save_file(processed, output_file)

        return (file_name, list(processed.keys()), fp8_dequantized, None)

    except Exception as e:
        import traceback
        error_str = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return (file_name, [], 0, error_str)


def main():
    parser = argparse.ArgumentParser(description="Dequantize FP8 checkpoint to bfloat16")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g. MiniMaxAI/MiniMax-M2.5)")
    parser.add_argument("-o", "--output-dir", help="Output directory for dequantized checkpoint.")
    parser.add_argument("--model-dir", help="Override model directory instead of resolving from HF cache")
    parser.add_argument("--config-source",
                        help="Alternative model ID or directory to copy config/tokenizer files from")
    parser.add_argument("-j", "--workers", type=int, default=min(8, multiprocessing.cpu_count()),
                        help="Number of parallel workers (default: min(8, cpu_count))")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        print(f"Resolving {args.model_id} from HuggingFace cache...")
        model_dir = resolve_model_dir(args.model_id)

    model_name = args.model_id.split("/")[-1]
    if not args.output_dir:
        parser.error("--output-dir is required")
    output_dir = args.output_dir

    print(f"Dequantizing {model_dir} to {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading index...")
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index['weight_map']
    metadata = index.get('metadata', {})

    safetensors_files = sorted(set(weight_map.values()))
    print(f"Found {len(safetensors_files)} safetensors files to process")

    file_to_weights = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in file_to_weights:
            file_to_weights[file_name] = []
        file_to_weights[file_name].append(weight_name)

    num_workers = args.workers
    print(f"\nProcessing {len(safetensors_files)} files with {num_workers} workers...")

    new_weight_map = {}
    total_dequantized = 0

    process_args = [
        (file_name, file_to_weights[file_name], str(model_dir), output_dir)
        for file_name in safetensors_files
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, a): a[0]
            for a in process_args
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            returned_file, weight_names, fp8_count, error = future.result()

            if error is not None:
                print(f"  ERROR processing {file_name}:")
                print(f"    {error[:200]}...")
            else:
                for weight_name in weight_names:
                    new_weight_map[weight_name] = returned_file

                total_dequantized += fp8_count
                completed = len([f for f in future_to_file if f.done()])
                print(f"  [{completed}/{len(safetensors_files)}] Completed {returned_file} ({fp8_count} FP8 weights)")

    print(f"\n{'='*80}")
    print(f"Parallel processing complete!")
    print(f"  Total files processed: {len(safetensors_files)}")
    print(f"  Total FP8 weights dequantized: {total_dequantized}")

    print("\nCreating new index...")
    new_index = {
        "metadata": metadata,
        "weight_map": new_weight_map
    }

    index_output = Path(output_dir) / "model.safetensors.index.json"
    with open(index_output, 'w') as f:
        json.dump(new_index, f, indent=2)

    # Copy config/tokenizer files from --config-source if given, else from model_dir.
    config_dir = model_dir
    if args.config_source:
        if Path(args.config_source).is_dir():
            config_dir = Path(args.config_source)
        else:
            print(f"Resolving config source {args.config_source} from HuggingFace cache...")
            config_dir = resolve_model_dir(args.config_source)

    COPY_EXTS = {'.json', '.txt', '.model', '.py', '.jinja'}
    print(f"Copying config and tokenizer files from {config_dir}...")
    for src in config_dir.rglob('*'):
        if not src.is_file() or src.suffix not in COPY_EXTS or 'safetensors' in src.name:
            continue
        try:
            rel_path = src.relative_to(config_dir)
            dst = Path(output_dir) / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.name == 'config.json':
                with open(src) as f:
                    config = json.load(f)
                if 'quantization_config' in config:
                    del config['quantization_config']
                with open(dst, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                shutil.copy(src, dst)
            print(f"  {rel_path}")
        except Exception as e:
            print(f"  Skipping {src.name}: {e}")

    print(f"\n{'='*80}")
    print(f"Dequantized checkpoint saved to: {output_dir}")
    print(f"  Total weights: {len(new_weight_map)}")
    print(f"  Total FP8 weights dequantized: {total_dequantized}")


if __name__ == "__main__":
    main()
