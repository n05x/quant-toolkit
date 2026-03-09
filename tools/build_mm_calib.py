#!/usr/bin/env python3
"""Build a multimodal calibration JSONL from COCO val2017.

Downloads ~200 images and pairs them with varied prompts for calibration
activation diversity.

Usage:
    python tools/build_mm_calib.py \
        --output data/multimodal/vqa_calib.jsonl \
        --image-dir data/multimodal/images \
        --count 200
"""

import argparse
import json
import os
import random
import zipfile
from io import BytesIO

import requests

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"

PROMPTS = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What is happening in this scene?",
    "Count the number of people visible in this image.",
    "What colors are dominant in this image?",
    "Describe the spatial layout of objects in this image.",
    "What is the setting or environment shown?",
    "Are there any text or signs visible? If so, what do they say?",
    "What emotions or mood does this image convey?",
    "Describe the lighting and time of day.",
    "What would happen next in this scene?",
    "Compare the foreground and background of this image.",
    "What is unusual or interesting about this image?",
    "Describe this image as if explaining it to someone who cannot see it.",
    "What category of scene is this (indoor, outdoor, urban, rural, etc.)?",
]


def download_annotations(cache_dir):
    ann_path = os.path.join(cache_dir, "instances_val2017.json")
    if os.path.exists(ann_path):
        print(f"Using cached annotations: {ann_path}")
        with open(ann_path) as f:
            return json.load(f)

    print("Downloading COCO val2017 annotations...")
    resp = requests.get(COCO_ANNOTATIONS_URL, stream=True)
    resp.raise_for_status()

    buf = BytesIO()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=8192):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            print(f"\r  {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        with zf.open("annotations/instances_val2017.json") as f:
            data = json.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump(data, f)
    print(f"Cached annotations to {ann_path}")
    return data


def download_image(file_name, output_dir):
    out_path = os.path.join(output_dir, file_name)
    if os.path.exists(out_path):
        return out_path

    url = COCO_IMAGE_URL.format(file_name=file_name)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build multimodal calibration JSONL from COCO val2017",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--image-dir", required=True,
                        help="Directory to store downloaded images")
    parser.add_argument("--count", type=int, default=200,
                        help="Number of images to include")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache dir for annotations (default: image-dir)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = args.cache_dir or args.image_dir
    os.makedirs(args.image_dir, exist_ok=True)

    ann = download_annotations(cache_dir)
    images = ann["images"]

    rng = random.Random(args.seed)
    selected = rng.sample(images, min(args.count, len(images)))

    print(f"Downloading {len(selected)} images to {args.image_dir}...")
    records = []
    for i, img in enumerate(selected):
        try:
            img_path = download_image(img["file_name"], args.image_dir)
            prompt = PROMPTS[i % len(PROMPTS)]
            record = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": os.path.abspath(img_path)},
                        {"type": "text", "text": prompt},
                    ],
                }],
            }
            records.append(record)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(selected)} images downloaded")
        except Exception as e:
            print(f"  Skipping {img['file_name']}: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
