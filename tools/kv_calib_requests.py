#!/usr/bin/env python3
"""Send calibration dataset to a running server for KV cache scale calibration.

Fires requests from a JSONL file at an OpenAI-compatible chat endpoint.
Only the prefill matters for KV scale calibration, so max_tokens is kept low.

Usage:
    python kv_calib_requests.py --jsonl data/text/agentic_coding_calib_v3.jsonl \\
        --model Qwen3.5 --limit 512 --concurrency 16
"""

import argparse
import asyncio
import json
import time

import aiohttp

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--jsonl", required=True, help="Calibration JSONL file.")
ap.add_argument("--limit", type=int, default=512, help="Max samples to send.")
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=128,
                help="Max tokens to generate (only prefill matters).")
ap.add_argument("--base-url", default="http://localhost:8000")
ap.add_argument("--model", required=True)
args = ap.parse_args()


def load_messages(path, limit):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            j = json.loads(line)
            if "messages" in j:
                out.append(j["messages"])
            elif "prompt" in j or "text" in j:
                text = j.get("prompt") or j.get("text")
                out.append([{"role": "user", "content": text}])
    return out


async def send_one(session, sem, url, messages, idx, total):
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    async with sem:
        try:
            async with session.post(url, json=payload) as resp:
                await resp.read()
                status = resp.status
        except Exception as e:
            status = str(e)
    if (idx + 1) % 50 == 0 or idx + 1 == total:
        print(f"  {idx + 1}/{total} (status={status})")


async def main():
    messages_list = load_messages(args.jsonl, args.limit)
    total = len(messages_list)
    url = f"{args.base_url}/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)

    print(f"Sending {total} requests to {args.base_url} (concurrency={args.concurrency})")
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_one(session, sem, url, msgs, i, total)
            for i, msgs in enumerate(messages_list)
        ]
        await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    print(f"Done. {total} requests in {elapsed:.1f}s ({total / elapsed:.1f} req/s)")


if __name__ == "__main__":
    asyncio.run(main())
