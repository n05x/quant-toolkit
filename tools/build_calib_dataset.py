#!/usr/bin/env python3
"""Build calibration datasets for MoE model quantization.

Pulls from multiple HuggingFace datasets to activate all expert modules.
Supports coding-focused and diverse conversation modes.

Usage:
    python build_calib_dataset.py --mode coding    # coding-focused (default)
    python build_calib_dataset.py --mode diverse   # broad natural conversation coverage
"""

import argparse
import json
import random
import re

from datasets import load_dataset

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mode", choices=["coding", "diverse"], default="coding")
ap.add_argument("--total", type=int, default=16384)
ap.add_argument("--output", type=str, required=True, help="Output JSONL path.")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)

CODING_SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a knowledgeable and friendly AI assistant.",
    "You are an expert software engineer. Write clean, efficient, well-tested code.",
    "You are a coding assistant that helps debug and fix issues in code repositories.",
    "You are a helpful AI that can search the web and use tools to answer questions.",
    "You are a data analyst. Help the user with spreadsheets, charts, and financial models.",
    "You are a multilingual assistant fluent in English, Chinese, and other languages.",
    "You are a senior developer doing code review. Be thorough and constructive.",
]

DIVERSE_SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a knowledgeable and friendly AI assistant.",
    "You are a helpful AI assistant. Answer questions clearly and concisely.",
    "You are an AI assistant. Be helpful, harmless, and honest.",
]

SYSTEM_PROMPTS = DIVERSE_SYSTEM_PROMPTS if args.mode == "diverse" else CODING_SYSTEM_PROMPTS

# Dataset plans by mode (weights must sum to 1.0).
# fmt: off
CODING_PLAN = [
    # (dataset_name, data_dir, split, weight)

    # Code — largest share, primary use case.
    ("microsoft/orca-agentinstruct-1M-v1", "", "code_",    0.15),
    ("sahil2801/CodeAlpaca-20k", "", "train",              0.06),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "", "train", 0.06),

    # Multilingual code (Go, Rust, C/C++, TypeScript, etc.)
    ("bigcode/starcoderdata", "", "train",                 0.10),

    # Tool calling / function calling.
    ("NousResearch/hermes-function-calling-v1", "", "train", 0.12),
    ("glaiveai/glaive-function-calling-v2", "", "train",   0.08),

    # Reasoning and general.
    ("microsoft/orca-agentinstruct-1M-v1", "", "analytical_reasoning", 0.04),
    ("microsoft/orca-agentinstruct-1M-v1", "", "creative_content",     0.03),
    ("microsoft/orca-agentinstruct-1M-v1", "", "follow_up",           0.05),
    ("Open-Orca/OpenOrca", "", "train",                    0.06),

    # Math.
    ("openai/gsm8k", "main", "train",                     0.04),
    ("TIGER-Lab/MathInstruct", "", "train",                0.04),

    # Chinese.
    ("shibing624/alpaca-zh", "", "train",                  0.08),
    ("m-a-p/COIG-CQIA", "segmentfault", "train",          0.06),
    ("wangrui6/Zhihu-KOL", "", "train",                   0.03),
]

DIVERSE_PLAN = [
    # Real user conversations — maximally diverse topics (pre-filtered for toxicity).
    ("allenai/WildChat-nontoxic", "", "train",             0.60),

    # Chatbot Arena — real multi-model conversations.
    ("lmsys/lmsys-chat-1m", "", "train",                   0.40),
]
# fmt: on

PLAN = DIVERSE_PLAN if args.mode == "diverse" else CODING_PLAN
assert abs(sum(w for *_, w in PLAN) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0 (got {sum(w for *_, w in PLAN)})"

_GH_TOKEN_RE = re.compile(
    r"<(reponame|gh_stars|filename|issue_start|issue_comment|issue_closed)>[^\n]*\n?"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_CONTENT_CHARS = 30000


def normalize_to_messages(messages):
    """Normalize various message formats to standard OpenAI-style messages."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (json.JSONDecodeError, ValueError):
            return None

    if not isinstance(messages, list):
        return None

    role_map = {
        "human": "user", "user": "user",
        "gpt": "assistant", "assistant": "assistant", "ai": "assistant",
        "system": "system",
        "tool": "tool", "function": "tool",
    }

    normalized = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = role_map.get(m.get("role", m.get("from", "")))
        content = m.get("content", m.get("value", ""))
        if role and content:
            normalized.append({"role": role, "content": content.strip()})

    return normalized if normalized else None


def extract_text(ex):
    """Extract a message list from a dataset example, handling many formats."""
    if isinstance(ex, str):
        ex = json.loads(ex)

    # Messages format (Orca, OpenOrca, Hermes).
    if "messages" in ex:
        return normalize_to_messages(ex["messages"])

    # Conversations format (Glaive, Firefly).
    if "conversations" in ex or "conversation" in ex:
        convs = ex.get("conversations") or ex.get("conversation")
        if isinstance(convs, list):
            return normalize_to_messages(convs)

    # Instruction/output format (Alpaca, CodeAlpaca).
    if "instruction" in ex:
        inst = ex.get("instruction", "")
        inp = ex.get("input", "")
        if inp:
            inst = f"{inst}\n{inp}"
        messages = [{"role": "user", "content": inst}]
        if ex.get("output"):
            messages.append({"role": "assistant", "content": ex["output"]})
        return messages

    # Question/answer format (GSM8K, Math).
    if "question" in ex:
        messages = [{"role": "user", "content": ex["question"]}]
        if ex.get("answer"):
            messages.append({"role": "assistant", "content": ex["answer"]})
        return messages

    # Problem/solution format.
    if "problem" in ex:
        messages = [{"role": "user", "content": ex["problem"]}]
        if ex.get("solution"):
            messages.append({"role": "assistant", "content": ex["solution"]})
        return messages

    # Glaive format: "system" + "chat" fields with USER:/ASSISTANT: prefixes.
    if "system" in ex and "chat" in ex:
        messages = []
        sys_text = ex["system"].replace("SYSTEM: ", "", 1).strip()
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        for turn in ex["chat"].split("\n\n"):
            turn = turn.strip()
            if turn.startswith("USER:"):
                messages.append({"role": "user", "content": turn[5:].strip()})
            elif turn.startswith("A:") or turn.startswith("ASSISTANT:"):
                content = turn.split(":", 1)[1].strip().replace("<|endoftext|>", "").strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
        if len(messages) > 1:
            return messages

    # Raw text (starcoderdata, etc.) — wrap as user message.
    for key in ("content", "text"):
        if key in ex:
            txt = ex[key]
            if isinstance(txt, str) and len(txt) > 50:
                return [{"role": "user", "content": txt[:2048]}]

    return None


def add_system_prompt(messages):
    """Randomly prepend a system prompt to ~50% of samples that don't already have one."""
    if messages and messages[0].get("role") != "system" and random.random() < 0.5:
        messages = [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + messages
    return messages


def clean_content(text):
    """Strip BOMs, starcoderdata tokens, and stray control chars."""
    text = text.replace("\ufeff", "").replace("\ufffe", "")
    text = _GH_TOKEN_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return text.strip()


def iter_prompts_from_split(dataset_name, dir_name, split_name, n):
    try:
        ds = load_dataset(dataset_name, split=split_name, streaming=True,
                          data_dir=dir_name or None)
        ds = ds.shuffle(seed=args.seed, buffer_size=10000)
    except Exception as e:
        print(f"Warning: Could not load {dataset_name} split {split_name}: {e}, skipping...")
        return

    picked = 0
    for ex in ds:
        messages = extract_text(ex)
        if messages:
            messages = add_system_prompt(messages)
            yield {"messages": messages}
            picked += 1
            if picked >= n:
                break


def main():
    # Collect according to the plan.
    samples = []
    for dataset_name, dir_name, split_name, weight in PLAN:
        n = max(1, round(args.total * weight))
        print(f"Loading {n} samples from {dataset_name}/{split_name} ({weight:.0%})...")
        batch = list(iter_prompts_from_split(dataset_name, dir_name, split_name, n))
        print(f"  Got {len(batch)}")
        samples.extend(batch)

    random.shuffle(samples)

    # Post-collection cleaning pass.
    cleaned = []
    dropped_long = 0
    dropped_empty = 0
    for ex in samples:
        new_msgs = []
        total_len = 0
        for m in ex["messages"]:
            c = clean_content(m.get("content", "") or "")
            if c:
                new_msgs.append({"role": m["role"], "content": c})
                total_len += len(c)
        if total_len > MAX_CONTENT_CHARS:
            dropped_long += 1
            continue
        if not new_msgs or not any(m["role"] == "user" for m in new_msgs):
            dropped_empty += 1
            continue
        cleaned.append({"messages": new_msgs})

    print(f"\nCleaning: dropped {dropped_long} too-long, {dropped_empty} empty. "
          f"{len(samples)} -> {len(cleaned)}")

    out_path = args.output
    with open(out_path, "w") as f:
        for ex in cleaned:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cleaned)} prompts to {out_path}")


if __name__ == "__main__":
    main()
