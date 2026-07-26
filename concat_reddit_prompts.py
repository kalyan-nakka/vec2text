#!/usr/bin/env python3

import json
from datasets import load_dataset

OUT = "data/reddit_prompts.jsonl"

ds = load_dataset("allenai/real-toxicity-prompts", split="train")

n = 0
with open(OUT, "w", encoding="utf-8") as f:
    for i, ex in enumerate(ds):
        prompt = ex.get("prompt") or {}
        continuation = ex.get("continuation") or {}

        p = prompt.get("text") or ""
        c = continuation.get("text") or ""
        text = (p + c).strip()
        if not text:
            continue

        record = {
            "id": i,
            "text": text,
            "prompt_text": p,
            "continuation_text": c,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n += 1

print(f"Wrote {n} rows to {OUT}")