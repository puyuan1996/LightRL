#!/usr/bin/env python3
"""Prepare math RLVR datasets for slime / OpenClaw-RL in the format used by
toolcall-rl (ReTool) and slime's native math path:

    {"prompt": [{"role": "user", "content": "..."}], "label": "<gt answer>"}

Consumed by slime with:
    --input-key prompt --label-key label --apply-chat-template
    rm_type = "dapo"   (answer extracted with  (?i)Answer\\s*:\\s*([^\\n]+) )

Outputs (under OUT_DIR):
    dapo-math-17k/dapo-math-17k.jsonl   train prompts (deduplicated)
    aime-2025/aime-2025.jsonl           eval, 30 problems, AIME 2025 I+II
    aime-2024/aime-2024.jsonl           eval, 30 problems (reference / comparability)

IMPORTANT: the eval prompt template is made byte-identical (same instruction
prefix/suffix) to the training template, otherwise the DAPO verifier regex sees
a different answer format at eval time and under-reports accuracy.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("MATH_DATA_ROOT", REPO / "benchmarks" / "math"))

# Exact instruction wrapper used by BytedTsinghua-SIA/DAPO-Math-17k
# (style "rule-lighteval/MATH_v2"). Reused verbatim for AIME eval sets.
PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is the answer to the problem.\n\n"
)
SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def wrap(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{PREFIX}{problem.strip()}{SUFFIX}"}]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {path}  rows={len(rows)}")


def strip_wrapper(content: str) -> str:
    """Recover the bare problem statement from a DAPO-Math-17k prompt."""
    body = content
    if body.startswith(PREFIX):
        body = body[len(PREFIX) :]
    if body.endswith(SUFFIX):
        body = body[: -len(SUFFIX)]
    return body.strip()


def prepare_dapo_math_17k() -> None:
    from datasets import load_dataset

    # HF ships this dataset pre-expanded (~1.79M rows = 17,398 unique prompts
    # repeated ~103x for multi-epoch verl runs). Deduplicate, or one "epoch"
    # silently becomes ~103 epochs over the same prompts.
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    print(f"[dapo-math-17k] raw rows = {len(ds)}")

    seen: set[str] = set()
    rows: list[dict] = []
    for rec in ds:
        prompt = rec["prompt"]
        content = prompt[0]["content"] if isinstance(prompt, list) else str(prompt)
        gt = str(rec["reward_model"]["ground_truth"]).strip()
        key = strip_wrapper(content)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"prompt": wrap(key), "label": gt})
    print(f"[dapo-math-17k] unique prompts = {len(rows)}")
    write_jsonl(rows, OUT_DIR / "dapo-math-17k" / "dapo-math-17k.jsonl")


def prepare_aime2025() -> None:
    from datasets import load_dataset

    ds = load_dataset("MathArena/aime_2025", split="train")
    rows = []
    for rec in ds:
        rows.append(
            {
                "prompt": wrap(rec["problem"]),
                "label": str(rec["answer"]).strip(),
                "problem_idx": rec.get("problem_idx"),
                "problem_type": rec.get("problem_type"),
            }
        )
    assert len(rows) == 30, f"expected 30 AIME2025 problems, got {len(rows)}"
    write_jsonl(rows, OUT_DIR / "aime-2025" / "aime-2025.jsonl")


def prepare_aime2024() -> None:
    from datasets import load_dataset

    ds = load_dataset("zhuzilin/aime-2024", split="train")
    rows = []
    for rec in ds:
        prompt = rec["prompt"]
        content = prompt[0]["content"] if isinstance(prompt, list) else str(prompt)
        rows.append({"prompt": wrap(content), "label": str(rec["label"]).strip()})
    write_jsonl(rows, OUT_DIR / "aime-2024" / "aime-2024.jsonl")


def prepare_amc23() -> None:
    from datasets import load_dataset

    ds = load_dataset("math-ai/amc23", split="test")
    rows = [{"prompt": wrap(r["question"]), "label": str(r["answer"]).strip()} for r in ds]
    write_jsonl(rows, OUT_DIR / "amc23" / "amc23.jsonl")


def prepare_math500() -> None:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [
        {
            "prompt": wrap(r["problem"]),
            "label": str(r["answer"]).strip(),
            "subject": r.get("subject"),
            "level": r.get("level"),
        }
        for r in ds
    ]
    write_jsonl(rows, OUT_DIR / "math-500" / "math-500.jsonl")


def sanity_check_verifier() -> None:
    """Confirm the shipped DAPO verifier accepts our template's answer form."""
    import sys

    repo = REPO
    sys.path.insert(0, str(repo / "slime"))
    from slime.rollout.rm_hub import compute_score_dapo

    good = compute_score_dapo("... reasoning ...\nAnswer: 70", "70")
    bad = compute_score_dapo("... reasoning ...\nAnswer: 71", "70")
    boxed = compute_score_dapo("... reasoning ...\n\\boxed{70}", "70")
    print(f"[verifier] Answer:70 vs 70 -> {good}")
    print(f"[verifier] Answer:71 vs 70 -> {bad}")
    print(f"[verifier] \\boxed{{70}} vs 70 -> {boxed}   <-- boxed-only WITHOUT 'Answer:' line")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=["train", "aime2025", "aime2024", "amc23", "math500", "verifier"],
        default=None,
    )
    args = ap.parse_args()

    if args.only in (None, "aime2025"):
        prepare_aime2025()
    if args.only in (None, "aime2024"):
        prepare_aime2024()
    if args.only in (None, "amc23"):
        prepare_amc23()
    if args.only in (None, "math500"):
        prepare_math500()
    if args.only in (None, "verifier"):
        sanity_check_verifier()
    if args.only in (None, "train"):
        prepare_dapo_math_17k()
