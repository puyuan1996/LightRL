#!/usr/bin/env python3
"""Compatibility entry point for the modular Math RLVR evaluator."""

from __future__ import annotations

import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.math_rlvr.eval_runner import main, evaluate, write_outputs  # noqa: E402
from tools.evaluation.math_rlvr.scorer import avg_pass_at_k  # noqa: E402
from tools.evaluation.math_rlvr.extractor import extract_answers  # noqa: E402
from tools.evaluation.math_rlvr.verifier import Verifier  # noqa: E402

REPO = _ROOT
DATA_ROOT = Path(os.environ.get("MATH_DATA_ROOT", REPO / "benchmarks" / "math"))
# PR #1 called the asynchronous driver ``run``; keep that name for callers
# while the implementation now lives in the modular ``evaluate`` function.
run = evaluate


def lenient_acc(text: str, ground_truth: object) -> bool:
    """Compatibility helper: semantic answer accuracy across all formats."""

    return Verifier("math").verify(text, ground_truth).correct


def compute_score_dapo(text: str, ground_truth: object) -> dict:
    """Drop-in shape of Slime's reward helper without importing optional Ray deps."""

    result = Verifier("dapo").verify(text, ground_truth)
    return {"score": 1.0 if result.correct else -1.0, "acc": result.correct, "pred": result.extracted}


def extract_boxed_answer(text: str) -> str | None:
    """Compatibility helper returning the last boxed candidate, if present."""

    result = extract_answers(text)
    boxed = [candidate for candidate in result.candidates if candidate.format == "boxed" and candidate.closed]
    return boxed[-1].value if boxed else None


def grade_answer_verl(text: str, ground_truth: object) -> bool:
    """Compatibility semantic verifier name used by older evaluation notebooks."""

    return Verifier("math").verify(text, ground_truth).correct


def avg_and_pass_at_k(per_problem, keep_fn=lambda _sample: True, key="acc"):
    """Compatibility name used by PR #1 tests and downstream notebooks."""

    # The original helper used ``acc``; modern records use explicit track keys.
    normalized = []
    for problem in per_problem:
        samples = []
        for sample in problem.get("samples", []):
            item = dict(sample)
            if key not in item and "configured_correct" in item:
                item[key] = item["configured_correct"]
            samples.append(item)
        normalized.append({**problem, "samples": samples})
    return avg_pass_at_k(normalized, key=key, keep=keep_fn)


if __name__ == "__main__":
    raise SystemExit(main())
