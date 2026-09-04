"""Paired, deterministic statistics for Math RLVR runs."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


def _sample_map(payload: dict[str, Any], key: str = "lenient_correct") -> dict[str, float]:
    result: dict[str, float] = {}
    for problem in payload.get("problems", payload.get("per_problem", [])):
        pid = str(problem.get("id", problem.get("problem_id", len(result))))
        for index, sample in enumerate(problem.get("samples", [])):
            result[f"{pid}:{index}"] = float(bool(sample.get(key)))
    return result


def paired_deltas(baseline: dict[str, Any], candidate: dict[str, Any], *, key: str = "lenient_correct") -> list[float]:
    left, right = _sample_map(baseline, key), _sample_map(candidate, key)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("no shared sample ids; paired comparison requires identical prompts and n")
    return [right[sample_id] - left[sample_id] for sample_id in shared]


def bootstrap_ci(values: Iterable[float], *, seed: int = 0, repetitions: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    values = list(values)
    if not values:
        return (0.0, 0.0)
    if repetitions <= 0 or not 0 < alpha < 1:
        raise ValueError("repetitions must be positive and alpha in (0, 1)")
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(repetitions)]
    means.sort()
    return means[max(0, math.floor(alpha / 2 * repetitions))], means[min(repetitions - 1, math.ceil((1 - alpha / 2) * repetitions) - 1)]


def compare_payloads(baseline: dict[str, Any], candidate: dict[str, Any], *, key: str = "lenient_correct", seed: int = 0) -> dict[str, Any]:
    deltas = paired_deltas(baseline, candidate, key=key)
    ci_low, ci_high = bootstrap_ci(deltas, seed=seed)
    return {
        "metric": key,
        "n_paired": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "bootstrap_ci_95": [ci_low, ci_high],
    }


def compare_files(baseline: str | Path, candidate: str | Path, *, key: str = "lenient_correct", seed: int = 0) -> dict[str, Any]:
    base = json.loads(Path(baseline).read_text(encoding="utf-8"))
    cand = json.loads(Path(candidate).read_text(encoding="utf-8"))
    result = compare_payloads(base, cand, key=key, seed=seed)
    result["baseline"] = str(baseline)
    result["candidate"] = str(candidate)
    return result


__all__ = ["bootstrap_ci", "compare_files", "compare_payloads", "paired_deltas"]
