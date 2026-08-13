#!/usr/bin/env python3
"""Apply the preregistered paired gate to SETA fixed48 eval artifacts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


STREAMS = {
    "exploitation": ("seta_fixed48_exploit", 1),
    "exploration": ("seta_fixed48_explore", 8),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _task_key(row: dict[str, Any]) -> str:
    prompt_index = row.get("prompt_index")
    return f"prompt:{prompt_index}" if prompt_index is not None else str(row["task_id"])


def _load_stream(run_dir: Path, dataset: str, step: int, expected_k: int) -> dict[str, float]:
    path = run_dir / "evaluations" / dataset / f"step_{step:04d}" / "tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation artifact: {path}")
    rows = _read_jsonl(path)
    if len(rows) != 48:
        raise ValueError(f"{path}: expected 48 tasks, got {len(rows)}")
    result: dict[str, float] = {}
    for row in rows:
        if int(row.get("k", -1)) != expected_k:
            raise ValueError(f"{path}: task {row.get('task_id')} has k={row.get('k')}, expected {expected_k}")
        key = _task_key(row)
        if key in result:
            raise ValueError(f"{path}: duplicate task key {key}")
        result[key] = float(row["pass_at_k"])
    return result


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    repetitions: int = 10_000,
    seed: int = 20_260_808,
) -> dict[str, float | int]:
    if set(baseline) != set(candidate):
        only_baseline = sorted(set(baseline) - set(candidate))
        only_candidate = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "task pairing mismatch: "
            f"baseline_only={only_baseline[:5]} candidate_only={only_candidate[:5]}"
        )
    keys = sorted(baseline)
    deltas = [candidate[key] - baseline[key] for key in keys]
    rng = random.Random(seed)
    boot = []
    for _ in range(repetitions):
        boot.append(sum(deltas[rng.randrange(len(deltas))] for _ in keys) / len(keys))
    boot.sort()
    return {
        "task_count": len(keys),
        "baseline": sum(baseline.values()) / len(keys),
        "candidate": sum(candidate.values()) / len(keys),
        "delta": sum(deltas) / len(keys),
        "ci95_low": _quantile(boot, 0.025),
        "ci95_high": _quantile(boot, 0.975),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def compare_runs(
    dapo_dir: Path,
    dive_dir: Path,
    step: int,
    *,
    repetitions: int = 10_000,
    seed: int = 20_260_808,
    noninferiority_margin: float = 0.05,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for offset, (dimension, (dataset, expected_k)) in enumerate(STREAMS.items()):
        dapo = _load_stream(dapo_dir, dataset, step, expected_k)
        dive = _load_stream(dive_dir, dataset, step, expected_k)
        result = paired_bootstrap(
            dapo,
            dive,
            repetitions=repetitions,
            seed=seed + offset,
        )
        result["metric"] = f"pass@{expected_k}"
        result["significantly_better"] = result["ci95_low"] > 0.0
        result["noninferior"] = result["ci95_low"] >= -noninferiority_margin
        metrics[dimension] = result

    passes = (
        metrics["exploitation"]["significantly_better"]
        and metrics["exploration"]["noninferior"]
    ) or (
        metrics["exploration"]["significantly_better"]
        and metrics["exploitation"]["noninferior"]
    )
    return {
        "protocol": "seta_fixed48_v2",
        "step": step,
        "candidate": "DIVE-PO",
        "baseline": "DAPO",
        "noninferiority_margin": noninferiority_margin,
        "metrics": metrics,
        "verdict": "优" if passes else "不优",
        "action": "continue_to_1000" if passes else "stop_dive_immediately",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapo-run-dir", type=Path, required=True)
    parser.add_argument("--dive-run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(100, 200), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_runs(args.dapo_run_dir, args.dive_run_dir, args.step)
    payload = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
