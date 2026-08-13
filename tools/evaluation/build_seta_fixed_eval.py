#!/usr/bin/env python3
"""Build a reproducible, stratified SETA held-out evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


TASK_PATTERNS = (
    re.compile(r"\[task=(\d+)(?:\s|\])"),
    re.compile(r"task_name[=:][\"' ]*(\d+)"),
    re.compile(r"task_path[=:][\"' ]*seta_env/(\d+)"),
    re.compile(r"seta_task-(\d+)(?:-|_)"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    task_id = str(metadata.get("task_name", "")).strip()
    if not task_id.isdigit():
        raise ValueError(f"row has no numeric metadata.task_name: {metadata!r}")
    return task_id


def _task_stratum(task_root: Path, task_id: str) -> tuple[str, str]:
    task_path = task_root / task_id / "task.yaml"
    task = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    difficulty = str(task.get("difficulty", "missing")).strip().lower()
    category = str(task.get("category", "missing")).strip().lower()
    return difficulty, category


def _seen_task_ids(log_paths: list[Path]) -> set[str]:
    seen: set[str] = set()
    for path in log_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in TASK_PATTERNS:
            seen.update(pattern.findall(text))
    return seen


def _rank(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_additions(
    *,
    existing_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    task_root: Path,
    difficulty_targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    strata = {
        _task_id(row): _task_stratum(task_root, _task_id(row))
        for row in existing_rows + candidate_rows
    }
    existing_by_difficulty = Counter(strata[_task_id(row)][0] for row in existing_rows)
    population_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selected_by_category: dict[str, Counter[str]] = defaultdict(Counter)

    for row in existing_rows + candidate_rows:
        difficulty, category = strata[_task_id(row)]
        population_by_category[difficulty][category] += 1
    for row in existing_rows:
        difficulty, category = strata[_task_id(row)]
        selected_by_category[difficulty][category] += 1
    for row in candidate_rows:
        difficulty, category = strata[_task_id(row)]
        candidates[(difficulty, category)].append(row)
    for rows in candidates.values():
        rows.sort(key=lambda row: (_rank(seed, _task_id(row)), int(_task_id(row))))

    additions: list[dict[str, Any]] = []
    for difficulty, target in difficulty_targets.items():
        needed = target - existing_by_difficulty[difficulty]
        if needed < 0:
            raise ValueError(
                f"existing split already exceeds target for {difficulty}: "
                f"{existing_by_difficulty[difficulty]} > {target}"
            )
        population = population_by_category[difficulty]
        population_total = sum(population.values())
        for _ in range(needed):
            available = [
                category
                for (candidate_difficulty, category), rows in candidates.items()
                if candidate_difficulty == difficulty and rows
            ]
            if not available:
                raise ValueError(f"not enough unseen candidates for {difficulty}")
            # Fill the category with the largest deficit relative to the target
            # population mix. Stable lexical/hash tie breaks make this reproducible.
            category = max(
                available,
                key=lambda item: (
                    target * population[item] / population_total
                    - selected_by_category[difficulty][item],
                    -int(_rank(seed, item)[:12], 16),
                ),
            )
            row = candidates[(difficulty, category)].pop(0)
            additions.append(row)
            selected_by_category[difficulty][category] += 1
    return additions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    existing_rows = _read_jsonl(args.existing_eval)
    training_rows = _read_jsonl(args.training_data)
    existing_ids = {_task_id(row) for row in existing_rows}
    seen_ids = _seen_task_ids(args.seen_logs)
    candidate_rows = [
        row
        for row in training_rows
        if _task_id(row) not in existing_ids and _task_id(row) not in seen_ids
    ]
    difficulty_targets = {
        difficulty: int(count)
        for difficulty, count in (
            item.split("=", 1) for item in args.difficulty_targets.split(",")
        )
    }
    additions = _select_additions(
        existing_rows=existing_rows,
        candidate_rows=candidate_rows,
        task_root=args.task_root,
        difficulty_targets=difficulty_targets,
        seed=args.seed,
    )
    eval_rows = existing_rows + additions
    eval_ids = {_task_id(row) for row in eval_rows}
    output_training_rows = [row for row in training_rows if _task_id(row) not in eval_ids]
    _write_jsonl(args.output_eval, eval_rows)
    _write_jsonl(args.output_training, output_training_rows)

    task_records = []
    for row in eval_rows:
        task_id = _task_id(row)
        difficulty, category = _task_stratum(args.task_root, task_id)
        task_records.append(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "category": category,
                "source": "fixed12" if task_id in existing_ids else "unseen_stratified",
                "selection_rank": _rank(args.seed, task_id),
            }
        )
    manifest = {
        "schema": "seta.fixed_eval_manifest.v1",
        "seed": args.seed,
        "difficulty_targets": difficulty_targets,
        "input": {
            "existing_eval": str(args.existing_eval),
            "existing_eval_sha256": _sha256(args.existing_eval),
            "training_data": str(args.training_data),
            "training_data_sha256": _sha256(args.training_data),
            "seen_logs": [
                {"path": str(path), "sha256": _sha256(path)} for path in args.seen_logs
            ],
        },
        "seen_task_count": len(seen_ids),
        "eval_task_count": len(eval_rows),
        "training_task_count": len(output_training_rows),
        "tasks": task_records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-eval", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--seen-log", dest="seen_logs", type=Path, action="append", required=True)
    parser.add_argument("--output-eval", type=Path, required=True)
    parser.add_argument("--output-training", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--difficulty-targets", default="hard=25,medium=22,easy=1")
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({key: result[key] for key in ("seed", "seen_task_count", "eval_task_count", "training_task_count")}, sort_keys=True))
