"""Unified, deterministic loaders for math training and evaluation sets."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DATASET_ALIASES = {
    "aime2025": "aime-2025.jsonl",
    "aime-2025": "aime-2025.jsonl",
    "aime2024": "aime-2024.jsonl",
    "aime-2024": "aime-2024.jsonl",
    "amc23": "amc23.jsonl",
    "amc-23": "amc23.jsonl",
    "math500": "math-500.jsonl",
    "math-500": "math-500.jsonl",
    "dapo": "dapo-math-17k.jsonl",
    "dapo-math-17k": "dapo-math-17k.jsonl",
}


@dataclass(frozen=True)
class MathExample:
    id: str
    prompt: Any
    label: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pick(row: dict[str, Any], names: tuple[str, ...], what: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    raise ValueError(f"dataset row has no {what} field; tried {', '.join(names)}")


def normalize_row(row: dict[str, Any], *, source: str, index: int) -> MathExample:
    prompt = _pick(row, ("prompt", "messages", "conversations", "question", "problem", "task", "input"), "prompt")
    label = row.get("reward_model")
    if label is None:
        label = _pick(row, ("label", "answer", "ground_truth", "ground_truth_answer", "target", "solution"), "label")
    if isinstance(label, dict):
        label = _pick(label, ("ground_truth", "answer", "value", "target"), "ground-truth label")
    explicit_id = row.get("id") or row.get("problem_id") or row.get("uid") or (row.get("extra_info") or {}).get("index")
    key = str(explicit_id) if explicit_id is not None else hashlib.sha256(
        json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    metadata = dict(row.get("metadata") or {})
    for key_name, value in row.items():
        if key_name not in {
            "id", "problem_id", "uid", "prompt", "messages", "conversations",
            "question", "problem", "task", "input", "label", "answer",
            "ground_truth", "ground_truth_answer", "target", "solution",
            "reward_model", "metadata",
        }:
            metadata.setdefault(key_name, value)
    return MathExample(str(key), prompt, str(label), source, metadata)


def _read_jsonl(path: Path, source: str) -> list[MathExample]:
    examples: list[MathExample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index + 1}: expected JSON object")
            examples.append(normalize_row(row, source=source, index=index))
    return examples


def _read_json(path: Path, source: str) -> list[MathExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("examples", [payload]))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list or object with data/examples")
    return [normalize_row(row, source=source, index=i) for i, row in enumerate(payload)]


def _read_huggingface(source: str, *, split: str, config: str | None) -> list[MathExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("loading a HuggingFace dataset requires `pip install datasets`") from exc
    kwargs = {"split": split}
    if config:
        dataset = load_dataset(source, config, **kwargs)
    else:
        dataset = load_dataset(source, **kwargs)
    return [normalize_row(dict(row), source=f"hf:{source}@{split}", index=i) for i, row in enumerate(dataset)]


def deduplicate_rows(rows: Iterable[MathExample]) -> list[MathExample]:
    """Stable de-duplication by normalized question, retaining first label."""

    unique: list[MathExample] = []
    seen: set[str] = set()
    for row in rows:
        prompt_key = hashlib.sha256(
            json.dumps(row.prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if prompt_key in seen:
            continue
        seen.add(prompt_key)
        unique.append(row)
    return unique


def resolve_dataset(name_or_path: str | os.PathLike[str], data_root: str | os.PathLike[str] | None = None) -> Path:
    value = str(name_or_path)
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    root = Path(data_root or os.environ.get("MATH_DATA_ROOT", "benchmarks/math"))
    filename = DATASET_ALIASES.get(value.lower(), value)
    return root / filename


def load_dataset(
    name_or_path: str,
    *,
    data_root: str | os.PathLike[str] | None = None,
    split: str = "train",
    config: str | None = None,
    deduplicate: bool = False,
    limit: int | None = None,
    seed: int = 0,
) -> list[MathExample]:
    """Load local JSON/JSONL or a HF dataset with stable optional sampling."""

    if name_or_path.startswith("hf://"):
        rows = _read_huggingface(name_or_path[5:], split=split, config=config)
    else:
        path = resolve_dataset(name_or_path, data_root)
        if not path.is_file():
            raise FileNotFoundError(f"math dataset not found: {path}; set MATH_DATA_ROOT or pass an explicit path")
        rows = _read_jsonl(path, str(path)) if path.suffix.lower() in {".jsonl", ".ndjson"} else _read_json(path, str(path))
    if deduplicate:
        rows = deduplicate_rows(rows)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit < len(rows):
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(len(rows)), limit))
            rows = [rows[i] for i in indices]
    return rows


def write_jsonl(rows: Iterable[MathExample], path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def write_manifest(rows: Iterable[MathExample], path: str | os.PathLike[str], *, dataset: str, deduplicated: bool) -> dict[str, Any]:
    materialized = list(rows)
    payload = {
        "dataset": dataset,
        "count": len(materialized),
        "unique_questions": len(deduplicate_rows(materialized)),
        "deduplicated": deduplicated,
        "ids": [row.id for row in materialized],
        "rows_sha256": hashlib.sha256(
            "".join(
                json.dumps(row.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for row in materialized
            ).encode("utf-8")
        ).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DATASET_ALIASES",
    "MathExample",
    "deduplicate_rows",
    "load_dataset",
    "normalize_row",
    "resolve_dataset",
    "write_jsonl",
    "write_manifest",
]
