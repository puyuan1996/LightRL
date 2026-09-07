"""Portable data-root discovery for Math RLVR datasets.

The evaluator and training launchers accept an explicit ``MATH_DATA_ROOT``.
When it is omitted, this module searches the canonical shared-data location,
the repository-local data directory. Keeping the fallback policy here prevents
each launcher from growing a different hard-coded path list.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SHARED_ROOT = Path("/mnt/shared-storage-user/puyuan/data/math_rlvr")
REPO_DATA_ROOT = REPO_ROOT / "data" / "math_rlvr"
REPO_BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "math"


def _has_math_data(path: Path) -> bool:
    return any((path / filename).is_file() for filename in ("aime-2025.jsonl", "dapo-math-17k.jsonl"))


def data_root_candidates(explicit: str | os.PathLike[str] | None = None) -> tuple[Path, ...]:
    """Return ordered data-root candidates, preserving explicit overrides."""

    values: list[Path] = []
    if explicit:
        values.append(Path(explicit).expanduser())
    for variable in ("MATH_DATA_ROOT", "LIGHTRL_DATA_ROOT"):
        value = os.environ.get(variable)
        if value:
            values.append(Path(value).expanduser())
    values.extend((CANONICAL_SHARED_ROOT, REPO_DATA_ROOT, REPO_BENCHMARK_ROOT))
    # Preserve order while removing duplicate spellings.
    return tuple(dict.fromkeys(values))


def resolve_data_root(
    explicit: str | os.PathLike[str] | None = None,
    *,
    require_exists: bool = False,
) -> Path:
    """Resolve the first usable Math RLVR root.

    If no candidate exists, return the canonical shared path unless
    ``require_exists`` is requested, in which case raise a useful error.
    """

    override = explicit or os.environ.get("MATH_DATA_ROOT") or os.environ.get("LIGHTRL_DATA_ROOT")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir() and _has_math_data(candidate):
            return candidate
        if require_exists:
            raise FileNotFoundError(f"configured Math RLVR data root does not exist or is incomplete: {candidate}")
        return candidate

    candidates = data_root_candidates()
    for candidate in candidates:
        if candidate.is_dir() and _has_math_data(candidate):
            return candidate
    if require_exists:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"no Math RLVR data root found; searched: {searched}")
    return candidates[0]


def dataset_path(name_or_path: str | os.PathLike[str], data_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a dataset alias or explicit path against the configured root."""

    value = str(name_or_path)
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    aliases = {
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
    return resolve_data_root(data_root) / aliases.get(value.lower(), value)


__all__ = ["CANONICAL_SHARED_ROOT", "REPO_DATA_ROOT", "data_root_candidates", "dataset_path", "resolve_data_root"]
