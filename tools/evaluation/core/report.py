"""Comparison report over multiple normalized ``eval_result.json`` files."""

from __future__ import annotations

import csv
import glob
import io
import json
from pathlib import Path


def load_results(patterns: list[str]) -> list[dict]:
    """Load eval_result.json dicts from paths and/or glob patterns."""
    results: list[dict] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).is_file() else [])
        for match in matches:
            data = json.loads(Path(match).read_text(encoding="utf-8"))
            data["_source"] = match
            results.append(data)
    if not results:
        raise ValueError(f"no eval_result.json matched: {patterns}")
    return results


def _top_exceptions(result: dict, limit: int = 2) -> str:
    counts = result.get("exception_counts") or {}
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return ", ".join(f"{name}x{count}" for name, count in ranked)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(results: list[dict]) -> str:
    headers = ["model", "harness", "pass@1", "mean_reward", "completed", "errored", "top_exceptions"]
    rows = [
        [
            result.get("model_name", ""),
            result.get("harness", ""),
            _fmt(result.get("pass_at_1")),
            _fmt(result.get("mean_reward")),
            _fmt(result.get("n_completed")),
            _fmt(result.get("n_errored")),
            _top_exceptions(result),
        ]
        for result in results
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def render_csv(results: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["model", "harness", "job_name", "dataset", "pass_at_1", "mean_reward",
         "reward_best_at_k", "k", "task_count", "n_completed", "n_errored",
         "exception_counts", "source"]
    )
    for result in results:
        writer.writerow(
            [
                result.get("model_name", ""),
                result.get("harness", ""),
                result.get("job_name", ""),
                result.get("dataset", ""),
                result.get("pass_at_1"),
                result.get("mean_reward"),
                result.get("reward_best_at_k"),
                result.get("k"),
                result.get("task_count"),
                result.get("n_completed"),
                result.get("n_errored"),
                json.dumps(result.get("exception_counts") or {}, ensure_ascii=False),
                result.get("_source", ""),
            ]
        )
    return buffer.getvalue()


def write_report(results: list[dict], output_prefix: str) -> tuple[Path, Path]:
    """Write ``<output_prefix>.md`` and ``<output_prefix>.csv``."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = prefix.with_suffix(".md")
    csv_path = prefix.with_suffix(".csv")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    csv_path.write_text(render_csv(results), encoding="utf-8")
    return md_path, csv_path
