#!/usr/bin/env python3
"""Plot Math RLVR train reward and OOD eval reward curves for one run.

Primary source is the durable structured log ``<run_dir>/logs/metrics.jsonl``
(``terminal_rl.rollout_metrics.v1`` and ``terminal_rl.eval_dataset_metrics.v1``
records).  Runs started before those records existed fall back to parsing the
``data.py - rollout N: {...}`` and ``rollout.py - eval N: {...}`` lines of
``<run_dir>/logs/train.log``.

Outputs:
  math_rlvr_curves.png   train reward / OOD eval reward / truncation / logprob diff
  math_rlvr_summary.json per-series stats used by the figures

Usage:
  python tools/analysis/plot_math_rlvr_curves.py --run-dir runs/training/<run_id>

Optional:
  --log-file PATH  Override (default <run_dir>/logs/train.log)
  --out-dir DIR    Override output (default <run_dir>/metrics/analysis)
  --no-figs        Only emit math_rlvr_summary.json (no matplotlib needed)

Exits 0 on success, 1 if neither source exists, 2 if no usable series.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROLLOUT_RE = re.compile(r"data\.py:\d+ - rollout (\d+): (\{.+\})")
EVAL_RE = re.compile(r"rollout\.py:\d+ - eval (\d+): (\{.+\})")

TRAIN_REWARD_KEYS = ("raw_reward", "rewards")
TRUNC_KEYS = ("truncated", "truncated_ratio")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _series_stats(points: list[tuple[int, float]]) -> dict[str, Any]:
    if not points:
        return {}
    values = [y for _, y in points]
    return {
        "n_points": len(points),
        "first_x": points[0][0],
        "last_x": points[-1][0],
        "first": values[0],
        "last": values[-1],
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _load_jsonl(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, list[tuple[int, dict[str, Any]]]], list[dict[str, Any]]]:
    """Return (train rollout metrics by id, eval records by dataset, actor rows)."""
    train: dict[int, dict[str, Any]] = {}
    evals: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    actor: list[dict[str, Any]] = []
    if not path.is_file():
        return train, evals, actor
    for line in path.open(errors="replace"):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        schema = record.get("schema")
        rollout_id = record.get("rollout_id")
        if schema == "terminal_rl.rollout_metrics.v1" and rollout_id is not None:
            metrics = record.get("metrics")
            if isinstance(metrics, dict):
                train[int(rollout_id)] = metrics
        elif schema == "terminal_rl.eval_dataset_metrics.v1" and rollout_id is not None:
            dataset = str(record.get("dataset") or "unknown")
            evals.setdefault(dataset, []).append((int(rollout_id), record))
        elif schema == "terminal_rl.actor_update_metrics.v1":
            metrics = record.get("metrics")
            if isinstance(metrics, dict) and rollout_id is not None:
                actor.append({"rollout_id": int(rollout_id), **metrics})
    for dataset in evals:
        evals[dataset].sort(key=lambda item: item[0])
    return train, evals, actor


def _parse_train_log(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, list[tuple[int, dict[str, Any]]]]]:
    """Legacy fallback for runs without structured rollout/eval JSONL records."""
    train: dict[int, dict[str, Any]] = {}
    evals: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    if not path.is_file():
        return train, evals
    for line in path.open(errors="replace"):
        m = ROLLOUT_RE.search(line)
        if m:
            try:
                payload = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                continue
            train[int(m.group(1))] = {
                str(k).removeprefix("rollout/"): v for k, v in payload.items()
            }
            continue
        m = EVAL_RE.search(line)
        if m:
            try:
                payload = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                continue
            rollout_id = int(m.group(1))
            rests = {
                str(key)[len("eval/"):]: value
                for key, value in payload.items()
                if str(key).startswith("eval/")
            }
            # Dataset names may themselves contain dashes ("aime-2024"), so
            # resolve plain reward keys and "/" sub-metrics first, then match
            # "-"-suffixed metrics against the longest known dataset prefix.
            datasets = {rest for rest in rests if "/" not in rest and "-" not in rest}
            datasets |= {rest.split("/", 1)[0] for rest in rests if "/" in rest}
            top: dict[str, dict[str, Any]] = {}
            for rest, value in rests.items():
                if "/" in rest:
                    dataset, field = rest.split("/", 1)
                elif rest in datasets:
                    dataset, field = rest, "reward"
                else:
                    dataset = next(
                        (d for d in sorted(datasets, key=len, reverse=True) if rest.startswith(d + "-")),
                        rest.split("-", 1)[0],
                    )
                    field = rest[len(dataset) + 1:] if rest.startswith(dataset + "-") else "reward"
                top.setdefault(dataset, {})[field] = value
            for dataset, fields in top.items():
                evals.setdefault(dataset, []).append((rollout_id, fields))
    for dataset in evals:
        evals[dataset].sort(key=lambda item: item[0])
    return train, evals


def _pick(series: dict[int, dict[str, Any]], keys: tuple[str, ...]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for rollout_id in sorted(series):
        row = series[rollout_id]
        for key in keys:
            value = _num(row.get(key))
            if value is not None:
                points.append((rollout_id, value))
                break
    return points


def _eval_pick(records: list[tuple[int, dict[str, Any]]], key: str) -> list[tuple[int, float]]:
    points = []
    for rollout_id, fields in records:
        value = _num(fields.get(key))
        if value is not None:
            points.append((rollout_id, value))
    return points


def collect_curves(run_dir: Path, log_file: Path | None = None) -> dict[str, Any]:
    """Merge structured JSONL with the train.log fallback into curve series."""
    log_path = log_file or run_dir / "logs" / "train.log"
    train, evals, actor = _load_jsonl(run_dir / "logs" / "metrics.jsonl")
    if not train or not evals:
        log_train, log_evals = _parse_train_log(log_path)
        for rollout_id, row in log_train.items():
            train.setdefault(rollout_id, row)
        for dataset, records in log_evals.items():
            existing = {rid for rid, _ in evals.get(dataset, [])}
            merged = evals.setdefault(dataset, [])
            merged.extend((rid, row) for rid, row in records if rid not in existing)
            merged.sort(key=lambda item: item[0])

    curves: dict[str, Any] = {
        "train": {
            "raw_reward": _pick(train, TRAIN_REWARD_KEYS),
            "truncated": _pick(train, TRUNC_KEYS),
            "response_lengths": _pick(train, ("response_lengths",)),
        },
        "eval": {
            dataset: {
                "reward": _eval_pick(records, "reward"),
                "truncated_ratio": _eval_pick(records, "truncated_ratio"),
            }
            for dataset, records in sorted(evals.items())
        },
        "logprob_abs_diff": [
            (row["rollout_id"], value)
            for row in sorted(actor, key=lambda r: r["rollout_id"])
            if (value := _num(row.get("train_rollout_logprob_abs_diff"))) is not None
        ],
    }
    return curves


def build_summary(curves: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "train": {key: _series_stats(points) for key, points in curves["train"].items()},
        "eval": {
            dataset: {key: _series_stats(points) for key, points in series.items()}
            for dataset, series in curves["eval"].items()
        },
        "logprob_abs_diff": _series_stats(curves["logprob_abs_diff"]),
    }
    return summary


def _has_points(curves: dict[str, Any]) -> bool:
    if any(curves["train"].values()) or curves["logprob_abs_diff"]:
        return True
    return any(any(series.values()) for series in curves["eval"].values())


def plot_curves(curves: dict[str, Any], out_dir: Path, run_name: str) -> Path:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0][0]
    for key, label in (("raw_reward", "train raw reward"),):
        points = curves["train"].get(key) or []
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], ".-", lw=1.6, label=label)
    ax.set_title("train reward (AIME2025 train set)")
    ax.set_xlabel("rollout")
    ax.set_ylabel("mean reward")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "no train reward data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[0][1]
    for dataset, series in curves["eval"].items():
        points = series.get("reward") or []
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], ".-", lw=1.6, label=dataset)
    ax.set_title("OOD eval reward")
    ax.set_xlabel("rollout")
    ax.set_ylabel("mean reward")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "no eval data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1][0]
    points = curves["train"].get("truncated") or []
    if points:
        ax.plot([x for x, _ in points], [y for _, y in points], ".-", lw=1.4, label="train")
    for dataset, series in curves["eval"].items():
        ep = series.get("truncated_ratio") or []
        if ep:
            ax.plot([x for x, _ in ep], [y for _, y in ep], ".-", lw=1.4, label=f"eval {dataset}")
    ax.set_title("truncated ratio")
    ax.set_xlabel("rollout")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=9)

    ax = axes[1][1]
    points = curves["logprob_abs_diff"]
    if points:
        ax.semilogy([x for x, _ in points], [y for _, y in points], ".-", lw=1.4)
        ax.axhline(0.1, color="red", ls="--", lw=1.0, alpha=0.6, label="acceptance 0.1")
        ax.legend(fontsize=9)
    ax.set_title("train/rollout logprob abs diff")
    ax.set_xlabel("rollout")
    ax.grid(alpha=0.3)
    if not points:
        ax.text(0.5, 0.5, "no mismatch metric", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(run_name, fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "math_rlvr_curves.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-figs", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    jsonl = run_dir / "logs" / "metrics.jsonl"
    log_file = args.log_file or run_dir / "logs" / "train.log"
    if not jsonl.is_file() and not log_file.is_file():
        print(f"[!] neither {jsonl} nor {log_file} exists", file=sys.stderr)
        return 1

    curves = collect_curves(run_dir, log_file)
    if not _has_points(curves):
        print("[!] no usable train/eval series found", file=sys.stderr)
        return 2

    out_dir = args.out_dir or run_dir / "metrics" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(curves)
    summary_path = out_dir / "math_rlvr_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[+] wrote {summary_path}")

    if not args.no_figs:
        fig_path = plot_curves(curves, out_dir, run_dir.name)
        print(f"[+] wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
