"""Compare streaming LWM A/B runs (noreplay vs replay) across seeds.

Reads one or more ``stream_latent`` output directories (each containing
``metrics.jsonl`` with per-arm, per-chunk rows) and reports:

- per-seed final held-out pred loss per arm and the paired delta;
- steps-to-threshold: cumulative fresh transitions at which the replay arm
  first reaches the noreplay arm's final held-out pred loss (the
  sample-efficiency statement, in the spirit of ECHO's "fewer steps" claim);
- guardrails: NaN check, minimum effective rank, minimum action delta.

Usage:
    python examples/training/world_model/compare_stream_runs.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no metrics.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def _arm_curves(rows: list[dict]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(row["arm"], []).append(row)
    for arm_rows in arms.values():
        arm_rows.sort(key=lambda row: row["chunk"])
    return arms


def _val_pred_loss(row: dict) -> float | None:
    metrics = row.get("val_metrics") or {}
    value = metrics.get("wm/pred_loss")
    return float(value) if value is not None else None


def _first_reach(curve: list[dict], threshold: float) -> int | None:
    for row in curve:
        value = _val_pred_loss(row)
        if value is not None and value <= threshold:
            return int(row["cumulative_fresh"])
    return None


def analyze_run(run_dir: Path) -> dict:
    """Summarize one ``stream_latent`` run directory.

    Rows are grouped by arm and sorted by chunk.  Per arm, the report carries
    the final held-out pred loss, the ``(cumulative_fresh, val_pred_loss)``
    curve, min effective rank / action delta, NaN row count, total optimizer
    steps, wall time, and replay stats.  When both arms are present, a
    ``comparison`` block adds the paired final-loss delta, the cumulative-fresh
    count at which the replay arm first reaches the noreplay arm's final loss
    (steps-to-threshold), and the resulting sample-saving ratio.

    Args:
        run_dir: Run directory containing ``metrics.jsonl``.

    Returns:
        Report dict with ``run_dir``, per-arm stats under ``arms``, and an
        optional ``comparison`` block.

    Raises:
        FileNotFoundError: If ``run_dir`` has no ``metrics.jsonl``.
    """

    rows = _load_rows(run_dir)
    arms = _arm_curves(rows)
    result: dict = {"run_dir": str(run_dir), "arms": {}}
    for arm, curve in arms.items():
        losses = [_val_pred_loss(row) for row in curve]
        ranks = [
            float(row["val_metrics"].get("wm/effective_rank", float("nan")))
            for row in curve
            if row.get("val_metrics")
        ]
        deltas = [
            float(row["val_metrics"].get("wm/action_delta", float("nan")))
            for row in curve
            if row.get("val_metrics")
        ]
        nan_rows = sum(
            1
            for row in curve
            for value in [row.get("train_loss"), row.get("val_loss")]
            if value is not None and not math.isfinite(float(value))
        )
        result["arms"][arm] = {
            "chunks": len(curve),
            "final_val_pred_loss": losses[-1],
            "curve": [
                {"cumulative_fresh": row["cumulative_fresh"], "val_pred_loss": value}
                for row, value in zip(curve, losses)
            ],
            "min_val_effective_rank": min(ranks) if ranks else None,
            "min_val_action_delta": min(deltas) if deltas else None,
            "nan_rows": nan_rows,
            "total_steps": curve[-1]["cumulative_steps"] if curve else 0,
            "wall_seconds": curve[-1].get("seconds_since_arm_start") if curve else None,
            "replay_stats": curve[-1].get("replay_stats") if curve else None,
        }
    if "noreplay" in arms and "replay" in arms:
        base_final = result["arms"]["noreplay"]["final_val_pred_loss"]
        replay_curve = arms["replay"]
        threshold = base_final if base_final is not None else float("inf")
        reach = _first_reach(replay_curve, threshold)
        total_fresh = result["arms"]["noreplay"]["curve"][-1]["cumulative_fresh"]
        result["comparison"] = {
            "noreplay_final_val_pred_loss": base_final,
            "replay_final_val_pred_loss": result["arms"]["replay"]["final_val_pred_loss"],
            "paired_delta": (
                None
                if base_final is None or result["arms"]["replay"]["final_val_pred_loss"] is None
                else result["arms"]["replay"]["final_val_pred_loss"] - base_final
            ),
            "replay_fresh_to_reach_noreplay_final": reach,
            "noreplay_total_fresh": total_fresh,
            "sample_saving_ratio": (None if reach in (None, 0) else 1.0 - reach / total_fresh),
        }
    return result


def main() -> None:
    """Aggregate per-seed run reports and print the A/B comparison table.

    Each positional directory is one seed's ``stream_latent`` output.  The
    printed table shows per-arm final held-out pred loss, min effective rank,
    steps, and wall time, followed by the paired comparison row (delta,
    steps-to-threshold, sample saving); NaN rows trigger a stderr guardrail
    warning, and ``--json`` additionally dumps the full machine-readable report.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="Optional path for the full JSON report.")
    args = parser.parse_args()

    reports = [analyze_run(run_dir) for run_dir in args.run_dirs]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    header = f"{'run':<64} {'arm':<9} {'final val pred':>14} {'min erank':>9} {'steps':>6} {'wall(s)':>8}"
    print(header)
    print("-" * len(header))
    for report in reports:
        name = Path(report["run_dir"]).name[:62]
        for arm, stats in sorted(report["arms"].items()):
            final = stats["final_val_pred_loss"]
            erank = stats["min_val_effective_rank"]
            wall = stats["wall_seconds"]
            print(
                f"{name:<64} {arm:<9} "
                f"{('%.6f' % final) if final is not None else '-':>14} "
                f"{('%.2f' % erank) if erank is not None else '-':>9} "
                f"{stats['total_steps']:>6} "
                f"{('%.0f' % wall) if wall is not None else '-':>8}"
            )
        comparison = report.get("comparison")
        if comparison:
            reach = comparison["replay_fresh_to_reach_noreplay_final"]
            saving = comparison["sample_saving_ratio"]
            delta = comparison["paired_delta"]
            print(
                f"{'  -> comparison':<64} {'':<9} "
                f"delta={('%.6f' % delta) if delta is not None else '-'} "
                f"reach@{reach} saving={('%.1f%%' % (100 * saving)) if saving is not None else 'n/a'}"
            )
        if any((stats.get("nan_rows") or 0) > 0 for stats in report["arms"].values()):
            print("  !! NaN rows detected", file=sys.stderr)


if __name__ == "__main__":
    main()
