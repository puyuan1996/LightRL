#!/usr/bin/env python3
"""Compare fixed12 pass@1 across one or more (optionally resumed) runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt


INVALID_STATUSES = {"failed", "aborted"}


@dataclass(frozen=True)
class EvalPoint:
    run: str
    step: int
    pass_at_1: float
    valid: bool
    task_count: int
    invalid_task_count: int
    source: str


def parse_run(value: str) -> tuple[str, list[Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR[,RUN_DIR...]")
    label, raw_paths = value.split("=", 1)
    paths = [Path(path).resolve() for path in raw_paths.split(",") if path]
    if not label or not paths:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR[,RUN_DIR...]")
    return label, paths


def load_point(label: str, step_dir: Path) -> EvalPoint:
    tasks_path = step_dir / "tasks.jsonl"
    summary_path = step_dir / "summary.json"
    rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    invalid = sum(
        any(str(status).lower() in INVALID_STATUSES for status in row.get("statuses", []))
        for row in rows
    )
    task_count = len(rows)
    step = int(summary["global_step"])
    accuracy = float(summary.get("pass_at_1", summary["eval/pass_at_k"]))
    return EvalPoint(
        run=label,
        step=step,
        pass_at_1=accuracy,
        valid=task_count == 12 and invalid == 0,
        task_count=task_count,
        invalid_task_count=invalid,
        source=str(step_dir),
    )


def collect(label: str, segments: list[Path]) -> list[EvalPoint]:
    by_step: dict[int, EvalPoint] = {}
    for run_dir in segments:
        eval_root = run_dir / "evaluations" / "seta"
        if not eval_root.is_dir():
            continue
        for step_dir in sorted(eval_root.glob("step_*")):
            if (step_dir / "tasks.jsonl").is_file() and (step_dir / "summary.json").is_file():
                point = load_point(label, step_dir)
                # A valid continuation artifact replaces an invalid earlier
                # attempt at the same step; otherwise the later segment wins.
                previous = by_step.get(point.step)
                if previous is None or point.valid or not previous.valid:
                    by_step[point.step] = point
    return [by_step[step] for step in sorted(by_step)]


def write_outputs(points: list[EvalPoint], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(point) for point in points]
    (output_dir / "fixed12_eval_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "fixed12_eval_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]) if payload else list(EvalPoint.__annotations__))
        writer.writeheader()
        writer.writerows(payload)

    labels = list(dict.fromkeys(point.run for point in points))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label in labels:
        selected = [point for point in points if point.run == label]
        valid = [point for point in selected if point.valid]
        invalid = [point for point in selected if not point.valid]
        if valid:
            ax.plot([p.step for p in valid], [p.pass_at_1 for p in valid], marker="o", linewidth=2, label=label)
        if invalid:
            ax.scatter(
                [p.step for p in invalid],
                [p.pass_at_1 for p in invalid],
                marker="x",
                s=85,
                linewidths=2,
                label=f"{label} (infra-invalid)",
            )
    ax.set(title="SETA fixed12 evaluation", xlabel="eval rollout-step", ylabel="pass@1")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fixed12_eval_comparison.png", dpi=180)
    plt.close(fig)

    steps = sorted({point.step for point in points})
    lookup = {(point.run, point.step): point for point in points}
    lines = [
        "# DAPO / DIVE-PO fixed12 eval-step 对比",
        "",
        "`×`/`无效` 表示 12 个任务中存在基础设施 `failed/aborted`，该 0 分不得解释为模型准确率。",
        "",
        "| step | " + " | ".join(labels) + " |",
        "|---:|" + "|".join("---:" for _ in labels) + "|",
    ]
    for step in steps:
        cells = []
        for label in labels:
            point = lookup.get((label, step))
            if point is None:
                cells.append("—")
            elif point.valid:
                cells.append(f"{point.pass_at_1:.4f}")
            else:
                cells.append(f"无效（infra {point.invalid_task_count}/{point.task_count}）")
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    (output_dir / "fixed12_eval_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="LABEL=DIR[,DIR...]")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    points = [point for label, segments in args.run for point in collect(label, segments)]
    write_outputs(points, args.output_dir.resolve())


if __name__ == "__main__":
    main()
