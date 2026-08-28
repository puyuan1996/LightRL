#!/usr/bin/env python3
"""Evaluation CLI: run / batch / report / smoke.

Usage examples (paths are placeholders):

    python3 tools/evaluation/eval_cli.py run --config tools/evaluation/configs/tb21_terminus2.example.yaml --dry-run
    python3 tools/evaluation/eval_cli.py smoke --config <cfg> --task <task-name>
    python3 tools/evaluation/eval_cli.py batch --config tools/evaluation/configs/batch.example.yaml
    python3 tools/evaluation/eval_cli.py report --results '<runs>/*/eval_result.json' --output <out>/compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parents[1]
for _path in (str(_REPO_ROOT), str(_TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.config import build_specs, load_config  # noqa: E402


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key (dot path, YAML scalar value); repeatable",
    )
    parser.add_argument("--dry-run", action="store_true", help="print config and command without executing")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="seconds between progress polls")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="evaluate one checkpoint with one harness")
    _add_config_args(run_p)

    smoke_p = sub.add_parser("smoke", help="single-task shortcut of run (concurrency=1, retries=0)")
    _add_config_args(smoke_p)
    smoke_p.add_argument("--task", required=True, help="single task name to run")

    batch_p = sub.add_parser("batch", help="serially evaluate multiple checkpoints")
    _add_config_args(batch_p)

    report_p = sub.add_parser("report", help="compare multiple eval_result.json files")
    report_p.add_argument("--results", nargs="+", required=True, help="paths or glob patterns")
    report_p.add_argument("--output", required=True, help="output prefix for compare.md/.csv")

    args = parser.parse_args()

    if args.command == "report":
        from core.report import load_results, render_markdown, write_report

        results = load_results(args.results)
        print(render_markdown(results))
        md_path, csv_path = write_report(results, args.output)
        print(f"written: {md_path} {csv_path}")
        return 0

    config = load_config(args.config, args.overrides)

    if args.command == "batch":
        from core.batch import run_batch

        _, failures = run_batch(config, dry_run=args.dry_run, poll_interval_s=args.poll_interval)
        return 1 if failures else 0

    if args.command == "smoke":
        config.setdefault("dataset", {})["task_names"] = [args.task]
        config.setdefault("run", {})["concurrency"] = 1
        config["run"]["max_retries"] = 0

    from core.runner import run_eval

    spec, serving = build_specs(config)
    run_eval(spec, serving, dry_run=args.dry_run, poll_interval_s=args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
