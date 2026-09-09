#!/usr/bin/env python3
"""Compatibility entry point for paired Math RLVR statistics."""

from __future__ import annotations

import argparse
import json
import sys
import glob
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.math_rlvr.stats import compare_files  # noqa: E402

BASE = "T1.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?")
    parser.add_argument("candidate", nargs="*")
    parser.add_argument("--results-dir", default=None, help="discover baseline/candidates by tag")
    parser.add_argument("--baseline-tag", default=BASE)
    parser.add_argument("--candidate-tag", default=None)
    parser.add_argument("--metric", default="lenient_correct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    baseline = args.baseline
    candidates = list(args.candidate)
    if args.results_dir:
        files = sorted(glob.glob(str(Path(args.results_dir) / "*.detail.json")))
        if not files:
            parser.error(f"no detail files in {args.results_dir}")
        base_candidates = [path for path in files if f"_{args.baseline_tag}_" in Path(path).name]
        if not base_candidates:
            parser.error(f"no baseline tagged {args.baseline_tag!r} in {args.results_dir}")
        baseline = baseline or base_candidates[0]
        candidates = candidates or [
            path for path in files if path != baseline and (args.candidate_tag is None or f"_{args.candidate_tag}_" in Path(path).name)
        ]
    if not baseline or not candidates:
        parser.error("provide baseline and at least one candidate, or --results-dir")
    results = [compare_files(baseline, candidate, key=args.metric, seed=args.seed) for candidate in candidates]
    text = json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
