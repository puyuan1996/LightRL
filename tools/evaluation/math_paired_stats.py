#!/usr/bin/env python3
"""Compatibility entry point for paired Math RLVR statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.math_rlvr.stats import compare_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate", nargs="+")
    parser.add_argument("--metric", default="lenient_correct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    results = [compare_files(args.baseline, candidate, key=args.metric, seed=args.seed) for candidate in args.candidate]
    text = json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
