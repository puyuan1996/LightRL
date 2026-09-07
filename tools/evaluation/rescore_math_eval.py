#!/usr/bin/env python3
"""Compatibility entry point for post-hoc Math RLVR rescoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.math_rlvr.scorer import rescore_directory, rescore_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail", nargs="*", help="detail JSON files or a results directory")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reward-type", choices=("math", "dapo", "boxed"), default=None)
    parser.add_argument("--response-cap", type=int, default=None)
    args = parser.parse_args(argv)
    sources = args.detail or ([args.results_dir] if args.results_dir else [])
    if not sources:
        parser.error("provide detail files or --results-dir")
    for source in sources:
        if Path(source).is_dir():
            for path in rescore_directory(source, reward_type=args.reward_type, response_cap=args.response_cap):
                print(path)
            continue
        output = None
        if args.output_dir:
            output = Path(args.output_dir) / (Path(source).stem + ".rescore.json")
        print(rescore_file(source, output=output, reward_type=args.reward_type, response_cap=args.response_cap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
