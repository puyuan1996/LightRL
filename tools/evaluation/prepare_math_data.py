#!/usr/bin/env python3
"""Prepare one normalized math JSONL dataset and a reproducibility manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluation.math_rlvr.data import load_dataset, write_jsonl, write_manifest  # noqa: E402
from tools.evaluation.math_rlvr.paths import resolve_data_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="alias/path or hf://org/name (defaults to --dataset)")
    parser.add_argument("--dataset", default=None, help="output dataset name (defaults to source stem)")
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", default=None)
    parser.add_argument("--output", default=None, help="explicit output JSONL path")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None, help="HuggingFace dataset config")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source or args.dataset
    if not source:
        raise SystemExit("one of --source or --dataset is required")
    rows = load_dataset(
        source,
        data_root=args.data_root,
        split=args.split,
        config=args.config,
        deduplicate=args.deduplicate,
        limit=args.limit,
        seed=args.seed,
    )
    name = args.dataset or Path(source.rstrip("/")).stem
    output_dir = Path(args.output_dir) if args.output_dir else resolve_data_root(args.data_root)
    output = Path(args.output) if args.output else output_dir / f"{name}.jsonl"
    write_jsonl(rows, output)
    manifest = write_manifest(rows, output.with_suffix(".manifest.json"), dataset=source, deduplicated=args.deduplicate)
    print(f"prepared {len(rows)} unique={manifest['unique_questions']} -> {output}")
    print(f"manifest_sha256={manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
