#!/usr/bin/env python3
"""Stitch ordered run segments into one metrics.jsonl without duplicate steps."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any


def record_key(record: dict[str, Any], fallback: tuple[int, int]) -> tuple[Any, ...]:
    schema = record.get("schema")
    phase = record.get("phase")
    if schema == "terminal_rl.per_dataset_metrics.v1":
        return (schema, phase, record.get("dataset"), record.get("rollout_id"), record.get("global_step"))
    if schema == "terminal_rl.actor_update_metrics.v1":
        return (schema, phase, record.get("role"), record.get("rollout_id"), record.get("rollout_step_id"))
    return ("unkeyed", *fallback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    segments = [path.resolve() for path in args.segment]
    output = args.output_dir.resolve()
    if output in segments:
        raise SystemExit("output directory must differ from every source segment")

    records: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    counts: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        metrics = segment / "logs" / "metrics.jsonl"
        read_count = 0
        replacement_count = 0
        with metrics.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record_key(record, (segment_index, line_index))
                replacement_count += int(key in records)
                records[key] = record
                read_count += 1
        counts.append(
            {
                "segment": str(segment),
                "records_read": read_count,
                "records_replaced": replacement_count,
            }
        )

    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "config").mkdir(parents=True, exist_ok=True)
    with (output / "logs" / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for record in records.values():
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
    # plot_training_metrics accepts structured-only runs; keep an empty legacy
    # log so its normal run-directory interface remains usable.
    (output / "logs" / "train.log").touch()
    for filename in ("run_config.json", "rollout_config.yaml"):
        source = next(
            (segment / "config" / filename for segment in reversed(segments) if (segment / "config" / filename).is_file()),
            None,
        )
        if source:
            shutil.copy2(source, output / "config" / filename)
    manifest = {
        "schema": "lightrl.stitched_metrics.v1",
        "segments": counts,
        "records_written": len(records),
        "deduplication": "later segment replaces identical schema/phase/dataset/rollout key",
    }
    (output / "stitch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output / "logs" / "metrics.jsonl")


if __name__ == "__main__":
    main()
