"""Build an audited result-only target view from strict SETA turn bundles."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metadata import redact_sensitive_text, stable_hash
from .result_view import RESULT_VIEW_SCHEMA, parse_tool_result_bundle, render_result_only_view


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def build_result_only_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input records do not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=False)

    output_rows: list[dict[str, Any]] = []
    transition_map: list[dict[str, Any]] = []
    source_transition_ids: set[str] = set()
    output_transition_ids: set[str] = set()
    result_count_histogram: Counter[int] = Counter()
    tool_result_count_mismatch = 0
    result_body_chars = 0
    redaction_idempotence_mismatch_count = 0
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"record line {line_number} is not an object")
            if not value.get("has_tool_result"):
                raise ValueError(f"record line {line_number} lacks observed tool feedback")
            if value.get("schema") != "openclaw_terminal_transition_v3":
                raise ValueError(f"record line {line_number} is not a strict v3 transition")
            if value.get("redaction_applied") is not True:
                raise ValueError(f"record line {line_number} lacks redaction provenance")
            source_id = str(value.get("transition_id") or "")
            if not source_id or source_id in source_transition_ids:
                raise ValueError(f"missing or duplicate source transition_id at line {line_number}")
            source_transition_ids.add(source_id)

            feedback_text = value.get("next_observation_text")
            if not isinstance(feedback_text, str):
                raise ValueError(f"record line {line_number} lacks feedback text")
            results = parse_tool_result_bundle(feedback_text)
            result_count_histogram[len(results)] += 1
            result_body_chars += sum(len(result) for result in results)
            tool_names = value.get("tool_names")
            if not isinstance(tool_names, list):
                raise ValueError(f"record line {line_number} lacks tool-name audit labels")
            if len(tool_names) != len(results):
                tool_result_count_mismatch += 1
            target = render_result_only_view(results)
            if redact_sensitive_text(target) != target:
                redaction_idempotence_mismatch_count += 1
            row = dict(value)
            row.update(
                {
                    "feedback_text": target,
                    "feedback_source": RESULT_VIEW_SCHEMA,
                    "next_observation_text": target,
                    "next_observation_hash": stable_hash(target),
                }
            )
            row["transition_id"] = stable_hash(
                {
                    "trajectory_id": str(
                        row.get("trajectory_id") or row.get("uid") or ""
                    ),
                    "turn_idx": int(row.get("turn_idx", 0) or 0),
                    "context_hash": str(row.get("context_hash") or ""),
                    "action_text": str(row.get("action_text") or ""),
                    "feedback_text": target,
                }
            )
            output_id = str(row["transition_id"])
            if output_id in output_transition_ids:
                raise ValueError(f"duplicate converted transition_id at line {line_number}")
            output_transition_ids.add(output_id)
            output_rows.append(row)
            transition_map.append(
                {
                    "source_transition_id": source_id,
                    "result_only_transition_id": output_id,
                    "source_next_observation_hash": value.get("next_observation_hash"),
                    "result_only_next_observation_hash": row["next_observation_hash"],
                    "result_block_count": len(results),
                }
            )

    if not output_rows:
        raise ValueError("input records are empty")
    if expected_count is not None and len(output_rows) != expected_count:
        raise ValueError(
            f"record count mismatch: expected={expected_count} actual={len(output_rows)}"
        )
    if redaction_idempotence_mismatch_count:
        raise ValueError(
            "result-only target is not redaction-idempotent for "
            f"{redaction_idempotence_mismatch_count} records"
        )

    output_path = output_dir / "result_only_records.jsonl"
    _write_jsonl_atomic(output_path, output_rows)
    transition_map_path = output_dir / "transition_map.jsonl"
    _write_jsonl_atomic(transition_map_path, transition_map)
    wrapper_leakage_count = sum(
        "<tool_result" in str(row["next_observation_text"])
        or "</tool_result>" in str(row["next_observation_text"])
        for row in output_rows
    )
    if wrapper_leakage_count:
        raise ValueError("result-only output retained tool_result wrappers")
    manifest = {
        "schema_version": "openclaw_result_only_dataset_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_view": RESULT_VIEW_SCHEMA,
        "source": {
            "path": str(input_path),
            "count": len(output_rows),
            "sha256": _sha256(input_path),
        },
        "output": {
            "path": str(output_path),
            "count": len(output_rows),
            "sha256": _sha256(output_path),
        },
        "transition_map": {
            "path": str(transition_map_path),
            "count": len(transition_map),
            "sha256": _sha256(transition_map_path),
        },
        "audit": {
            "result_block_count": sum(
                count * frequency for count, frequency in result_count_histogram.items()
            ),
            "result_count_histogram": {
                str(key): value for key, value in sorted(result_count_histogram.items())
            },
            "single_result_record_count": result_count_histogram[1],
            "multi_result_record_count": (
                len(output_rows) - result_count_histogram[1]
            ),
            "tool_name_result_count_mismatch_records": tool_result_count_mismatch,
            "result_body_chars": result_body_chars,
            "wrapper_leakage_count": wrapper_leakage_count,
            "redaction_idempotence_mismatch_count": (
                redaction_idempotence_mismatch_count
            ),
            "source_transition_id_count": len(source_transition_ids),
            "output_transition_id_count": len(output_transition_ids),
        },
        "contract": {
            "ordering": "logged result order is preserved",
            "target_encoding": (
                "ordered generic <result index=N> blocks with original newlines"
            ),
            "tool_names_in_target": False,
            "tool_names_preserved_as_diagnostic_labels": True,
            "multi_call_policy": "retain observational bundle; never infer atomic boundaries",
        },
        "claim_boundary": (
            "This target supports observational turn/action-bundle result prediction. "
            "It does not identify atomic tool-call effects, counterfactual outcomes, "
            "verified execution success, or online policy improvement."
        ),
    }
    manifest_path = output_dir / "result_only_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_result_only_dataset(
        args.input,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(manifest["audit"], ensure_ascii=False, sort_keys=True))
    print(f"wrote result-only records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
