"""Rebuild strict, redacted SETA world-model records from raw trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .action_view import ACTION_VIEW_SCHEMA, render_tool_call_bundle
from .metadata import is_verified_execution_reward_contract, redact_sensitive_jsonable, stable_hash
from .seta_dataset import TerminalTransition, transitions_from_seta_trajectory


TRUNCATION_MARKER = "[openclaw_truncated_middle]"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    return {"path": str(path), "count": len(rows), "sha256": _sha256(path)}


def _record_from_transition(transition: TerminalTransition) -> dict[str, Any]:
    """Serialize once after a complete structured redaction pass.

    ``TerminalTransition.to_dict`` defensively redacts several derived fields
    repeatedly. Rebuild already holds redacted transitions, but this single
    canonical pass is retained so metadata and source paths get the same
    protection without quadratic work on long cumulative contexts.
    """

    value = redact_sensitive_jsonable(asdict(transition))
    canonical = TerminalTransition.from_dict(value)
    value["tool_names"] = list(canonical.tool_names)
    context_text = json.dumps(
        canonical.context_messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    next_context_text = None
    if canonical.next_context_messages:
        next_context_text = json.dumps(
            canonical.next_context_messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    context_hash = stable_hash(context_text)
    action_text = canonical.action_text
    feedback_text = canonical.feedback_text
    trajectory_id = canonical.trajectory_id
    transition_id = stable_hash(
        {
            "trajectory_id": trajectory_id,
            "turn_idx": canonical.turn_idx,
            "context_hash": context_hash,
            "action_text": action_text,
            "feedback_text": feedback_text,
        }
    )
    value.update(
        {
            "schema": "openclaw_terminal_transition_v3",
            "transition_id": transition_id,
            "trajectory_id": trajectory_id,
            "uid": trajectory_id,
            "has_next": bool(canonical.next_context_messages),
            "context_text": context_text,
            "context_hash": context_hash,
            "context_hash_schema": "canonical_redacted_text_v1",
            "action_hash": stable_hash(action_text),
            "next_observation_text": feedback_text,
            "next_observation_hash": stable_hash(feedback_text),
            "next_context_text": next_context_text,
            "next_context_hash": stable_hash(next_context_text) if next_context_text else None,
            "reward_score": canonical.reward,
            "redaction_applied": True,
        }
    )
    return value


def _validate_raw_trajectory(payload: dict[str, Any], path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if payload.get("trajectory_format") != "openclaw-terminal-rl-1":
        raise ValueError(f"unsupported or missing trajectory_format: {path}")
    info = payload.get("info")
    if not isinstance(info, dict):
        raise TypeError(f"trajectory info must be an object: {path}")
    if not str(info.get("uid") or "").strip():
        raise ValueError(f"trajectory uid is missing: {path}")
    if not str(info.get("task_id") or "").strip():
        raise ValueError(f"trajectory task_id is missing: {path}")
    raw_turns = payload.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise TypeError(f"trajectory turns must be a non-empty list: {path}")
    if not all(isinstance(turn, dict) for turn in raw_turns):
        raise TypeError(f"trajectory contains a non-object turn: {path}")
    turns = list(raw_turns)
    turn_indices = [turn.get("turn_idx") for turn in turns]
    if turn_indices != list(range(len(turns))):
        raise ValueError(f"turn_idx must be unique, contiguous, and ordered: {path}")
    for turn in turns:
        context = turn.get("context_messages")
        if not isinstance(context, list) or not all(isinstance(item, dict) for item in context):
            raise TypeError(f"context_messages must contain only objects: {path}")
        sdk_model_turns = turn.get("sdk_model_turns")
        if not isinstance(sdk_model_turns, list) or len(sdk_model_turns) != 1:
            raise ValueError(f"each SETA turn must map to exactly one sdk_model_turn: {path}")
        calls = turn.get("tool_calls")
        if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
            raise TypeError(f"tool_calls must contain only objects: {path}")
        call_ids = [str(call.get("tool_call_id") or "").strip() for call in calls]
        if any(not value for value in call_ids):
            raise ValueError(f"tool call ID is missing: {path}")
        if len(call_ids) != len(set(call_ids)):
            raise ValueError(f"tool call ID is duplicated within a turn: {path}")
        for call in calls:
            if not str(call.get("tool_name") or call.get("name") or "").strip():
                raise ValueError(f"tool name is missing: {path}")
            if call.get("result") is None:
                raise ValueError(f"tool call result is missing: {path}")
    return info, turns


def rebuild_dataset(
    source: Path,
    output_dir: Path,
    *,
    expected_file_count: int | None = None,
    expected_index_sha256: str | None = None,
    expected_tree_sha256: str | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"SETA source directory does not exist: {source}")
    paths = sorted(source.rglob("traj.json"))
    if not paths:
        raise ValueError(f"SETA source contains no traj.json files: {source}")

    index_path = source / "index.jsonl"
    index_sha256 = _sha256(index_path) if index_path.is_file() else None
    tree_sha256 = _tree_sha256(source, paths)
    if expected_file_count is not None and len(paths) != expected_file_count:
        raise ValueError(
            f"trajectory file count mismatch: expected={expected_file_count} actual={len(paths)}"
        )
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise ValueError(
            f"trajectory index digest mismatch: expected={expected_index_sha256} "
            f"actual={index_sha256}"
        )
    if expected_tree_sha256 is not None and tree_sha256 != expected_tree_sha256:
        raise ValueError(
            f"trajectory tree digest mismatch: expected={expected_tree_sha256} "
            f"actual={tree_sha256}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    all_tool_rows: list[TerminalTransition] = []
    all_tool_turns: list[dict[str, Any]] = []
    single_call_rows: list[TerminalTransition] = []
    multi_call_rows: list[TerminalTransition] = []
    statuses: Counter[str] = Counter()
    source_marker_count = 0
    trajectory_ids: set[str] = set()
    task_ids: set[str] = set()
    missing_task_id = 0
    max_tool_call_count = 0
    max_action_chars = 0
    max_feedback_chars = 0
    source_inventory: list[dict[str, Any]] = []
    seen_trajectory_uids: set[str] = set()

    # sys.maxsize makes the existing redaction adapter lossless with respect to
    # character length while preserving its strict schema and secret handling.
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"SETA trajectory must be a JSON object: {path}")
        info, turns = _validate_raw_trajectory(payload, path)
        uid = str(info["uid"])
        if uid in seen_trajectory_uids:
            raise ValueError(f"duplicate trajectory uid: {uid}")
        seen_trajectory_uids.add(uid)
        source_inventory.append(
            {
                "relative_path": path.relative_to(source).as_posix(),
                "sha256": _sha256(path),
                "trajectory_id_hash": stable_hash(uid),
                "task_id_hash": stable_hash(str(info["task_id"])),
                "turn_count": len(turns),
            }
        )
        source_marker_count += json.dumps(payload, ensure_ascii=False).count(TRUNCATION_MARKER)
        statuses[str(info.get("status"))] += 1
        transitions = transitions_from_seta_trajectory(
            payload,
            source_path=str(path),
            max_text_chars=sys.maxsize,
        )
        if len(transitions) != len(turns):
            raise ValueError(f"turn/transition count mismatch: {path}")
        for turn, transition in zip(turns, transitions, strict=True):
            trajectory_ids.add(transition.trajectory_id)
            if transition.task_id:
                task_ids.add(transition.task_id)
            else:
                missing_task_id += 1
            calls = [call for call in (turn.get("tool_calls") or []) if isinstance(call, dict)]
            result_count = sum(call.get("result") is not None for call in calls)
            max_tool_call_count = max(max_tool_call_count, len(calls))
            if not transition.has_tool_result:
                continue
            if result_count <= 0:
                raise ValueError(f"tool-feedback transition has no observed result: {path}")
            all_tool_rows.append(transition)
            all_tool_turns.append(turn)
            max_action_chars = max(max_action_chars, len(transition.action_text))
            max_feedback_chars = max(max_feedback_chars, len(transition.feedback_text))
            if len(calls) == 1 and result_count == 1:
                single_call_rows.append(transition)
            elif len(calls) > 1:
                # Multi-call turns remain observational bundles. Splitting them
                # would invent unavailable intermediate state boundaries.
                multi_call_rows.append(transition)

    if not all_tool_rows:
        raise ValueError("rebuild produced no tool-feedback transitions")
    if missing_task_id:
        raise ValueError(
            f"task_id is missing for {missing_task_id} transitions; held-out split is unsafe"
        )
    records = [_record_from_transition(row) for row in all_tool_rows]
    transition_ids = [str(record["transition_id"]) for record in records]
    if len(transition_ids) != len(set(transition_ids)):
        raise ValueError("duplicate transition_id detected in rebuilt dataset")
    single_objects = {id(row) for row in single_call_rows}
    multi_objects = {id(row) for row in multi_call_rows}
    single_records = [
        record
        for row, record in zip(all_tool_rows, records, strict=True)
        if id(row) in single_objects
    ]
    multi_records = [
        record
        for row, record in zip(all_tool_rows, records, strict=True)
        if id(row) in multi_objects
    ]
    tool_call_bundle_records: list[dict[str, Any]] = []
    canonical_call_count = 0
    canonical_action_chars = 0
    for source_record, turn in zip(records, all_tool_turns, strict=True):
        calls = [
            call
            for call in (turn.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("result") is not None
        ]
        action_text = render_tool_call_bundle(calls)
        record = dict(source_record)
        record["action_text"] = action_text
        record["action_view_schema"] = ACTION_VIEW_SCHEMA
        record["action_hash"] = stable_hash(action_text)
        record["transition_id"] = stable_hash(
            {
                "trajectory_id": record["trajectory_id"],
                "turn_idx": record["turn_idx"],
                "context_hash": record["context_hash"],
                "action_text": action_text,
                "feedback_text": record["feedback_text"],
            }
        )
        tool_call_bundle_records.append(record)
        canonical_call_count += len(calls)
        canonical_action_chars += len(action_text)
    canonical_transition_ids = [
        str(record["transition_id"]) for record in tool_call_bundle_records
    ]
    if len(canonical_transition_ids) != len(set(canonical_transition_ids)):
        raise ValueError("duplicate canonical tool-call transition_id detected")
    output_marker_count = sum(
        json.dumps(record, ensure_ascii=False).count(TRUNCATION_MARKER)
        for record in records
    )
    if output_marker_count > source_marker_count:
        raise ValueError(
            "the rebuild introduced truncation markers; refusing a lossy recovered dataset"
        )

    outputs = {
        "turn_bundle_records": _write_jsonl_atomic(
            output_dir / "turn_bundle_records.jsonl", records
        ),
        "single_call_records": _write_jsonl_atomic(
            output_dir / "single_call_records.jsonl", single_records
        ),
        "multi_call_bundle_records": _write_jsonl_atomic(
            output_dir / "multi_call_bundle_records.jsonl", multi_records
        ),
        "tool_call_bundle_records": _write_jsonl_atomic(
            output_dir / "tool_call_bundle_records.jsonl",
            tool_call_bundle_records,
        ),
    }
    manifest = {
        "schema_version": "openclaw_seta_rebuild_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "trajectory_file_count": len(paths),
            "index_path": str(index_path) if index_path.is_file() else None,
            "index_sha256": index_sha256,
            "tree_sha256": tree_sha256,
            "expected": {
                "trajectory_file_count": expected_file_count,
                "index_sha256": expected_index_sha256,
                "tree_sha256": expected_tree_sha256,
            },
            "inventory": source_inventory,
        },
        "adapter_contract": {
            "redaction_applied": True,
            "character_truncation": False,
            "max_text_chars": sys.maxsize,
            "multi_call_policy": "retain_turn_bundle_never_split",
            "single_call_subset_policy": (
                "exactly_one_tool_call_with_observed_result; still a turn record, "
                "not a reconstructed per-call state transition"
            ),
            "source_truncation_marker_count": source_marker_count,
            "output_truncation_marker_count": output_marker_count,
            "adapter_source_sha256": {
                Path(__file__).name: _sha256(Path(__file__)),
                "seta_dataset.py": _sha256(Path(__file__).with_name("seta_dataset.py")),
            },
            "loader_contract": (
                "Pass one exact JSONL output to the trainer. The output directory contains "
                "overlapping diagnostic views and must not be loaded recursively."
            ),
            "tool_call_bundle": {
                "schema": ACTION_VIEW_SCHEMA,
                "unit": "observational_turn_bundle",
                "assistant_reasoning_included": False,
                "ordering": "logged tool-call order is preserved",
                "arguments": "canonical redacted JSON",
                "atomic_state_boundary_inferred": False,
            },
        },
        "audit": {
            "trajectory_id_count": len(trajectory_ids),
            "task_id_count": len(task_ids),
            "tool_feedback_transition_count": len(all_tool_rows),
            "single_call_transition_count": len(single_call_rows),
            "multi_call_bundle_transition_count": len(multi_call_rows),
            "unclassified_tool_feedback_count": (
                len(all_tool_rows) - len(single_call_rows) - len(multi_call_rows)
            ),
            "max_tool_call_count": max_tool_call_count,
            "max_action_chars": max_action_chars,
            "max_feedback_chars": max_feedback_chars,
            "verified_execution_reward_label_count": sum(
                is_verified_execution_reward_contract(record) for record in records
            ),
            "canonical_tool_call_count": canonical_call_count,
            "canonical_tool_call_record_count": len(tool_call_bundle_records),
            "canonical_action_chars": canonical_action_chars,
            "status_counts": dict(sorted(statuses.items())),
        },
        "outputs": outputs,
        "claim_boundary": (
            "Turn bundles are observational logged transitions. Multi-call turns are not "
            "atomic actions and do not support counterfactual execution claims."
        ),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-file-count", type=int)
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--expected-tree-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = rebuild_dataset(
        args.source,
        args.output_dir,
        expected_file_count=args.expected_file_count,
        expected_index_sha256=args.expected_index_sha256,
        expected_tree_sha256=args.expected_tree_sha256,
    )
    print(json.dumps(manifest["audit"], ensure_ascii=False, sort_keys=True))
    print(f"wrote rebuilt SETA records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
