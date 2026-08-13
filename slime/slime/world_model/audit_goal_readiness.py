"""Audit whether an offline dataset can support diagnostic, T2, or T3 claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


T2_REQUIRED_FIELDS = (
    "env_snapshot_digest",
    "snapshot_restore_probe_digest",
    "task_id",
    "task_cluster_id",
    "tool_call_idx",
    "tool_name",
    "canonical_args",
    "result_text",
    "exit_code",
    "status",
    "error_type",
    "collector_version",
)
T3_REQUIRED_FIELDS = T2_REQUIRED_FIELDS + (
    "candidate_set_id",
    "candidate_index",
    "branch_id",
    "terminal_verifier_pass",
    "unsafe",
    "docker_execution_count",
)


def _present(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    return value is not None and value != "" and value != [] and value != {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError("dataset is empty")

    field_counts = Counter(
        field
        for row in rows
        for field in set(T3_REQUIRED_FIELDS)
        if _present(row, field)
    )
    tool_counts = [len(row.get("tool_names") or ()) for row in rows]
    t2_ready_rows = [
        row for row in rows if all(_present(row, field) for field in T2_REQUIRED_FIELDS)
    ]
    t3_ready_rows = [
        row for row in rows if all(_present(row, field) for field in T3_REQUIRED_FIELDS)
    ]
    verified_execution_rewards = sum(
        row.get("reward_label_is_execution_outcome") is True for row in rows
    )
    task_cluster_count = sum(_present(row, "task_cluster_id") for row in rows)
    has_next_count = sum(row.get("has_next") is True for row in rows)
    result_count = sum(
        bool(str(row.get("next_observation_text") or row.get("result_text") or "").strip())
        for row in rows
    )
    diagnostic_ready = bool(has_next_count and result_count)
    t2_ready = bool(t2_ready_rows) and task_cluster_count == len(rows)
    candidate_sets = {
        str(row["candidate_set_id"])
        for row in t3_ready_rows
        if _present(row, "candidate_set_id")
    }
    t3_ready = t2_ready and bool(t3_ready_rows) and bool(candidate_sets)

    blockers: dict[str, list[str]] = {"t2": [], "t3": []}
    if task_cluster_count != len(rows):
        blockers["t2"].append("task_cluster_id is incomplete")
    missing_t2 = [field for field in T2_REQUIRED_FIELDS if field_counts[field] == 0]
    if missing_t2:
        blockers["t2"].append("missing atomic fields: " + ", ".join(missing_t2))
    if not t2_ready_rows:
        blockers["t2"].append("no row satisfies the complete atomic T2 contract")
    missing_t3 = [field for field in T3_REQUIRED_FIELDS if field_counts[field] == 0]
    if missing_t3:
        blockers["t3"].append("missing branched fields: " + ", ".join(missing_t3))
    if not candidate_sets:
        blockers["t3"].append("no resolved candidate_set_id is available")

    return {
        "schema_version": "openclaw_world_model_goal_readiness_v1",
        "input": str(path),
        "input_sha256": _sha256(path),
        "record_count": len(rows),
        "turn_structure": {
            "zero_tool_rows": sum(count == 0 for count in tool_counts),
            "single_call_turn_rows": sum(count == 1 for count in tool_counts),
            "multi_call_turn_rows": sum(count > 1 for count in tool_counts),
        },
        "label_counts": {
            "has_next": has_next_count,
            "observed_result_text": result_count,
            "verified_execution_reward": verified_execution_rewards,
            "task_cluster_id": task_cluster_count,
            "complete_atomic_t2_rows": len(t2_ready_rows),
            "complete_branched_t3_rows": len(t3_ready_rows),
            "candidate_sets": len(candidate_sets),
        },
        "field_coverage": {
            field: {"count": field_counts[field], "rate": field_counts[field] / len(rows)}
            for field in T3_REQUIRED_FIELDS
        },
        "gates": {
            "turn_bundle_diagnostic_ready": diagnostic_ready,
            "strict_t2_ready": t2_ready,
            "strict_t3_ready": t3_ready,
        },
        "blockers": blockers,
        "claim_boundary": (
            "A turn-bundle diagnostic gate does not authorize atomic execution-result "
            "or same-snapshot counterfactual claims."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require", choices=["diagnostic", "t2", "t3"], default=None)
    args = parser.parse_args(argv)
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    gate = {
        "diagnostic": "turn_bundle_diagnostic_ready",
        "t2": "strict_t2_ready",
        "t3": "strict_t3_ready",
    }.get(args.require)
    if gate is not None and not result["gates"][gate]:
        raise SystemExit(2)
    print(f"wrote goal-readiness audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
