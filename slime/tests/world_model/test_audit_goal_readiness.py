from __future__ import annotations

import json

from slime.world_model.audit_goal_readiness import T2_REQUIRED_FIELDS, audit


def test_turn_bundle_is_diagnostic_but_not_atomic_ready(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "transition_id": "t1",
                "task_id": "task",
                "tool_names": ["shell"],
                "has_next": True,
                "next_observation_text": "ok",
            }
        )
        + "\n"
    )
    result = audit(path)
    assert result["gates"]["turn_bundle_diagnostic_ready"] is True
    assert result["gates"]["strict_t2_ready"] is False
    assert result["gates"]["strict_t3_ready"] is False
    assert result["label_counts"]["complete_atomic_t2_rows"] == 0


def test_complete_atomic_row_passes_t2_not_t3(tmp_path) -> None:
    row = {field: f"value-{field}" for field in T2_REQUIRED_FIELDS}
    row.update(
        {
            "transition_id": "t1",
            "tool_names": ["shell"],
            "has_next": True,
            "next_observation_text": "ok",
            "canonical_args": {"cmd": "pwd"},
            "exit_code": 0,
            "tool_call_idx": 0,
        }
    )
    path = tmp_path / "atomic.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = audit(path)
    assert result["gates"]["strict_t2_ready"] is True
    assert result["gates"]["strict_t3_ready"] is False
