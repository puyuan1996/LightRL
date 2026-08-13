import json

from slime.world_model.audit_state_view import audit


def test_state_view_audit_reports_structural_gate(tmp_path):
    row = {
        "schema": "openclaw_terminal_transition_v3",
        "uid": "trajectory-a",
        "trajectory_id": "trajectory-a",
        "task_id": "task-a",
        "turn_idx": 0,
        "context_messages": [{"role": "user", "content": "inspect"}],
        "action_text": "pwd",
        "feedback_text": "/tmp",
        "next_observation_text": "/tmp",
        "next_context_messages": [
            {"role": "user", "content": "inspect"},
            {"role": "tool", "content": "/tmp"},
        ],
        "has_tool_result": True,
    }
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = audit(path, max_events=3)

    assert result["record_count"] == 1
    assert result["gate"]["passed"] is True
    assert result["unchanged_current_next_rate"] == 0.0
    assert result["next_duplicate_rate"] == 0.0
