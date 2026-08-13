import json

import pytest

from slime.world_model.seta_dataset import load_terminal_transitions, transitions_from_seta_trajectory


def _trajectory():
    return {
        "info": {
            "uid": "trajectory-1",
            "task_id": "task-42",
            "task_cluster_id": "cluster-7",
            "task_name": "42",
            "data_source": "terminal_bench",
            "status": "Status.COMPLETED",
            "rollout_id": 3,
            "train_step": 6,
        },
        "reward": {"score": 1.0},
        "turns": [
            {
                "turn_idx": 0,
                "context_messages": [{"role": "user", "content": "inspect"}],
                "assistant_output": "run pwd",
                "tool_calls": [{"tool_name": "bash", "args": {"command": "pwd"}, "result": "/tmp"}],
            },
            {
                "turn_idx": 1,
                "context_messages": [
                    {"role": "user", "content": "inspect"},
                    {"role": "tool", "content": "/tmp"},
                ],
                "assistant_output": "done",
                "tool_calls": [],
            },
        ],
    }


def test_seta_transition_boundaries_include_next_context():
    transitions = transitions_from_seta_trajectory(_trajectory(), source_path="/tmp/traj.json")

    assert len(transitions) == 2
    assert transitions[0].action_text.startswith("run pwd")
    assert transitions[0].task_id == "task-42"
    assert transitions[0].task_cluster_id == "cluster-7"
    assert "/tmp" in transitions[0].feedback_text
    assert transitions[0].has_next is True
    assert transitions[0].next_context_messages[-1]["role"] == "tool"
    assert transitions[0].reward is None
    assert transitions[0].reward_label_scope == "missing_nonterminal"
    assert transitions[1].done is True
    assert transitions[1].has_next is False
    assert transitions[1].reward == 1.0
    assert transitions[1].reward_label_scope == "trajectory_terminal"
    assert "score" not in transitions[1].feedback_text


def test_load_terminal_transitions_reads_trajectory_directory(tmp_path):
    run = tmp_path / "sample"
    run.mkdir()
    (run / "traj.json").write_text(json.dumps(_trajectory()), encoding="utf-8")

    transitions = load_terminal_transitions(tmp_path, max_transitions=1)

    assert len(transitions) == 1
    assert transitions[0].trajectory_id == "trajectory-1"


def test_seta_transition_redacts_context_action_and_result():
    payload = _trajectory()
    payload["info"]["task_name"] = "password=task-name-secret"
    payload["info"]["task_id"] = "token=task-id-secret"
    payload["turns"][0]["context_messages"][0]["content"] = "OPENAI_API_KEY=sk-abcdefghijklmnop"
    payload["turns"][0]["context_messages"][0]["metadata"] = {"password": "nested-secret"}
    payload["turns"][0]["tool_calls"][0]["args"] = {
        "authorization": "Bearer secret-token-value"
    }
    payload["turns"][0]["tool_calls"][0]["result"] = "password=hunter2"

    transition = transitions_from_seta_trajectory(
        payload,
        source_path="/tmp/password=source-path-secret/traj.json",
    )[0]
    persisted = json.dumps(transition.to_dict(), ensure_ascii=False)

    assert "sk-abcdefghijklmnop" not in persisted
    assert "secret-token-value" not in persisted
    assert "hunter2" not in persisted
    assert "nested-secret" not in persisted
    assert "task-name-secret" not in persisted
    assert "task-id-secret" not in persisted
    assert "source-path-secret" not in persisted
    assert "[REDACTED]" in persisted


def test_seta_multi_interaction_turn_fails_closed():
    payload = _trajectory()
    payload["turns"][0]["sdk_model_turns"] = [{"id": 1}, {"id": 2}]

    with pytest.raises(ValueError, match="multi-interaction"):
        transitions_from_seta_trajectory(payload, source_path="/tmp/traj.json")


def test_unknown_record_schema_is_rejected(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps({"schema": "unknown_v99", "action_text": "pwd", "feedback_text": "/tmp"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported world-model record schema"):
        load_terminal_transitions(records)


def test_unverified_world_model_view_records_are_allowed_when_opted_in(tmp_path):
    records = tmp_path / "records.jsonl"
    record = {
        "schema": "openclaw_terminal_transition_v3",
        "uid": "a",
        "turn_idx": 0,
        "action_view_schema": "tool_call_bundle_v1",
        "action_text": "invalid_action_view",
        "feedback_source": "result_only_v1",
        "next_observation_text": "invalid result view",
        "context_text": "[{\"role\":\"user\",\"content\":\"inspect\"}]",
        "context_hash": "abc",
        "source_path": str(records),
        "done": False,
    }
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")

    transitions = load_terminal_transitions(
        records,
        allow_unverified_world_model_views=True,
    )
    assert len(transitions) == 1
    assert transitions[0].action_view_schema == "tool_call_bundle_v1_unverified"
    assert transitions[0].feedback_source == "result_only_v1_unverified"


def test_strict_load_rejects_malformed_canonical_records(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "schema": "openclaw_terminal_transition_v3",
                "uid": "a",
                "turn_idx": 0,
                "action_view_schema": "tool_call_bundle_v1",
                "action_text": "invalid_action_view",
                "feedback_source": "result_only_v1",
                "next_observation_text": "invalid result view",
                "context_text": "[{\"role\":\"user\",\"content\":\"inspect\"}]",
                "context_hash": "abc",
                "source_path": str(records),
                "done": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_terminal_transitions(records)


def test_empty_and_duplicate_records_fail_closed(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text(
        json.dumps({"schema": "openclaw_terminal_transition_v3", "uid": "a"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty action or feedback"):
        load_terminal_transitions(empty)

    row = {
        "schema": "openclaw_terminal_transition_v3",
        "uid": "a",
        "action_text": "pwd",
        "feedback_text": "/tmp",
    }
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join([json.dumps(row), json.dumps(row)]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate transition_id"):
        load_terminal_transitions(duplicate)


def test_redacted_record_reload_preserves_original_source_path(tmp_path):
    source = transitions_from_seta_trajectory(
        _trajectory(),
        source_path="/original/run/traj.json",
    )[0]
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(source.to_dict()) + "\n", encoding="utf-8")

    restored = load_terminal_transitions(records)

    assert restored[0].source_path == "/original/run/traj.json"
    assert restored[0].to_dict() == source.to_dict()
