from types import SimpleNamespace

import pytest
import torch

from slime.world_model.action_view import parse_tool_call_bundle, render_tool_call_bundle
from slime.world_model.replay_buffer import (
    TrajectoryReplayBuffer,
    _records_sha256,
    world_model_records_from_samples,
)
from slime.world_model.result_view import parse_result_only_view, render_result_only_view


def test_trajectory_replay_buffer_roundtrip(tmp_path):
    buffer = TrajectoryReplayBuffer(buffer_size=2, seed=7)
    buffer.push(
        [
            {"uid": "a", "turn_idx": 0, "action_text": "one", "reward_score": 0.0},
            {"uid": "b", "turn_idx": 0, "action_text": "two", "reward_score": 1.0},
            {"uid": "c", "turn_idx": 0, "action_text": "three", "reward_score": -1.0},
        ],
        current_step=3,
    )
    path = tmp_path / "replay.pt"
    buffer.save(path)
    loaded = TrajectoryReplayBuffer.load(path)

    assert len(loaded) == 2
    assert {row["uid"] for row in loaded.records()} == {"b", "c"}
    assert loaded.provenance_verified is True
    assert loaded.total_evicted == 1
    assert loaded.stats()["wm_replay_total_evicted"] == 1.0
    assert len(loaded.sample(10, current_step=4)) == 2


def test_replay_digest_tampering_is_rejected(tmp_path):
    buffer = TrajectoryReplayBuffer(buffer_size=2)
    buffer.push([{"uid": "a", "turn_idx": 0, "action_text": "one"}])
    state = buffer.state_dict()
    state["records"][0]["action_text"] = "tampered"
    path = tmp_path / "tampered.pt"
    torch.save(state, path)

    with pytest.raises(ValueError, match="digest mismatch"):
        TrajectoryReplayBuffer.load(path)


def test_verified_replay_rejects_duplicate_and_non_object_rows(tmp_path):
    buffer = TrajectoryReplayBuffer(buffer_size=2)
    buffer.push([{"uid": "a", "turn_idx": 0, "action_text": "one"}])

    duplicate = buffer.state_dict()
    duplicate["buffer_size"] = 2
    duplicate["records"].append(dict(duplicate["records"][0]))
    duplicate["records_sha256"] = _records_sha256(duplicate["records"])
    duplicate_path = tmp_path / "duplicate.pt"
    torch.save(duplicate, duplicate_path)
    with pytest.raises(ValueError, match="duplicate transition_id"):
        TrajectoryReplayBuffer.load(duplicate_path)

    invalid = buffer.state_dict()
    invalid["buffer_size"] = 2
    invalid["records"].append("not-an-object")
    invalid["records_sha256"] = _records_sha256(invalid["records"])
    invalid_path = tmp_path / "invalid.pt"
    torch.save(invalid, invalid_path)
    with pytest.raises(TypeError, match="not an object"):
        TrajectoryReplayBuffer.load(invalid_path)


def test_legacy_replay_requires_explicit_diagnostic_opt_in(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema_version": "openclaw_terminal_wm_replay_v1",
            "buffer_size": 2,
            "records": [{"uid": "a", "turn_idx": 0, "action_text": "one"}],
        },
        path,
    )

    with pytest.raises(ValueError, match="legacy replay"):
        TrajectoryReplayBuffer.load(path)
    loaded = TrajectoryReplayBuffer.load(path, require_verified=False)
    assert loaded.provenance_verified is False


def test_verified_replay_rejects_falsely_marked_unredacted_records(tmp_path):
    buffer = TrajectoryReplayBuffer(buffer_size=2)
    buffer.push([{"uid": "a", "turn_idx": 0, "action_text": "safe"}])
    state = buffer.state_dict()
    state["records"][0]["action_text"] = "password=hunter2"
    state["records_sha256"] = _records_sha256(state["records"])
    path = tmp_path / "unredacted.pt"
    torch.save(state, path)

    with pytest.raises(ValueError, match="unredacted record"):
        TrajectoryReplayBuffer.load(path)


def test_replay_rejects_skipped_world_model_records():
    buffer = TrajectoryReplayBuffer(buffer_size=2)

    admitted = buffer.push(
        [{"world_model_skipped": {"reason": "multi_interaction_turn_requires_harness_adapter"}}]
    )

    assert admitted == 0
    assert len(buffer) == 0
    assert buffer.total_rejected == 1


def test_replay_redacts_records_before_persistence():
    buffer = TrajectoryReplayBuffer(buffer_size=2)

    buffer.push(
        [
            {
                "uid": "a",
                "turn_idx": 0,
                "action_text": (
                    'Authorization: Bearer secret-token-value '
                    '{"password":{"value":"nested-cleartext"}}'
                ),
                "feedback": {"password": "hunter2"},
            }
        ]
    )

    payload = str(buffer.state_dict())
    assert "secret-token-value" not in payload
    assert "hunter2" not in payload
    assert "nested-cleartext" not in payload
    assert "[REDACTED]" in payload
    assert buffer.records()[0]["redaction_applied"] is True


def test_replay_redaction_preserves_canonical_structured_views():
    action = render_tool_call_bundle(
        [
            {
                "tool_name": "shell_write_content_to_file",
                "args": {
                    "content": (
                        'password = "[REDACTED]"\n'
                        "hashed_password = [REDACTED], salt)"
                    ),
                    "file_path": "/tmp/generate_hash.py",
                },
            },
            {"tool_name": "shell_exec", "args": {"command": "python /tmp/generate_hash.py"}},
        ]
    )
    feedback = render_result_only_view(
        ['password = "[REDACTED]"\ncompleted']
    )
    buffer = TrajectoryReplayBuffer(buffer_size=2)

    admitted = buffer.push(
        [
            {
                "uid": "canonical",
                "turn_idx": 0,
                "action_view_schema": "tool_call_bundle_v1",
                "action_text": action,
                "feedback_source": "result_only_v1",
                "feedback_text": feedback,
                "next_observation_text": feedback,
            }
        ]
    )

    assert admitted == 1
    record = buffer.records()[0]
    assert record["action_text"] == action
    assert record["feedback_text"] == feedback
    assert record["next_observation_text"] == feedback
    parse_tool_call_bundle(record["action_text"])
    parse_result_only_view(record["feedback_text"])


def test_replay_loads_unverified_canonical_records_when_enabled(tmp_path):
    buffer = TrajectoryReplayBuffer(buffer_size=2)
    buffer.push(
        [
            {
                "uid": "canonical",
                "turn_idx": 0,
                "action_view_schema": "tool_call_bundle_v1",
                "action_text": render_tool_call_bundle(
                    [
                        {
                            "tool_name": "shell_exec",
                            "args": {"command": "python -V"},
                        }
                    ]
                ),
            }
        ]
    )
    state = buffer.state_dict()
    state["records"][0]["action_text"] = "invalid<unparseable>"
    state["records"][0]["feedback_source"] = "result_only_v1"
    state["records"][0]["feedback_text"] = "not-result-only"
    state["records"][0]["next_observation_text"] = "not-result-only"
    state["records_sha256"] = _records_sha256(state["records"])
    path = tmp_path / "mixed_invalid_records.pt"
    torch.save(state, path)

    loaded = TrajectoryReplayBuffer.load(path, require_verified=False, allow_unverified_records=True)
    assert len(loaded.records()) == 1
    loaded_record = loaded.records()[0]
    assert loaded_record["action_view_schema"] == "tool_call_bundle_v1_unverified"
    assert loaded_record["feedback_source"] == "result_only_v1_unverified"
    assert loaded.provenance_verified is False


def test_replay_returns_deep_copies_of_nested_records():
    buffer = TrajectoryReplayBuffer(buffer_size=2)
    buffer.push([{"uid": "a", "turn_idx": 0, "feedback": {"status": "safe"}}])

    records = buffer.records()
    sampled = buffer.sample(1)
    records[0]["feedback"]["status"] = "mutated-records"
    sampled[0]["feedback"]["status"] = "mutated-sample"

    assert buffer.records()[0]["feedback"]["status"] == "safe"


def test_world_model_records_from_grouped_samples():
    sample = SimpleNamespace(
        train_metadata={"world_model": {"uid": "x", "turn_idx": 0}},
        metadata={},
    )

    assert world_model_records_from_samples([[sample]]) == [{"uid": "x", "turn_idx": 0}]


def test_world_model_record_collection_rejects_silent_empty_batches():
    sample = SimpleNamespace(train_metadata={}, metadata={})

    with pytest.raises(RuntimeError, match="produced no transition metadata"):
        world_model_records_from_samples([[sample]], require_nonempty=True)
