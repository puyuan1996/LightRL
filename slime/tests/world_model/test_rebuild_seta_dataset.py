import hashlib
import json
from pathlib import Path

import pytest

from slime.world_model.rebuild_seta_dataset import _record_from_transition, rebuild_dataset
from slime.world_model.seta_dataset import load_terminal_transitions, transitions_from_seta_trajectory


def _tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        file_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest)
    return digest.hexdigest()


def _payload() -> dict:
    return {
        "trajectory_format": "openclaw-terminal-rl-1",
        "info": {"uid": "traj-1", "task_id": "task-1", "status": "completed"},
        "reward": {"score": 1.0},
        "turns": [
            {
                "turn_idx": 0,
                "context_messages": [{"role": "user", "content": "x" * 5000}],
                "assistant_output": "a" * 5000,
                "sdk_model_turns": [{}],
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "bash",
                        "args": {"command": "pwd"},
                        "result": "r" * 5000,
                    }
                ],
            },
            {
                "turn_idx": 1,
                "context_messages": [{"role": "user", "content": "next"}],
                "assistant_output": "two calls",
                "sdk_model_turns": [{}],
                "tool_calls": [
                    {
                        "tool_call_id": "call-2",
                        "tool_name": "bash",
                        "args": {"command": "pwd"},
                        "result": "/tmp",
                    },
                    {
                        "tool_call_id": "call-3",
                        "tool_name": "bash",
                        "args": {"command": "id"},
                        "result": "uid=1",
                    },
                ],
            },
        ],
    }


def test_rebuild_is_lossless_and_does_not_split_multi_call(tmp_path):
    source = tmp_path / "source"
    run = source / "run"
    run.mkdir(parents=True)
    trajectory = run / "traj.json"
    trajectory.write_text(json.dumps(_payload()), encoding="utf-8")
    (source / "index.jsonl").write_text('{"uid":"traj-1"}\n', encoding="utf-8")
    index_digest = hashlib.sha256((source / "index.jsonl").read_bytes()).hexdigest()
    tree_digest = _tree_digest(source, [trajectory])

    manifest = rebuild_dataset(
        source,
        tmp_path / "rebuilt",
        expected_file_count=1,
        expected_index_sha256=index_digest,
        expected_tree_sha256=tree_digest,
    )

    assert manifest["audit"]["tool_feedback_transition_count"] == 2
    assert manifest["audit"]["single_call_transition_count"] == 1
    assert manifest["audit"]["multi_call_bundle_transition_count"] == 1
    assert manifest["audit"]["canonical_tool_call_count"] == 3
    assert manifest["audit"]["canonical_tool_call_record_count"] == 2
    assert manifest["adapter_contract"]["character_truncation"] is False
    assert (
        manifest["adapter_contract"]["tool_call_bundle"][
            "assistant_reasoning_included"
        ]
        is False
    )
    rows = load_terminal_transitions(
        tmp_path / "rebuilt" / "turn_bundle_records.jsonl",
        max_text_chars=10_000,
    )
    assert len(rows[0].action_text) > 4096
    assert len(rows[0].feedback_text) > 4096
    multi = load_terminal_transitions(
        tmp_path / "rebuilt" / "multi_call_bundle_records.jsonl",
        max_text_chars=10_000,
    )
    assert len(multi) == 1
    assert multi[0].tool_names == ("bash", "bash")
    canonical = [
        json.loads(line)
        for line in (
            tmp_path / "rebuilt" / "tool_call_bundle_records.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(canonical) == 2
    assert canonical[0]["action_text"].startswith('<call index=0 name="bash">')
    assert "a" * 100 not in canonical[0]["action_text"]
    assert canonical[1]["action_text"].count("<call index=") == 2
    assert canonical[0]["action_hash"] != rows[0].to_dict()["action_hash"]


def test_fast_rebuild_serializer_matches_canonical_record():
    transition = transitions_from_seta_trajectory(
        _payload(), source_path="/tmp/source/traj.json", max_text_chars=10_000
    )[0]
    assert _record_from_transition(transition) == transition.to_dict()


def test_rebuild_fails_closed_on_provenance_mismatch(tmp_path):
    source = tmp_path / "source"
    run = source / "run"
    run.mkdir(parents=True)
    (run / "traj.json").write_text(json.dumps(_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="file count mismatch"):
        rebuild_dataset(source, tmp_path / "rebuilt", expected_file_count=2)


def test_rebuild_rejects_invalid_turn_and_call_boundaries(tmp_path):
    payload = _payload()
    payload["turns"][1]["turn_idx"] = 3
    source = tmp_path / "source"
    run = source / "run"
    run.mkdir(parents=True)
    (run / "traj.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="turn_idx"):
        rebuild_dataset(source, tmp_path / "bad-turn")

    payload = _payload()
    payload["turns"][0]["tool_calls"][0]["tool_call_id"] = ""
    (run / "traj.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="call ID"):
        rebuild_dataset(source, tmp_path / "bad-call")
