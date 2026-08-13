from __future__ import annotations

import json
from pathlib import Path

from slime.world_model.build_result_only_dataset import build_result_only_dataset
from slime.world_model.seta_dataset import TerminalTransition


def _transition() -> TerminalTransition:
    return TerminalTransition(
        trajectory_id="traj-1",
        task_name="task",
        data_source="test",
        turn_idx=0,
        context_messages=[{"role": "user", "content": "run"}],
        action_text="use tools",
        feedback_text=(
            "<tool_result name=shell_exec>\none\n</tool_result>\n\n"
            "<tool_result name=python>\ntwo\n</tool_result>"
        ),
        next_context_messages=None,
        done=True,
        reward=None,
        status="completed",
        source_path="/redacted/source",
        task_id="task-1",
        has_tool_result=True,
        feedback_source="tool_result",
        tool_names=("shell_exec", "python"),
    )


def test_build_result_only_dataset_audits_and_preserves_labels(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps(_transition().to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_result_only_dataset(source, tmp_path / "output", expected_count=1)
    row = json.loads(
        (tmp_path / "output/result_only_records.jsonl").read_text(encoding="utf-8")
    )

    assert row["feedback_source"] == "result_only_v1"
    assert row["tool_names"] == ["shell_exec", "python"]
    assert row["next_observation_text"] == (
        "<result index=0>\none\n</result>\n\n"
        "<result index=1>\ntwo\n</result>"
    )
    assert row["next_observation_hash"] != _transition().to_dict()["next_observation_hash"]
    assert manifest["audit"]["result_block_count"] == 2
    assert manifest["audit"]["wrapper_leakage_count"] == 0
    assert manifest["audit"]["tool_name_result_count_mismatch_records"] == 0
    assert manifest["audit"]["redaction_idempotence_mismatch_count"] == 0
    mapping = json.loads(
        (tmp_path / "output/transition_map.jsonl").read_text(encoding="utf-8")
    )
    assert mapping["source_transition_id"] == _transition().transition_id
    assert mapping["result_only_transition_id"] == row["transition_id"]
