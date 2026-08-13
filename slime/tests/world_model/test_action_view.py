from __future__ import annotations

import json
import sys

import pytest

from slime.world_model.action_view import (
    parse_tool_call_bundle,
    render_tool_call_bundle,
)
from slime.world_model.result_view import render_result_only_view
from slime.world_model.seta_dataset import (
    TerminalTransition,
    load_terminal_transitions,
)


def test_tool_call_bundle_is_canonical_ordered_and_redacted() -> None:
    rendered = render_tool_call_bundle(
        [
            {
                "tool_name": "shell_exec",
                "args": {
                    "command": "pwd",
                    "Authorization": "Bearer secret-token",
                },
            },
            {"tool_name": "shell_view", "args": {"path": "/tmp/a"}},
        ]
    )
    assert rendered.startswith('<call index=0 name="shell_exec">')
    assert '\n\n<call index=1 name="shell_view">' in rendered
    assert "secret-token" not in rendered
    parsed = parse_tool_call_bundle(rendered)
    assert [row["tool_name"] for row in parsed] == ["shell_exec", "shell_view"]
    assert render_tool_call_bundle(parsed) == rendered


@pytest.mark.parametrize(
    "value",
    [
        "reasoning\n<call index=0 name=\"x\">\n{}\n</call>",
        '<call index=1 name="x">\n{}\n</call>',
        '<call index=0 name="x">\nnot-json\n</call>',
        '<call index=0 name="x">\n{}\n</call>\ntrailing',
    ],
)
def test_tool_call_bundle_rejects_noncanonical_text(value: str) -> None:
    with pytest.raises(ValueError):
        parse_tool_call_bundle(value)


def test_canonical_bundle_survives_transition_roundtrip(tmp_path) -> None:
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
            }
        ]
    )
    feedback = render_result_only_view(
        ['password = "[REDACTED]"\ncompleted']
    )
    transition = TerminalTransition(
        trajectory_id="trajectory-1",
        task_name="task",
        data_source="test",
        turn_idx=0,
        context_messages=[{"role": "user", "content": "write a file"}],
        action_text=action,
        feedback_text=feedback,
        next_context_messages=None,
        done=True,
        reward=None,
        status="completed",
        source_path="test",
        has_tool_result=True,
        feedback_source="result_only_v1",
        action_view_schema="tool_call_bundle_v1",
        tool_names=("shell_write_content_to_file",),
    )
    record = transition.to_dict()
    assert record["action_text"] == action
    assert record["next_observation_text"] == feedback

    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_terminal_transitions(
        records,
        require_tool_feedback=True,
        max_text_chars=sys.maxsize,
    )

    assert loaded[0].action_text == action
    assert loaded[0].feedback_text == feedback
    assert loaded[0].to_dict() == record


def test_result_only_loader_rejects_nonredacted_target(tmp_path) -> None:
    action = render_tool_call_bundle(
        [{"tool_name": "shell_exec", "args": {"command": "true"}}]
    )
    record = {
        "schema": "openclaw_terminal_transition_v3",
        "uid": "trajectory-1",
        "turn_idx": 0,
        "context_messages": [{"role": "user", "content": "run"}],
        "action_text": action,
        "next_observation_text": render_result_only_view(
            ['password = "not-redacted"']
        ),
        "done": True,
        "has_tool_result": True,
        "feedback_source": "result_only_v1",
        "action_view_schema": "tool_call_bundle_v1",
        "tool_names": ["shell_exec"],
    }
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not canonically redacted"):
        load_terminal_transitions(
            records,
            require_tool_feedback=True,
            max_text_chars=sys.maxsize,
        )
