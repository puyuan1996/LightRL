import json

from slime.world_model.state_view import (
    BELIEF_VIEW_V1,
    belief_view_metadata,
    belief_view_parts,
)


def _messages():
    return [
        {
            "role": "user",
            "content": "inspect repository",
            "reward": 99,
            "rollout_id": "secret-id",
        },
        {"role": "assistant", "content": "rm -rf should not enter state"},
        {
            "role": "tool",
            "name": "bash",
            "content": "file_a.py\nfile_b.py",
            "eval_status": "hidden",
        },
    ]


def test_belief_view_separates_task_anchor_and_dynamic_suffix():
    conditioning, suffix = belief_view_parts(_messages(), max_events=3)

    assert "inspect repository" in conditioning
    assert "inspect repository" not in suffix
    assert "file_a.py" in suffix
    assert "rm -rf" not in suffix
    assert "reward" not in suffix
    assert "eval_status" not in suffix
    assert "rollout_id" not in conditioning + suffix
    assert suffix.startswith(f'<STATE_VIEW version="{BELIEF_VIEW_V1}">')

    payload = json.loads(suffix.split("\n", 1)[1].rsplit("\n", 1)[0])
    assert payload["version"] == BELIEF_VIEW_V1
    assert payload["events"] == [
        {"content": "file_a.py\nfile_b.py", "name": "bash", "role": "tool"}
    ]


def test_belief_view_keeps_only_bounded_visible_events():
    messages = [{"role": "user", "content": "task"}] + [
        {"role": "tool", "content": f"result-{index}"}
        for index in range(5)
    ]

    _, suffix = belief_view_parts(messages, max_events=2)

    assert "result-2" not in suffix
    assert "result-3" in suffix
    assert "result-4" in suffix


def test_belief_view_metadata_binds_current_and_next_views():
    current = [{"role": "user", "content": "task"}]
    future = current + [{"role": "tool", "content": "done"}]

    metadata = belief_view_metadata(current, future, max_events=3)

    assert metadata["state_view"] == BELIEF_VIEW_V1
    assert metadata["state_view_pooling"] == "suffix_last_v1"
    assert metadata["state_view_hash"] != metadata["next_state_view_hash"]
    assert metadata["state_view_allowlist"] == ["role", "name", "content"]
