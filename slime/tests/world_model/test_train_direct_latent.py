from __future__ import annotations

import json
import sys

import pytest
import torch

from slime.world_model.action_view import render_tool_call_bundle
from slime.world_model.result_view import render_result_only_view
from slime.world_model.seta_dataset import TerminalTransition
from slime.world_model.train_direct_latent import (
    RawHiddenDirectPredictor,
    _nearest_hidden_width,
    _parameter_count_formula,
    _target_keys,
    main as direct_main,
)
from slime.world_model.train_latent import main as latent_main


def test_parameter_matching_width_is_near_reference_budget() -> None:
    target = 52_627_507
    width = _nearest_hidden_width(4096, target)
    count = _parameter_count_formula(4096, width)
    assert width == 4280
    assert count == 52_625_592
    assert abs(count - target) / target < 0.001


def test_direct_input_modes_are_parameter_matched_and_invariant() -> None:
    state = torch.randn(4, 8)
    action = torch.randn(4, 8)
    other = torch.randn(4, 8)
    observed = RawHiddenDirectPredictor(8, 12, input_mode="observed")
    state_only = RawHiddenDirectPredictor(8, 12, input_mode="state_only")
    action_only = RawHiddenDirectPredictor(8, 12, input_mode="action_only")
    state_only.load_state_dict(observed.state_dict())
    action_only.load_state_dict(observed.state_dict())

    assert torch.equal(state_only(state, action), state_only(state, other))
    assert torch.equal(action_only(state, action), action_only(other, action))
    counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in (observed, state_only, action_only)
    }
    assert len(counts) == 1


def test_invalid_direct_input_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown input_mode"):
        RawHiddenDirectPredictor(8, 12, input_mode="bad")


def test_feedback_and_next_state_target_keys_are_explicit() -> None:
    records = [
        {
            "next_observation_hash": "feedback-a",
            "next_context_hash": "context-a",
        }
    ]
    cache = {"record_metadata": [{"next_state_view_hash": "belief-a"}]}
    assert _target_keys(
        cache, records, [0], prediction_target="feedback"
    ) == ["feedback-a"]
    assert _target_keys(
        cache, records, [0], prediction_target="next_state"
    ) == ["belief-a"]


def test_target_keys_fail_closed_when_requested_view_is_missing() -> None:
    with pytest.raises(ValueError, match="feedback equivalence"):
        _target_keys(
            {"record_metadata": [{}]},
            [{"next_context_hash": "next"}],
            [0],
            prediction_target="feedback",
        )


def test_feedback_direct_baseline_runs_against_verified_cache(
    tmp_path, monkeypatch
) -> None:
    rows = [
        TerminalTransition(
            trajectory_id=f"trajectory-{index}",
            task_name=f"task-{index}",
            data_source="test",
            turn_idx=0,
            context_messages=[{"role": "user", "content": f"state {index}"}],
            action_text=render_tool_call_bundle(
                [{"tool_name": "shell_exec", "args": {"index": index}}]
            ),
            feedback_text=render_result_only_view([f"result {index}"]),
            next_context_messages=None,
            done=True,
            reward=None,
            status="completed",
            source_path="test",
            task_id=f"task-{index}",
            has_tool_result=True,
            feedback_source="result_only_v1",
            action_view_schema="tool_call_bundle_v1",
            tool_names=("shell_exec",),
        )
        for index in range(4)
    ]
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(reference),
            "--encoder",
            "hash",
            "--hash-hidden-dim",
            "8",
            "--latent-dim",
            "8",
            "--predictor-num-heads",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--val-ratio",
            "0.5",
            "--split-group-key",
            "task_id",
        ],
    )
    latent_main()

    output = tmp_path / "direct"
    assert direct_main(
        [
            "--records",
            str(reference / "records.jsonl"),
            "--cache",
            str(reference / "hidden_cache.pt"),
            "--reference-checkpoint",
            str(reference / "latent_world_model.pt"),
            "--output-dir",
            str(output),
            "--prediction-target",
            "feedback",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["prediction_target"] == "feedback"
    assert summary["population"] == "all"
    assert summary["train_count"] + summary["val_count"] == 4
