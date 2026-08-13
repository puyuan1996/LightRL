from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import pytest

from slime.world_model.cache_text_hidden import validate_hidden_cache_integrity
from slime.world_model.hidden_encoder import hash_hidden_batch
from slime.world_model.action_view import render_tool_call_bundle
from slime.world_model.result_view import render_result_only_view
from slime.world_model.seta_dataset import TerminalTransition, load_terminal_transitions
from slime.world_model.sharded_hidden_cache import (
    merge_shards,
    split_records,
    validate_canonical_records,
    wait_for_start_barrier,
)
from slime.world_model.train_latent import _save_hidden_cache


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transition(index: int) -> TerminalTransition:
    return TerminalTransition(
        trajectory_id=f"traj-{index}",
        task_name="task",
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
        has_tool_result=True,
        feedback_source="result_only_v1",
        action_view_schema="tool_call_bundle_v1",
        tool_names=("shell_exec",),
    )


def test_split_and_merge_verified_hidden_cache(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    rows = [_transition(index) for index in range(5)]
    records.write_text(
        "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    shard_dir = tmp_path / "shards"
    manifest = split_records(records, shard_dir, 2)
    encoder_config = {
        "encoder": "hash",
        "hidden_dim": 8,
        "schema": "test_hash_v1",
    }
    for shard in manifest["shards"]:
        shard_records = Path(shard["records"])
        transitions = load_terminal_transitions(
            shard_records,
            require_tool_feedback=True,
            max_text_chars=10_000,
        )
        hidden = hash_hidden_batch(transitions, 8)
        _save_hidden_cache(
            Path(shard["cache"]),
            hidden=hidden,
            transitions=transitions,
            input_records_sha256=_sha256(shard_records),
            encoder_config=encoder_config,
        )

    output = tmp_path / "merged.pt"
    summary = merge_shards(shard_dir / "shard_manifest.json", records, output)
    payload = torch.load(output, map_location="cpu", weights_only=True)

    assert summary["record_count"] == 5
    assert payload["state_hidden"].shape == (5, 8)
    assert [
        row["transition_id"] for row in payload["record_metadata"]
    ] == [row.transition_id for row in rows]
    assert validate_hidden_cache_integrity(payload, require_verified=True)[
        "verified"
    ]


def test_validate_canonical_records_roundtrips_loader(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    rows = [_transition(index) for index in range(3)]
    records.write_text(
        "".join(
            json.dumps(row.to_dict(), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    summary = validate_canonical_records(
        records,
        tmp_path / "audit.json",
        expected_count=3,
        expected_calls=3,
    )

    assert summary["records"] == 3
    assert summary["calls"] == summary["results"] == 3
    assert summary["roundtrip_mismatch_count"] == 0


def test_encoder_start_barrier_writes_ready_and_observes_start(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready.json"
    start = tmp_path / "start"
    start.touch()

    wait_for_start_barrier(ready, start, timeout_seconds=1.0)

    payload = json.loads(ready.read_text(encoding="utf-8"))
    assert payload["pid"] > 0
    assert payload["ready_at"]


def test_encoder_start_barrier_times_out(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="start barrier"):
        wait_for_start_barrier(
            tmp_path / "ready.json",
            tmp_path / "missing-start",
            timeout_seconds=0.01,
        )


def test_json_signal_is_written_atomically(tmp_path: Path) -> None:
    from slime.world_model.sharded_hidden_cache import _write_json

    signal = tmp_path / "compute_done.json"
    _write_json(signal, {"record_count": 3})

    assert json.loads(signal.read_text(encoding="utf-8")) == {
        "record_count": 3
    }
    assert not signal.with_suffix(".json.tmp").exists()
