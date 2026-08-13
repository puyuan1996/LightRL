import hashlib
import json

import torch

from slime.world_model.cache_text_hidden import (
    _build_cache_integrity_metadata,
    validate_hidden_cache_integrity,
)
from slime.world_model.subset_hidden_cache import build_single_call_subset


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_single_call_subset_preserves_alignment_and_provenance(tmp_path):
    records = [
        {"transition_id": "a", "tool_names": ["shell_exec"], "task_id": "x"},
        {
            "transition_id": "b",
            "tool_names": ["shell_exec", "shell_view"],
            "task_id": "y",
        },
        {"transition_id": "c", "tool_names": ["shell_view"], "task_id": "z"},
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    payload = {
        "record_count": 3,
        "record_metadata": records,
        "state_hidden": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "action_hidden": torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
        "target_hidden": torch.arange(24, 36, dtype=torch.float32).reshape(3, 4),
        "has_next": torch.tensor([True, False, True]),
    }
    payload["metadata"] = _build_cache_integrity_metadata(
        payload,
        input_records_sha256=_sha256(records_path),
        encoder_config={
            "encoder": "hf-policy",
            "encoder_long_text_mode": "tail_v1",
        },
    )
    cache_path = tmp_path / "cache.pt"
    torch.save(payload, cache_path)

    output_dir = tmp_path / "subset" / "result_only_dataset"
    manifest = build_single_call_subset(
        records_path, cache_path, output_dir, expected_count=2
    )

    subset_records = [
        json.loads(line)
        for line in (output_dir / "result_only_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["transition_id"] for row in subset_records] == ["a", "c"]
    subset = torch.load(
        tmp_path / "subset" / "hierarchical_hidden_cache.pt",
        map_location="cpu",
        weights_only=True,
    )
    validate_hidden_cache_integrity(subset, require_verified=True)
    assert subset["record_count"] == 2
    assert torch.equal(subset["state_hidden"], payload["state_hidden"][[0, 2]])
    assert manifest["output"]["record_count"] == 2
    assert (
        subset["metadata"]["subset_provenance"]["source_records_sha256"]
        == _sha256(records_path)
    )
    assert subset["metadata"]["encoder_config"]["fixed_target_backbone"] is False
    assert (
        subset["metadata"]["encoder_config"]["fixed_target_model_name_or_path"]
        is None
    )
    assert [
        row["field"]
        for row in subset["metadata"]["subset_provenance"][
            "encoder_config_migrations"
        ]
    ] == ["fixed_target_backbone", "fixed_target_model_name_or_path"]
