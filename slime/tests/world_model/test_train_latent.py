import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import slime.world_model.train_latent as train_latent_module

from slime.world_model.cache_text_hidden import validate_hidden_cache_integrity
from slime.world_model.checkpoint import select_evaluation_indices
from slime.world_model.replay_buffer import TrajectoryReplayBuffer
from slime.world_model.seta_dataset import TerminalTransition
from slime.world_model.train_latent import (
    _build_parser,
    _cache_hidden,
    _encoder_config,
    _group_value,
    _loss_curve_summary,
    _roundtrip_replay,
    _run_epoch,
    _split_indices,
    _split_indices_from_manifest,
    _training_contract_pending,
    _validate_args,
    _validate_next_state_supervision,
    _value_label_contract,
    _write_records,
    main,
)
from slime.world_model.hidden_encoder import hash_hidden_batch
from slime.world_model.modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _transition(trajectory_id: str, turn_idx: int = 0) -> TerminalTransition:
    return TerminalTransition(
        trajectory_id=trajectory_id,
        task_name=f"task-{trajectory_id}",
        data_source="test",
        turn_idx=turn_idx,
        context_messages=[{"role": "user", "content": f"state {trajectory_id}"}],
        action_text=f"act {trajectory_id}",
        feedback_text=f"result {trajectory_id}",
        next_context_messages=None,
        done=True,
        reward=None,
        status="completed",
        source_path="test.jsonl",
    )


def test_final_hidden_cache_is_inference_only():
    class RecordingEncoder:
        def __init__(self) -> None:
            self.grad_enabled: list[bool] = []

        def __call__(self, rows, *, include_auxiliary_targets):
            assert include_auxiliary_targets is True
            self.grad_enabled.append(torch.is_grad_enabled())
            count = len(rows)
            return {
                "state_hidden": torch.ones(count, 2, requires_grad=True),
                "action_hidden": torch.ones(count, 2, requires_grad=True),
                "target_hidden": torch.ones(count, 2, requires_grad=True),
                "next_state_hidden": torch.ones(count, 2, requires_grad=True),
                "has_next": torch.ones(count, dtype=torch.bool),
            }

    encoder = RecordingEncoder()
    hidden = _cache_hidden(
        [_transition("a"), _transition("b")],
        encoder_kind="hf-policy",
        hash_hidden_dim=2,
        policy_encoder=encoder,
        batch_size=1,
        state_view="full_context_v1",
        belief_max_events=3,
    )

    assert encoder.grad_enabled == [False, False]
    assert hidden["state_hidden"].shape == (2, 2)
    assert hidden["state_hidden"].requires_grad is False


def test_group_split_is_disjoint_and_fails_closed_without_groups():
    rows = [_transition("a", 0), _transition("a", 1), _transition("b", 0), _transition("c", 0)]
    train, val, metadata = _split_indices(rows, 0.25, 7, "trajectory_id")

    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(len(rows)))
    assert metadata["strategy"] == "group_holdout"
    assert {rows[index].trajectory_id for index in train}.isdisjoint(
        {rows[index].trajectory_id for index in val}
    )

    with pytest.raises(ValueError, match="complete 'missing_key' metadata"):
        _split_indices(rows, 0.25, 7, "missing_key")


def test_frozen_group_kfold_manifest_is_digest_bound_and_group_disjoint(tmp_path):
    rows = [_transition("a", 0), _transition("a", 1), _transition("b", 0), _transition("c", 0)]
    records_sha256 = _write_records(tmp_path / "records.jsonl", rows)
    manifest_path = tmp_path / "fold.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "openclaw_group_kfold_split_v1",
                "records_sha256": records_sha256,
                "record_count": 4,
                "group_key": "trajectory_id",
                "fold_index": 0,
                "fold_count": 2,
                "assignment_seed": 20260811,
                "train_indices": [2, 3],
                "val_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )

    train, val, metadata = _split_indices_from_manifest(
        rows,
        manifest_path,
        records_sha256=records_sha256,
        group_key="trajectory_id",
    )

    assert train == [2, 3]
    assert val == [0, 1]
    assert metadata["strategy"] == "group_holdout"
    assert metadata["source"] == "frozen_group_kfold_manifest"
    assert metadata["fold_index"] == 0
    assert metadata["group_disjoint"] is True

    with pytest.raises(ValueError, match="records_sha256"):
        _split_indices_from_manifest(
            rows,
            manifest_path,
            records_sha256="0" * 64,
            group_key="trajectory_id",
        )


def test_frozen_group_kfold_manifest_rejects_group_overlap(tmp_path):
    rows = [_transition("a", 0), _transition("a", 1), _transition("b", 0)]
    records_sha256 = _write_records(tmp_path / "records.jsonl", rows)
    manifest_path = tmp_path / "bad-fold.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "openclaw_group_kfold_split_v1",
                "records_sha256": records_sha256,
                "record_count": 3,
                "group_key": "trajectory_id",
                "fold_index": 0,
                "fold_count": 2,
                "assignment_seed": 20260811,
                "train_indices": [1, 2],
                "val_indices": [0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not group-disjoint"):
        _split_indices_from_manifest(
            rows,
            manifest_path,
            records_sha256=records_sha256,
            group_key="trajectory_id",
        )


def test_hash_trainer_consumes_frozen_group_kfold_manifest(tmp_path, monkeypatch):
    rows = [_transition("a"), _transition("b"), _transition("c"), _transition("d")]
    records = tmp_path / "records.jsonl"
    records_sha256 = _write_records(records, rows)
    manifest = tmp_path / "fold.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "openclaw_group_kfold_split_v1",
                "records_sha256": records_sha256,
                "record_count": 4,
                "group_key": "trajectory_id",
                "fold_index": 1,
                "fold_count": 2,
                "assignment_seed": 20260811,
                "train_indices": [0, 2],
                "val_indices": [1, 3],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(output),
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
            "1",
            "--split-group-key",
            "trajectory_id",
            "--split-manifest",
            str(manifest),
        ],
    )

    main()

    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["split"]["source"] == "frozen_group_kfold_manifest"
    assert summary["split"]["fold_index"] == 1
    assert summary["split"]["train_indices"] == [0, 2]
    assert summary["split"]["val_indices"] == [1, 3]
    assert summary["hyperparameters"]["split_manifest"] == str(manifest.resolve())


def test_group_value_reads_declared_field_without_serializing(monkeypatch):
    row = _transition("a")

    def fail_serialization(_self):
        raise AssertionError("declared group field should not serialize the record")

    monkeypatch.setattr(TerminalTransition, "to_dict", fail_serialization)

    assert _group_value(row, "trajectory_id") == "a"


def test_only_task_cluster_split_is_confirmatory():
    rows = [
        replace(_transition("a"), task_id="task-a", task_cluster_id="cluster-a"),
        replace(_transition("b"), task_id="task-b", task_cluster_id="cluster-b"),
    ]

    _, _, task_split = _split_indices(rows, 0.5, 7, "task_id")
    _, _, cluster_split = _split_indices(rows, 0.5, 7, "task_cluster_id")

    assert task_split["confirmatory_group_key"] is False
    assert cluster_split["confirmatory_group_key"] is False
    assert cluster_split["strongest_available_group_key"] is True


def test_offline_replay_roundtrip_fails_before_capacity_eviction(tmp_path):
    rows = [_transition("a"), _transition("b")]

    with pytest.raises(ValueError, match="would evict transitions"):
        _roundtrip_replay(
            rows,
            output_path=tmp_path / "too-small.pt",
            buffer_size=1,
            seed=7,
        )

    sampled, stats = _roundtrip_replay(
        rows,
        output_path=tmp_path / "complete.pt",
        buffer_size=2,
        seed=7,
    )
    assert {row.transition_id for row in sampled} == {row.transition_id for row in rows}
    assert [row.transition_id for row in sampled] == [row.transition_id for row in rows]
    assert stats["wm_replay_size"] == 2.0
    assert stats["wm_replay_total_evicted"] == 0.0
    assert stats["wm_replay_total_sampled"] == 2.0
    assert TrajectoryReplayBuffer.load(tmp_path / "complete.pt").total_sampled == 2


def test_loss_curve_summary_reports_heldout_reduction():
    summary = _loss_curve_summary(
        [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.2},
            {"epoch": 2, "train_loss": 0.5, "val_loss": 0.9},
            {"epoch": 3, "train_loss": 0.25, "val_loss": 1.0},
        ]
    )

    assert summary["train"]["relative_reduction"] == pytest.approx(0.75)
    assert summary["validation"]["absolute_reduction"] == pytest.approx(0.2)
    assert summary["validation"]["best_epoch"] == 2


def test_duration_contract_requires_both_epoch_and_elapsed_time_floors():
    assert _training_contract_pending(
        completed_epochs=2,
        minimum_epochs=3,
        cumulative_train_seconds=21600.0,
        minimum_train_seconds=21600.0,
    )
    assert _training_contract_pending(
        completed_epochs=3,
        minimum_epochs=3,
        cumulative_train_seconds=21599.9,
        minimum_train_seconds=21600.0,
    )
    assert not _training_contract_pending(
        completed_epochs=3,
        minimum_epochs=3,
        cumulative_train_seconds=21600.0,
        minimum_train_seconds=21600.0,
    )


def test_split_seed_is_independent_from_training_seed():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--seed",
            "17",
            "--split-seed",
            "20260723",
        ]
    )

    assert args.seed == 17
    assert args.split_seed == 20260723


def test_belief_view_cli_is_explicit_and_versioned():
    args = _build_parser().parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--state-view",
            "belief_view_v1",
            "--belief-max-events",
            "4",
        ]
    )

    assert args.state_view == "belief_view_v1"
    assert args.belief_max_events == 4


def test_long_text_mode_and_chunk_batch_are_cache_fingerprinted():
    parser = _build_parser()
    base = [
        "--input",
        "records.jsonl",
        "--output-dir",
        "out",
        "--encoder",
        "hf-policy",
        "--hf-model",
        "model",
    ]
    tail = parser.parse_args(base)
    chunks = parser.parse_args(
        base
        + [
            "--encoder-long-text-mode",
            "hierarchical_chunks_v1",
            "--chunk-forward-batch-size",
            "7",
        ]
    )

    tail_config = _encoder_config(tail, final_backbone=False)
    chunk_config = _encoder_config(chunks, final_backbone=False)
    assert tail_config["encoder_long_text_mode"] == "tail_v1"
    assert chunk_config["encoder_long_text_mode"] == "hierarchical_chunks_v1"
    assert chunk_config["chunk_forward_batch_size"] == 7
    assert tail_config != chunk_config


def test_predictor_input_ablation_cli_is_explicit():
    args = _build_parser().parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--predictor-input-mode",
            "state_only",
        ]
    )

    assert args.predictor_input_mode == "state_only"


def test_prediction_form_cli_is_explicit():
    args = _build_parser().parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--prediction-form",
            "residual",
        ]
    )

    assert args.prediction_form == "residual"


def test_bounded_training_chunk_cli_is_explicit():
    args = _build_parser().parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--train-batches-per-epoch",
            "256",
            "--validation-batches-per-epoch",
            "64",
        ]
    )

    assert args.train_batches_per_epoch == 256
    assert args.validation_batches_per_epoch == 64


def test_validation_sigreg_is_repeatable_and_does_not_advance_rng():
    rows = [_transition("a"), _transition("b")]
    hidden = hash_hidden_batch(rows, hidden_dim=8)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            predictor_num_heads=2,
            sigreg_num_proj=8,
            value_head=False,
            uncertainty_head=False,
        )
    )
    torch.manual_seed(123)
    initial_rng_state = torch.random.get_rng_state().clone()

    first_loss, first_metrics, _, _ = _run_epoch(
        model=model,
        transitions=rows,
        indices=[0, 1],
        cached_hidden=hidden,
        policy_encoder=None,
        optimizer=None,
        batch_size=2,
        device=torch.device("cpu"),
        seed=20260723,
        sigreg_coef=0.09,
        action_contrast_coef=0.1,
        alignment_coef=0.1,
        value_coef=0.0,
    )
    assert torch.equal(torch.random.get_rng_state(), initial_rng_state)
    torch.rand(7)
    second_loss, second_metrics, _, _ = _run_epoch(
        model=model,
        transitions=rows,
        indices=[0, 1],
        cached_hidden=hidden,
        policy_encoder=None,
        optimizer=None,
        batch_size=2,
        device=torch.device("cpu"),
        seed=20260723,
        sigreg_coef=0.09,
        action_contrast_coef=0.1,
        alignment_coef=0.1,
        value_coef=0.0,
    )

    assert second_loss == pytest.approx(first_loss)
    assert second_metrics == pytest.approx(first_metrics)


def test_epoch_value_mask_count_is_a_count_not_a_batch_weighted_mean():
    rows = [replace(_transition("a"), reward=1.0), _transition("b")]
    hidden = hash_hidden_batch(rows, hidden_dim=8)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            predictor_num_heads=2,
            sigreg_num_proj=8,
            value_head=True,
            uncertainty_head=False,
        )
    )

    _, metrics, _, _ = _run_epoch(
        model=model,
        transitions=rows,
        indices=[0, 1],
        cached_hidden=hidden,
        policy_encoder=None,
        optimizer=None,
        batch_size=1,
        device=torch.device("cpu"),
        seed=7,
        sigreg_coef=0.09,
        action_contrast_coef=0.1,
        alignment_coef=0.1,
        value_coef=0.1,
    )

    assert metrics["wm/value_mask_count"] == 1.0


def test_run_epoch_trains_only_student_and_checkpoints_it():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(use_cache=True)
            self.gradient_checkpointing_calls = 0

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing_calls += 1

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.target_model = Backbone()
            self.target_model.requires_grad_(False)
            self.target_model.eval()
            self.backprop_to_llm = True

        def forward(self, transitions):
            count = len(transitions)
            base = torch.ones(count, 8)
            student = base * self.model.scale
            with torch.no_grad():
                target = base * self.target_model.scale
            return {
                "state_hidden": student,
                "action_hidden": student,
                "target_hidden": target,
                "next_state_hidden": target,
                "has_next": torch.ones(count, dtype=torch.bool),
            }

    rows = [
        replace(
            _transition("a"),
            next_context_messages=[{"role": "tool", "content": "result a"}],
            done=False,
        ),
        replace(
            _transition("b"),
            next_context_messages=[{"role": "tool", "content": "result b"}],
            done=False,
        ),
    ]
    encoder = Encoder()
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="mlp",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            stop_grad_target=True,
            value_head=False,
            uncertainty_head=False,
        )
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(encoder.model.parameters()),
        lr=1e-4,
    )

    _run_epoch(
        model=model,
        transitions=rows,
        indices=[0, 1],
        cached_hidden=None,
        policy_encoder=encoder,
        optimizer=optimizer,
        batch_size=2,
        device=torch.device("cpu"),
        seed=7,
        sigreg_coef=0.0,
        action_contrast_coef=0.0,
        alignment_coef=0.0,
        value_coef=0.0,
    )

    assert encoder.model.training is True
    assert encoder.target_model.training is False
    assert encoder.model.gradient_checkpointing_calls == 1
    assert encoder.model.config.use_cache is False
    assert encoder.target_model.config.use_cache is True
    assert encoder.model.scale.grad is not None
    assert encoder.target_model.scale.grad is None
    assert all(parameter.requires_grad is False for parameter in encoder.target_model.parameters())

    _run_epoch(
        model=model,
        transitions=rows,
        indices=[0, 1],
        cached_hidden=None,
        policy_encoder=encoder,
        optimizer=None,
        batch_size=2,
        device=torch.device("cpu"),
        seed=7,
        sigreg_coef=0.0,
        action_contrast_coef=0.0,
        alignment_coef=0.0,
        value_coef=0.0,
    )
    assert encoder.model.training is False
    assert encoder.target_model.training is False


def test_backprop_requires_persisting_updated_backbone():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--encoder",
            "hf-policy",
            "--hf-model",
            "model",
            "--backprop-to-llm",
        ]
    )
    with pytest.raises(ValueError, match="requires --save-updated-llm"):
        _validate_args(args)


def test_lora_and_feedback_auxiliary_require_their_training_contracts():
    parser = _build_parser()
    lora = parser.parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--encoder",
            "hf-policy",
            "--hf-model",
            "model",
            "--llm-train-mode",
            "lora",
        ]
    )
    with pytest.raises(ValueError, match="requires --backprop-to-llm"):
        _validate_args(lora)

    feedback = parser.parse_args(
        [
            "--input",
            "records.jsonl",
            "--output-dir",
            "out",
            "--feedback-aux-coef",
            "0.3",
        ]
    )
    with pytest.raises(ValueError, match="requires --prediction-target next_state"):
        _validate_args(feedback)


def test_value_label_contract_rejects_mixed_execution_semantics():
    first = replace(
        _transition("a"),
        reward=1.0,
        reward_label_source="execution.status",
        reward_label_semantics="success",
        reward_label_is_execution_outcome=True,
    )
    second = replace(
        _transition("b"),
        reward=0.0,
        reward_label_source="verifier.delta",
        reward_label_semantics="progress",
        reward_label_is_execution_outcome=True,
    )

    contract = _value_label_contract([first, second], [0, 1])

    assert contract["labeled_count"] == 2
    assert contract["consistent"] is False
    assert contract["verified_execution_outcome"] is False


def test_value_label_contract_requires_complete_execution_provenance():
    transition = replace(
        _transition("a"),
        reward=1.0,
        reward_label_is_execution_outcome=True,
    )

    contract = _value_label_contract([transition], [0])

    assert contract["consistent"] is True
    assert contract["verified_execution_outcome"] is False


def test_trainer_rejects_train_validation_value_contract_mismatch(tmp_path, monkeypatch):
    verified = replace(
        _transition("a"),
        reward=1.0,
        reward_label_scope="tool_execution",
        reward_label_source="execution.status",
        reward_label_semantics="execution_outcome",
        reward_label_is_execution_outcome=True,
    )
    composite = replace(
        _transition("b"),
        reward=0.0,
        reward_label_scope="per_turn",
        reward_label_source="reward.per_turn_scores.score",
        reward_label_semantics="training_step_score_unspecified",
        reward_label_is_execution_outcome=False,
    )
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(row.to_dict()) for row in (verified, composite)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(tmp_path / "out"),
            "--encoder",
            "hash",
            "--hash-hidden-dim",
            "8",
            "--latent-dim",
            "8",
            "--predictor-num-heads",
            "2",
            "--val-ratio",
            "0.5",
            "--split-group-key",
            "trajectory_id",
            "--value-coef",
            "0.1",
            "--allow-unverified-value-labels",
        ],
    )

    with pytest.raises(ValueError, match="contracts do not match"):
        main()


def test_hash_trainer_exports_verified_artifacts(tmp_path, monkeypatch):
    transitions = [
        replace(
            _transition("trajectory-a"),
            reward=1.0,
            reward_label_scope="tool_execution",
            reward_label_source="execution.status",
            reward_label_semantics="execution_outcome",
            reward_label_is_execution_outcome=True,
        ),
        replace(
            _transition("trajectory-b"),
            reward=0.0,
            reward_label_scope="tool_execution",
            reward_label_source="execution.status",
            reward_label_semantics="execution_outcome",
            reward_label_is_execution_outcome=True,
        ),
    ]
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(row.to_dict(), ensure_ascii=False) for row in transitions) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(output),
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
            "1",
            "--val-ratio",
            "0.5",
            "--split-group-key",
            "trajectory_id",
            "--value-coef",
            "0.1",
        ],
    )

    main()

    cache = torch.load(output / "hidden_cache.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(output / "latent_world_model.pt", map_location="cpu", weights_only=False)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert validate_hidden_cache_integrity(cache, require_verified=True)["verified"] is True
    assert checkpoint["config"]["architecture_version"] == "shared_latent_v2"
    assert checkpoint["metadata"]["split"]["strategy"] == "group_holdout"
    assert checkpoint["metadata"]["cache_metadata"] == cache["metadata"]
    assert checkpoint["metadata"]["value_update_step_count"] > 0
    assert checkpoint["metadata"]["value_head_parameter_delta_l2"] > 0.0
    assert checkpoint["metadata"]["train_value_label_contract"]["verified_execution_outcome"] is True
    assert summary["diagnostic_only"] is True
    assert (output / "predictions.jsonl").is_file()

    reused_output = tmp_path / "reused"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(reused_output),
            "--encoder",
            "hash",
            "--hash-hidden-dim",
            "8",
            "--hidden-cache-input",
            str(output / "hidden_cache.pt"),
            "--use-dapo-replay-buffer",
            "--replay-buffer-size",
            "2",
            "--latent-dim",
            "8",
            "--predictor-num-heads",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--val-ratio",
            "0.5",
            "--split-group-key",
            "trajectory_id",
        ],
    )

    main()

    reused_summary = json.loads((reused_output / "run_summary.json").read_text(encoding="utf-8"))
    reused_cache = torch.load(
        reused_output / "hidden_cache.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert reused_summary["hidden_cache_reused"] is True
    assert validate_hidden_cache_integrity(reused_cache, require_verified=True)["verified"] is True


def test_next_state_trainer_filters_terminal_rows_and_records_objective(tmp_path, monkeypatch):
    transitions = []
    for trajectory_id in ("trajectory-a", "trajectory-b"):
        transitions.extend(
            [
                replace(
                    _transition(trajectory_id, 0),
                    next_context_messages=[
                        {"role": "user", "content": f"state {trajectory_id}"},
                        {"role": "assistant", "content": f"act {trajectory_id}"},
                        {"role": "tool", "content": f"result {trajectory_id}"},
                    ],
                    done=False,
                ),
                _transition(trajectory_id, 1),
            ]
        )
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(row.to_dict(), ensure_ascii=False) for row in transitions) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "next-state"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(output),
            "--encoder",
            "hash",
            "--hash-hidden-dim",
            "8",
            "--latent-dim",
            "8",
            "--predictor-num-heads",
            "2",
            "--prediction-target",
            "next_state",
            "--pred-loss-type",
            "scaled_mse",
            "--alignment-coef",
            "0",
            "--action-contrast-coef",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--val-ratio",
            "0.5",
            "--split-group-key",
            "trajectory_id",
        ],
    )

    main()

    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    target_filter = summary["split"]["prediction_target_filter"]
    assert summary["model_config"]["prediction_target"] == "next_state"
    assert summary["hyperparameters"]["pred_loss_type"] == "scaled_mse"
    assert target_filter["pre_filter_train_count"] == 2
    assert target_filter["pre_filter_val_count"] == 2
    assert target_filter["train_count"] == 1
    assert target_filter["val_count"] == 1
    assert summary["final"]["train_metrics"]["wm/prediction_mask_count"] == 1.0
    checkpoint = torch.load(
        output / "latent_world_model.pt", map_location="cpu", weights_only=False
    )
    cache = torch.load(output / "hidden_cache.pt", map_location="cpu", weights_only=False)
    val_indices, scope = select_evaluation_indices(
        checkpoint["metadata"],
        cache["metadata"],
        count=4,
        requested_split="val",
    )
    assert len(val_indices) == 2
    assert scope["scope"] == "group_heldout"
    assert set(summary["split"]["train_indices"]) | set(summary["split"]["val_indices"]) == set(
        range(4)
    )


def test_next_state_supervision_rejects_cache_record_mask_mismatch():
    transitions = [
        replace(
            _transition("a"),
            next_context_messages=[{"role": "tool", "content": "result"}],
            done=False,
        ),
        _transition("b"),
    ]
    hidden = hash_hidden_batch(transitions, hidden_dim=8)
    hidden["has_next"] = torch.tensor([False, False])

    with pytest.raises(ValueError, match="does not match"):
        _validate_next_state_supervision(hidden, transitions)


def test_checkpoint_saves_final_epoch_while_best_metric_is_informational(
    tmp_path, monkeypatch
):
    transitions = [_transition("trajectory-a"), _transition("trajectory-b")]
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(row.to_dict()) for row in transitions) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "final-epoch"
    calls = 0
    final_state = {}

    def fake_run_epoch(*, model, optimizer, **_kwargs):
        nonlocal calls, final_state
        assert optimizer is not None
        calls += 1
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        final_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        return (0.1 if calls == 1 else 0.2), {}, 1, 0

    monkeypatch.setattr(train_latent_module, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(output),
            "--encoder",
            "hash",
            "--hash-hidden-dim",
            "8",
            "--latent-dim",
            "8",
            "--predictor-num-heads",
            "2",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--val-ratio",
            "0",
        ],
    )

    train_latent_module.main()

    checkpoint = torch.load(
        output / "latent_world_model.pt", map_location="cpu", weights_only=False
    )
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert calls == 2
    assert checkpoint["metadata"]["checkpoint_selection"] == "final_epoch"
    assert checkpoint["metadata"]["best_epoch"] == 1
    assert checkpoint["metadata"]["final_train_loss"] == pytest.approx(0.2)
    assert summary["checkpoint_selection"] == "final_epoch"
    assert summary["final"]["epoch"] == 2
    assert summary["best_epoch"] == 1
    assert checkpoint["state_dict"].keys() == final_state.keys()
    for key, value in checkpoint["state_dict"].items():
        assert torch.equal(value, final_state[key])


def test_checkpoint_can_restore_best_validation_state(tmp_path, monkeypatch):
    transitions = [_transition("trajectory-a"), _transition("trajectory-b")]
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(row.to_dict()) for row in transitions) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "best-validation"
    calls = 0
    states = []

    def fake_run_epoch(*, model, optimizer, **_kwargs):
        nonlocal calls
        assert optimizer is not None
        calls += 1
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return (0.1 if calls == 1 else 0.2), {}, 1, 0

    monkeypatch.setattr(train_latent_module, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_latent",
            "--input",
            str(records),
            "--output-dir",
            str(output),
            "--encoder",
            "hash",
            "--hash-hidden-dim",
            "8",
            "--latent-dim",
            "8",
            "--predictor-num-heads",
            "2",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--val-ratio",
            "0",
            "--checkpoint-selection",
            "best_validation",
        ],
    )

    train_latent_module.main()

    checkpoint = torch.load(
        output / "latent_world_model.pt", map_location="cpu", weights_only=False
    )
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert calls == 2
    assert checkpoint["metadata"]["checkpoint_selection"] == "best_validation"
    assert checkpoint["metadata"]["selected_epoch"] == 1
    assert summary["selected"]["epoch"] == 1
    for key, value in checkpoint["state_dict"].items():
        assert torch.equal(value, states[0][key])


def test_trainable_backbone_snapshot_restores_only_trainable_parameters():
    module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    module[1].requires_grad_(False)
    snapshot = train_latent_module._clone_trainable_state_dict_cpu(module)
    frozen_after_change = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.add_(3.0)
            if not parameter.requires_grad:
                frozen_after_change[name] = parameter.detach().clone()

    train_latent_module._restore_trainable_state_dict(module, snapshot)

    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, snapshot[name])
        else:
            assert torch.equal(parameter, frozen_after_change[name])
