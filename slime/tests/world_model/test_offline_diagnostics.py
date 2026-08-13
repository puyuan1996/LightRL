from __future__ import annotations

import pytest
import torch

from slime.world_model.offline_diagnostics import (
    _atomic_partition,
    _classification_metrics,
    _collapse_gate_from_control,
    _inferred_final_diagnostics,
    _matched_action_derangement,
    _next_state_partition,
    _parse_args,
    _prediction_metrics,
    _relative_collapse_gate,
    _retrieval_metrics,
    _tool_partition,
    _validate_alignment,
)


def test_retrieval_uses_multi_positive_feedback_equivalence() -> None:
    similarities = torch.tensor(
        [
            [0.8, 0.9, 0.1],
            [0.9, 0.8, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )
    metrics = _retrieval_metrics(similarities, ["same", "same", "unique"])
    assert metrics["top1_accuracy"] == pytest.approx(1.0)
    assert metrics["duplicate_group_count"] == 1
    assert metrics["queries_with_equivalent_targets"] == 2
    assert metrics["baselines"]["random_multi_positive_top1"] == pytest.approx(5 / 9)


def test_control_checkpoint_is_optional_for_single_run_diagnostics() -> None:
    args = _parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--cache",
            "cache.pt",
            "--records",
            "records.jsonl",
            "--output",
            "diagnostics.json",
        ]
    )
    assert args.control_checkpoint is None


def test_next_state_control_helper_reports_unevaluated_without_control() -> None:
    result = _collapse_gate_from_control(
        control_checkpoint_path=None,
        candidate_diagnostics={
            "pred": {"variance_mean": 1.0, "effective_rank": 2.0, "pairwise_cosine": 0.0},
            "target": {"variance_mean": 1.0, "effective_rank": 2.0, "pairwise_cosine": 0.0},
        },
        cache_metadata={},
        val_indices=[0],
        record_count=1,
        min_retention=0.5,
    )
    assert result["evaluated"] is False
    assert result["passed"] is None


def test_relative_collapse_gate_can_reject_variance_expansion() -> None:
    control = {
        "pred": {"variance_mean": 1.0, "effective_rank": 4.0, "pairwise_cosine": 0.0},
        "target": {"variance_mean": 1.0, "effective_rank": 4.0, "pairwise_cosine": 0.0},
    }
    candidate = {
        "pred": {"variance_mean": 6.0, "effective_rank": 4.0, "pairwise_cosine": 0.0},
        "target": {"variance_mean": 1.0, "effective_rank": 4.0, "pairwise_cosine": 0.0},
    }

    result = _relative_collapse_gate(candidate, control, 0.5, 2.0)

    assert result["lower_bound_passed"] is True
    assert result["upper_bound_passed"] is False
    assert result["passed"] is False


def test_inferred_final_diagnostics_support_checkpoint_without_saved_distribution_metrics() -> None:
    latent = {
        "pred_latent": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
        ),
        "target_latent": torch.zeros(3, 2),
        "next_state_target_latent": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
        ),
    }

    diagnostics = _inferred_final_diagnostics(
        latent, [0, 1, 2], prediction_target="next_state"
    )

    assert diagnostics["pred"]["variance_mean"] > 0.0
    assert diagnostics["pred"]["effective_rank"] > 1.0
    assert diagnostics["target"] == pytest.approx(diagnostics["pred"])


def test_prediction_metrics_include_paired_and_retrieval_scores() -> None:
    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = prediction.clone()
    metrics = _prediction_metrics(prediction, target, ["a", "b"], torch.device("cpu"))
    assert metrics["paired_mse"] == pytest.approx(0.0)
    assert metrics["paired_cosine_similarity"] == pytest.approx(1.0)
    assert metrics["paired_cosine_distance"] == pytest.approx(0.0)
    assert metrics["top1_accuracy"] == pytest.approx(1.0)


def test_next_state_partition_checks_record_cache_alignment() -> None:
    records = [
        {"has_next": True, "next_context_hash": "next-a"},
        {"has_next": False, "next_context_hash": None},
        {"has_next": True, "next_context_hash": "next-c"},
    ]
    cache = {
        "next_state_hidden": torch.zeros(3, 4),
        "has_next": torch.tensor([True, False, True]),
    }
    assert _next_state_partition(records, cache, [2, 1, 0]) == (
        [0, 2],
        [2, 0],
        ["next-c", "next-a"],
    )
    cache["has_next"] = torch.tensor([False, False, True])
    with pytest.raises(ValueError, match="has_next mismatch"):
        _next_state_partition(records, cache, [0])


def test_classification_metrics_report_imbalance_robust_scores() -> None:
    metrics = _classification_metrics([0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 2])
    assert metrics["top1_accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(3 / 7)


def test_atomic_partition_rejects_multi_tool_records() -> None:
    records = [
        {"tool_names": ["shell_exec"]},
        {"tool_names": ["shell_exec", "shell_view"]},
        {"tool_names": []},
        {},
    ]
    assert _atomic_partition(records, list(range(4))) == ([0], ["shell_exec"])
    assert _tool_partition(records, list(range(4)), mode="first_tool") == (
        [0, 1],
        ["shell_exec", "shell_exec"],
    )


def test_relative_collapse_gate_detects_low_variance_ema_target() -> None:
    control = {
        "pred": {"variance_mean": 0.007, "effective_rank": 46.0, "pairwise_cosine": 0.02},
        "target": {"variance_mean": 0.0077, "effective_rank": 59.0, "pairwise_cosine": 0.01},
    }
    candidate = {
        "pred": {"variance_mean": 0.0006, "effective_rank": 65.0, "pairwise_cosine": 0.92},
        "target": {"variance_mean": 0.0016, "effective_rank": 34.0, "pairwise_cosine": 0.79},
    }
    result = _relative_collapse_gate(candidate, control, 0.5)
    assert result["passed"] is False
    assert result["ratios"]["target_variance_mean_retention"] < 0.5


def test_alignment_fails_closed_on_transition_reordering() -> None:
    records = [{"transition_id": "a"}, {"transition_id": "b"}]
    cache = {
        "record_count": 2,
        "record_metadata": [{"transition_id": "b"}, {"transition_id": "a"}],
        "metadata": {"input_records_sha256": "digest"},
        "state_hidden": torch.zeros(2, 4),
        "action_hidden": torch.zeros(2, 4),
        "target_hidden": torch.zeros(2, 4),
    }
    with pytest.raises(ValueError, match="transition order"):
        _validate_alignment(records, cache, "digest")


def test_matched_action_derangement_preserves_tool_and_length_strata() -> None:
    records = [
        {
            "tool_names": ["shell_exec"],
            "action_text": "a" * 20,
            "next_observation_text": "r" * 20,
        },
        {
            "tool_names": ["shell_exec"],
            "action_text": "b" * 30,
            "next_observation_text": "s" * 30,
        },
        {
            "tool_names": ["shell_view"],
            "action_text": "c" * 20,
            "next_observation_text": "t" * 20,
        },
    ]
    kept, shuffled, audit = _matched_action_derangement(records, [0, 1, 2])

    assert kept == [0, 1]
    assert shuffled == [1, 0]
    assert audit["coverage"] == pytest.approx(2 / 3)
    assert audit["same_record_count"] == 0
    _matched_action_derangement,
