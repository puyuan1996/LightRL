"""Strict offline diagnostics for cached text-JEPA world-model representations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .cache_text_hidden import validate_hidden_cache_integrity
from .checkpoint import select_evaluation_indices, validate_cache_encoder
from .metrics import effective_rank
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required_fields = (
        "transition_id",
        "tool_names",
        "next_observation_hash",
        "next_observation_text",
        "next_context_hash",
        "has_next",
        "action_text",
    )
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"record line {line_number} is not an object")
            # Context text stays in the verified cache; canonical action text
            # is retained only to construct length-matched shuffle controls.
            rows.append({key: row.get(key) for key in required_fields})
    if not rows:
        raise ValueError("records JSONL is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_alignment(
    records: Sequence[Mapping[str, Any]], cache: Mapping[str, Any], records_sha256: str
) -> None:
    metadata = cache.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("hidden cache metadata is missing")
    if metadata.get("input_records_sha256") != records_sha256:
        raise ValueError("records digest does not match hidden cache")
    count = len(records)
    if int(cache.get("record_count", -1)) != count:
        raise ValueError("records/cache count mismatch")
    cached_records = cache.get("record_metadata")
    if not isinstance(cached_records, list) or len(cached_records) != count:
        raise ValueError("cache record_metadata is missing or misaligned")
    record_ids = [str(row.get("transition_id")) for row in records]
    cached_ids = [str(row.get("transition_id")) for row in cached_records]
    if record_ids != cached_ids or any(value == "None" for value in record_ids):
        raise ValueError("records/cache transition order mismatch")
    required = ("state_hidden", "action_hidden", "target_hidden")
    for key in required:
        value = cache.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[0] != count:
            raise ValueError(f"cache tensor {key!r} is missing or misaligned")


def _classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]
) -> dict[str, float]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("classification inputs must be non-empty and aligned")
    observed = [label for label in labels if label in set(y_true)]
    if not observed:
        raise ValueError("classification labels do not cover validation targets")
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    f1_values: list[float] = []
    recalls: list[float] = []
    for label in observed:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        recalls.append(recall)
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "top1_accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(recalls) / len(recalls),
    }


def _tool_partition(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int], *, mode: str
) -> tuple[list[int], list[str]]:
    if mode not in {"single_call", "first_tool"}:
        raise ValueError(f"unknown tool-label mode: {mode}")
    kept: list[int] = []
    labels: list[str] = []
    for index in indices:
        names = records[index].get("tool_names")
        valid = [str(name).strip() for name in names] if isinstance(names, list) else []
        valid = [name for name in valid if name]
        if valid and (mode == "first_tool" or len(valid) == 1):
            kept.append(index)
            labels.append(valid[0])
    return kept, labels


def _atomic_partition(
    records: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> tuple[list[int], list[str]]:
    """Backward-compatible alias for the original single-call diagnostic."""
    return _tool_partition(records, indices, mode="single_call")


def _linear_probe(
    features: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    seeds: Sequence[int],
    epochs: int,
    device: torch.device,
    *,
    label_mode: str,
) -> dict[str, Any]:
    train, train_names = _tool_partition(records, train_indices, mode=label_mode)
    val, val_names = _tool_partition(records, val_indices, mode=label_mode)
    classes = sorted(set(train_names))
    if len(classes) < 2 or not val:
        raise ValueError("tool probe requires at least two train classes and non-empty validation")
    unseen = sorted(set(val_names) - set(classes))
    if unseen:
        raise ValueError(f"validation contains unseen atomic tool classes: {unseen}")
    class_id = {name: index for index, name in enumerate(classes)}
    x_train = features[torch.tensor(train)].flatten(1).float()
    x_val = features[torch.tensor(val)].flatten(1).float()
    mean = x_train.mean(0, keepdim=True)
    scale = x_train.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_train = ((x_train - mean) / scale).to(device)
    x_val = ((x_val - mean) / scale).to(device)
    y_train = torch.tensor([class_id[name] for name in train_names], device=device)
    y_val = [class_id[name] for name in val_names]
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        probe = torch.nn.Linear(x_train.shape[1], len(classes)).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(x_train), y_train)
            loss.backward()
            optimizer.step()
        prediction = probe(x_val).argmax(-1).detach().cpu().tolist()
        confusion = [
            [
                sum(a == row and b == column for a, b in zip(y_val, prediction))
                for column in range(len(classes))
            ]
            for row in range(len(classes))
        ]
        runs.append(
            {
                "seed": seed,
                "final_train_ce": float(loss.detach().cpu()),
                "confusion_matrix": confusion,
                **_classification_metrics(y_val, prediction, range(len(classes))),
            }
        )
    metric_names = ("top1_accuracy", "macro_f1", "balanced_accuracy")
    means = {key: sum(row[key] for row in runs) / len(runs) for key in metric_names}
    stds = {
        key: math.sqrt(sum((row[key] - means[key]) ** 2 for row in runs) / len(runs))
        for key in metric_names
    }
    train_counts = Counter(train_names)
    val_counts = Counter(val_names)
    majority_name = train_counts.most_common(1)[0][0]
    majority_prediction = [class_id[majority_name]] * len(y_val)
    return {
        "scope": f"task_group_heldout_{label_mode}",
        "label_semantics": (
            "first executable tool in the assistant action"
            if label_mode == "first_tool"
            else "assistant action containing exactly one tool call"
        ),
        "train_count": len(train),
        "val_count": len(val),
        "classes": classes,
        "train_class_counts": dict(sorted(train_counts.items())),
        "val_class_counts": dict(sorted(val_counts.items())),
        "majority_baseline": _classification_metrics(
            y_val, majority_prediction, range(len(classes))
        ),
        "runs": runs,
        "mean": means,
        "std": stds,
    }


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T


def _retrieval_on_device(
    left: torch.Tensor, right: torch.Tensor, keys: Sequence[str], device: torch.device
) -> dict[str, Any]:
    return _retrieval_metrics(_cosine_matrix(left.to(device), right.to(device)), keys)


def _prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    keys: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must be aligned rank-2 tensors")
    if prediction.shape[0] != len(keys) or not keys:
        raise ValueError("prediction metrics require one non-empty key per row")
    prediction_float = prediction.float()
    target_float = target.float()
    paired_cosine = F.cosine_similarity(prediction_float, target_float, dim=-1)
    return {
        "paired_mse": float(F.mse_loss(prediction_float, target_float)),
        "paired_cosine_similarity": float(paired_cosine.mean()),
        "paired_cosine_distance": float((1.0 - paired_cosine).mean()),
        **_retrieval_on_device(prediction, target, keys, device),
    }


def _retrieval_metrics(similarities: torch.Tensor, keys: Sequence[str]) -> dict[str, Any]:
    if similarities.ndim != 2 or similarities.shape != (len(keys), len(keys)) or not keys:
        raise ValueError("retrieval similarities and keys must form a non-empty square problem")
    order = similarities.detach().float().cpu().argsort(dim=1, descending=True)
    top1 = 0.0
    reciprocal_rank = 0.0
    recall5 = 0.0
    for query, key in enumerate(keys):
        positives = {index for index, candidate_key in enumerate(keys) if candidate_key == key}
        ranked = order[query].tolist()
        positive_ranks = [rank for rank, index in enumerate(ranked, 1) if index in positives]
        top1 += float(ranked[0] in positives)
        reciprocal_rank += 1.0 / min(positive_ranks)
        recall5 += len(positives.intersection(ranked[:5])) / len(positives)
    count = len(keys)
    key_counts = Counter(keys)
    random_top1 = sum(value * value for value in key_counts.values()) / (count * count)
    constant_top1 = max(key_counts.values()) / count
    return {
        "query_count": count,
        "candidate_count": count,
        "unique_target_count": len(key_counts),
        "duplicate_group_count": sum(value > 1 for value in key_counts.values()),
        "queries_with_equivalent_targets": sum(value for value in key_counts.values() if value > 1),
        "top1_accuracy": top1 / count,
        "mrr": reciprocal_rank / count,
        "recall_at_5": recall5 / count,
        "baselines": {
            "random_multi_positive_top1": random_top1,
            "most_frequent_target_top1": constant_top1,
            "random_expected_recall_at_5": min(5, count) / count,
        },
    }


def _feedback_key(record: Mapping[str, Any]) -> str:
    value = record.get("next_observation_hash")
    if value:
        return str(value)
    text = record.get("next_observation_text")
    if not isinstance(text, str) or not text:
        raise ValueError("T2 retrieval requires next_observation_hash or next_observation_text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _next_state_partition(
    records: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    indices: Sequence[int],
) -> tuple[list[int], list[int], list[str]]:
    next_state_hidden = cache.get("next_state_hidden")
    has_next = cache.get("has_next")
    count = len(records)
    if (
        not isinstance(next_state_hidden, torch.Tensor)
        or next_state_hidden.ndim < 2
        or next_state_hidden.shape[0] != count
    ):
        raise ValueError("next-state diagnostics require an aligned next_state_hidden tensor")
    if (
        not isinstance(has_next, torch.Tensor)
        or has_next.ndim == 0
        or has_next.shape[0] != count
    ):
        raise ValueError("next-state diagnostics require an aligned has_next tensor")

    record_metadata = cache.get("record_metadata")
    encoder_config = cache.get("metadata", {}).get("encoder_config", {})
    belief_view = encoder_config.get("state_view") == "belief_view_v1"
    if not isinstance(record_metadata, list) or len(record_metadata) != count:
        if belief_view:
            raise ValueError("belief-view diagnostics require aligned cache record_metadata")
        record_metadata = list(records)
    positions: list[int] = []
    record_indices: list[int] = []
    keys: list[str] = []
    for position, index in enumerate(indices):
        record_has_next = bool(records[index].get("has_next"))
        cache_has_next = bool(has_next[index].item())
        if record_has_next != cache_has_next:
            raise ValueError("records/cache has_next mismatch")
        if not record_has_next:
            continue
        key = record_metadata[index].get("next_state_view_hash")
        if not key:
            key = records[index].get("next_context_hash")
        if not key:
            raise ValueError("has-next record is missing next_context_hash")
        positions.append(position)
        record_indices.append(index)
        keys.append(str(key))
    if not positions:
        raise ValueError("validation split has no next-state supervision")
    return positions, record_indices, keys


def _infer_latents(
    model: TextLatentWorldModel,
    cache: Mapping[str, Any],
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    *,
    shuffle_actions: bool = False,
    zero_actions: bool = False,
    action_indices: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    if sum((shuffle_actions, zero_actions, action_indices is not None)) > 1:
        raise ValueError("action controls are mutually exclusive")
    output_rows: dict[str, list[torch.Tensor]] = {
        "state_latent": [],
        "pred_latent": [],
        "target_latent": [],
        "next_state_latent": [],
        "next_state_target_latent": [],
    }
    next_state_hidden = cache.get("next_state_hidden")
    if not isinstance(next_state_hidden, torch.Tensor):
        raise ValueError("latent inference requires next_state_hidden")
    selected_action_indices = (
        list(action_indices) if action_indices is not None else list(indices)
    )
    if len(selected_action_indices) != len(indices):
        raise ValueError("action_indices must align with indices")
    if shuffle_actions:
        if len(selected_action_indices) < 2:
            raise ValueError("shuffled-action control requires at least two records")
        generator = torch.Generator().manual_seed(20260723)
        shift = int(
            torch.randint(
                1, len(selected_action_indices), (1,), generator=generator
            ).item()
        )
        selected_action_indices = (
            selected_action_indices[shift:] + selected_action_indices[:shift]
        )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = list(indices[start : start + batch_size])
            action_batch = selected_action_indices[start : start + batch_size]
            action_hidden = cache["action_hidden"][action_batch].to(device)
            if zero_actions:
                action_hidden = torch.zeros_like(action_hidden)
            rows = model(
                state_hidden=cache["state_hidden"][batch].to(device),
                action_hidden=action_hidden,
                target_hidden=cache["target_hidden"][batch].to(device),
                next_state_hidden=next_state_hidden[batch].to(device),
            )
            for key in output_rows:
                value = rows[key]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"model did not return {key}")
                output_rows[key].append(value.detach().float().cpu())
    return {key: torch.cat(values) for key, values in output_rows.items()}


def _length_bin(length: int) -> str:
    for boundary in (128, 512, 2048, 8192):
        if length <= boundary:
            return f"le_{boundary}"
    return "gt_8192"


def _call_count_bin(count: int) -> str:
    if count <= 4:
        return str(count)
    if count <= 8:
        return "5_8"
    if count <= 16:
        return "9_16"
    return "gt_16"


def _matched_action_derangement(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Derange actions within tool/call-count/length strata, with explicit fallback."""

    levels = (
        ("first_tool", "call_count", "action_length", "result_length"),
        ("first_tool", "call_count", "action_length"),
        ("first_tool", "call_count"),
        ("first_tool",),
    )
    remaining = list(indices)
    assignments: dict[int, int] = {}
    level_counts: Counter[str] = Counter()
    for level in levels:
        groups: dict[tuple[str, ...], list[int]] = {}
        for index in remaining:
            names = records[index].get("tool_names")
            names = names if isinstance(names, list) else []
            first_tool = str(names[0]) if names else ""
            values = {
                "first_tool": first_tool,
                "call_count": _call_count_bin(len(names)),
                "action_length": _length_bin(
                    len(str(records[index].get("action_text") or ""))
                ),
                "result_length": _length_bin(
                    len(str(records[index].get("next_observation_text") or ""))
                ),
            }
            groups.setdefault(tuple(values[key] for key in level), []).append(index)
        matched_now: set[int] = set()
        for group in groups.values():
            if len(group) < 2:
                continue
            ordered = sorted(group)
            rotated = ordered[1:] + ordered[:1]
            assignments.update(zip(ordered, rotated, strict=True))
            matched_now.update(ordered)
            level_counts["+".join(level)] += len(ordered)
        remaining = [index for index in remaining if index not in matched_now]
    kept = [index for index in indices if index in assignments]
    shuffled = [assignments[index] for index in kept]
    return kept, shuffled, {
        "requested_count": len(indices),
        "matched_count": len(kept),
        "coverage": len(kept) / max(len(indices), 1),
        "unmatched_count": len(indices) - len(kept),
        "matching_levels": dict(level_counts),
        "same_record_count": sum(a == b for a, b in zip(kept, shuffled)),
        "control_semantics": (
            "deterministic derangement within first-tool, call-count, "
            "action-length, and result-length strata with registered fallback"
        ),
    }


def _build_next_state_diagnostics(
    *,
    records: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    val_indices: Sequence[int],
    val_latent: Mapping[str, torch.Tensor],
    shuffled_latent: Mapping[str, torch.Tensor],
    zero_latent: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    next_positions, next_record_indices, next_keys = _next_state_partition(
        records, cache, val_indices
    )
    next_position_tensor = torch.tensor(next_positions)
    next_record_tensor = torch.tensor(next_record_indices)
    next_state_latent = val_latent["next_state_target_latent"][next_position_tensor]
    encoder_config = cache.get("metadata", {}).get("encoder_config", {})
    state_view = encoder_config.get("state_view", "full_context_v1")
    return {
        "scope": "observational_task_group_heldout_has_next_context",
        "state_view": state_view,
        "target_semantics": (
            "Qwen hidden at the belief_view_v1 dynamic STATE_VIEW suffix"
            if state_view == "belief_view_v1"
            else "Qwen hidden of the next full agent context after appending the observed tool feedback"
        ),
        "query_count": len(next_positions),
        "jepa_pred_to_next_state": _prediction_metrics(
            val_latent["pred_latent"][next_position_tensor],
            next_state_latent,
            next_keys,
            device,
        ),
        "state_identity_to_next_state": _prediction_metrics(
            val_latent["state_latent"][next_position_tensor],
            next_state_latent,
            next_keys,
            device,
        ),
        "feedback_target_to_next_state": _prediction_metrics(
            val_latent["target_latent"][next_position_tensor],
            next_state_latent,
            next_keys,
            device,
        ),
        "shuffled_action_jepa_pred_to_next_state": _prediction_metrics(
            shuffled_latent["pred_latent"][next_position_tensor],
            next_state_latent,
            next_keys,
            device,
        ),
        "zero_action_jepa_pred_to_next_state": _prediction_metrics(
            zero_latent["pred_latent"][next_position_tensor],
            next_state_latent,
            next_keys,
            device,
        ),
        "raw_state_to_raw_next_state": _prediction_metrics(
            cache["state_hidden"][next_record_tensor],
            cache["next_state_hidden"][next_record_tensor],
            next_keys,
            device,
        ),
    }


def _build_tool_probe_diagnostics(
    *,
    cache: Mapping[str, Any],
    latent: Mapping[str, torch.Tensor],
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    seeds: Sequence[int],
    epochs: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    atomic = {
        "raw_qwen_state_hidden": _linear_probe(
            cache["state_hidden"], records, train_indices, val_indices, seeds, epochs,
            device, label_mode="single_call"
        ),
        "jepa_state_latent": _linear_probe(
            latent["state_latent"], records, train_indices, val_indices, seeds, epochs,
            device, label_mode="single_call"
        ),
    }
    first = {
        "raw_qwen_state_hidden": _linear_probe(
            cache["state_hidden"], records, train_indices, val_indices, seeds, epochs,
            device, label_mode="first_tool"
        ),
        "jepa_state_latent": _linear_probe(
            latent["state_latent"], records, train_indices, val_indices, seeds, epochs,
            device, label_mode="first_tool"
        ),
    }
    return first, atomic


def _final_diagnostics(artifact: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    final = artifact.get("final")
    if not isinstance(final, Mapping):
        history = artifact.get("history")
        final = history[-1] if isinstance(history, list) and history else None
    metrics = final.get("val_metrics") if isinstance(final, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ValueError("run summary lacks final validation metrics")
    result: dict[str, dict[str, float]] = {}
    for side in ("pred", "target"):
        result[side] = {}
        for metric in ("variance_mean", "effective_rank", "pairwise_cosine"):
            value = float(metrics[f"wm/{side}_{metric}"])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {side} {metric}")
            result[side][metric] = value
    return result


def _distribution_summary(value: torch.Tensor) -> dict[str, float]:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("collapse diagnostics require at least two rank-2 latent rows")
    value = value.detach().float().cpu()
    normalized = F.normalize(value, dim=-1)
    count = value.shape[0]
    pairwise_cosine = (
        normalized.sum(dim=0).square().sum() - float(count)
    ) / float(count * (count - 1))
    result = {
        "variance_mean": float(value.var(dim=0, unbiased=False).mean()),
        "effective_rank": float(effective_rank(value)),
        "pairwise_cosine": float(pairwise_cosine),
    }
    if not all(math.isfinite(metric) for metric in result.values()):
        raise ValueError("non-finite inferred latent distribution diagnostic")
    return result


def _inferred_final_diagnostics(
    latent: Mapping[str, torch.Tensor],
    val_indices: Sequence[int],
    *,
    prediction_target: str,
) -> dict[str, dict[str, float]]:
    if not val_indices:
        raise ValueError("inferred collapse diagnostics require a non-empty validation split")
    indices = torch.tensor(list(val_indices), dtype=torch.long)
    target_key = (
        "next_state_target_latent" if prediction_target == "next_state" else "target_latent"
    )
    try:
        prediction = latent["pred_latent"][indices]
        target = latent[target_key][indices]
    except (KeyError, IndexError) as error:
        raise ValueError("inferred latent outputs are missing or misaligned") from error
    return {
        "pred": _distribution_summary(prediction),
        "target": _distribution_summary(target),
    }


def _relative_collapse_gate(
    candidate: Mapping[str, Mapping[str, float]],
    control: Mapping[str, Mapping[str, float]],
    min_retention: float,
    max_retention: float | None = None,
) -> dict[str, Any]:
    if not 0.0 < min_retention <= 1.0:
        raise ValueError("collapse retention threshold must be in (0, 1]")
    if max_retention is not None and max_retention < 1.0:
        raise ValueError("maximum collapse retention must be at least 1")
    ratios: dict[str, float] = {}
    for side in ("pred", "target"):
        if control[side]["effective_rank"] <= 3.0:
            raise ValueError(f"control {side} effective rank must exceed 3")
        for metric in ("variance_mean", "effective_rank"):
            denominator = control[side][metric]
            if denominator <= 0.0:
                raise ValueError(f"control {side} {metric} must be positive")
            ratios[f"{side}_{metric}_retention"] = candidate[side][metric] / denominator
        control_isotropy = 1.0 - abs(control[side]["pairwise_cosine"])
        candidate_isotropy = 1.0 - abs(candidate[side]["pairwise_cosine"])
        if control_isotropy <= 0.0:
            raise ValueError(f"control {side} isotropy must be positive")
        ratios[f"{side}_isotropy_retention"] = candidate_isotropy / control_isotropy
    finite = all(math.isfinite(value) for value in ratios.values())
    lower_passed = all(value >= min_retention for value in ratios.values())
    upper_passed = max_retention is None or all(
        value <= max_retention for value in ratios.values()
    )
    return {
        "passed": finite and lower_passed and upper_passed,
        "min_retention": min_retention,
        "max_retention": max_retention,
        "lower_bound_passed": lower_passed,
        "upper_bound_passed": upper_passed,
        "ratios": ratios,
    }


def _collapse_gate_from_control(
    *,
    control_checkpoint_path: Path | None,
    candidate_diagnostics: Mapping[str, Mapping[str, float]],
    cache_metadata: Mapping[str, Any],
    val_indices: Sequence[int],
    record_count: int,
    min_retention: float,
    max_retention: float | None = None,
) -> dict[str, Any]:
    collapse: dict[str, Any] = {
        "evaluated": False,
        "passed": None,
        "reason": "no same-cache control checkpoint was supplied",
        "candidate": candidate_diagnostics,
    }
    if control_checkpoint_path is None:
        return collapse
    control_checkpoint = torch.load(
        control_checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(control_checkpoint, dict):
        raise TypeError("control checkpoint must be a dictionary")
    control_metadata = control_checkpoint.get("metadata")
    validate_cache_encoder(control_metadata, cache_metadata)
    control_val_indices, _ = select_evaluation_indices(
        control_metadata,
        cache_metadata,
        count=record_count,
        requested_split="val",
    )
    if control_val_indices != list(val_indices):
        raise ValueError("candidate/control checkpoints do not use the same validation split")
    return {
        "evaluated": True,
        **_relative_collapse_gate(
            candidate_diagnostics,
            _final_diagnostics(control_checkpoint),
            min_retention,
            max_retention,
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        help=(
            "Optional same-cache reference checkpoint for a relative collapse gate. "
            "When omitted, absolute latent diagnostics are reported without inventing a control."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-seeds", default="11,13,17")
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--collapse-min-retention", type=float, default=0.5)
    parser.add_argument("--collapse-max-retention", type=float)
    parser.add_argument(
        "--next-state-only",
        action="store_true",
        help="Skip tool probes and feedback retrieval when only next-context diagnostics are needed.",
    )
    parser.add_argument(
        "--include-tool-probes",
        action="store_true",
        help="With --next-state-only, also run state-representation first-tool and atomic probes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.batch_size <= 0 or args.probe_epochs <= 0:
        raise ValueError("batch size and probe epochs must be positive")
    if args.include_tool_probes and not args.next_state_only:
        raise ValueError("--include-tool-probes is only needed with --next-state-only")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    records = _read_jsonl(args.records)
    records_sha256 = _sha256(args.records)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(cache, dict):
        raise TypeError("hidden cache must be a dictionary")
    validate_hidden_cache_integrity(cache, require_verified=True)
    _validate_alignment(records, cache, records_sha256)
    cache_metadata = cache["metadata"]
    # Integrity and alignment are established. Keep lightweight record metadata
    # because belief-view retrieval keys are cache-bound provenance.
    cache.pop("reward", None)
    cache.pop("reward_mask", None)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    checkpoint_metadata = checkpoint.get("metadata")
    validate_cache_encoder(checkpoint_metadata, cache_metadata)
    train_indices, train_scope = select_evaluation_indices(
        checkpoint_metadata, cache_metadata, count=len(records), requested_split="train"
    )
    val_indices, val_scope = select_evaluation_indices(
        checkpoint_metadata, cache_metadata, count=len(records), requested_split="val"
    )
    if not val_scope.get("group_disjoint"):
        raise ValueError("offline diagnostics require a group-disjoint validation split")
    config = TextLatentWorldModelConfig(**checkpoint["config"])
    if config.prediction_target == "next_state" and not args.next_state_only:
        raise ValueError(
            "next-state checkpoints require --next-state-only; feedback retrieval "
            "metrics are not valid for a next-state predictor"
        )
    model = TextLatentWorldModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    del checkpoint

    all_indices = list(range(len(records)))
    latent = _infer_latents(model, cache, all_indices, device, args.batch_size)
    candidate_diagnostics = _inferred_final_diagnostics(
        latent,
        val_indices,
        prediction_target=config.prediction_target,
    )
    collapse = _collapse_gate_from_control(
        control_checkpoint_path=args.control_checkpoint,
        candidate_diagnostics=candidate_diagnostics,
        cache_metadata=cache_metadata,
        val_indices=val_indices,
        record_count=len(records),
        min_retention=args.collapse_min_retention,
        max_retention=args.collapse_max_retention,
    )
    if args.next_state_only:
        seeds = [int(value.strip()) for value in args.probe_seeds.split(",") if value.strip()]
        if args.include_tool_probes and not seeds:
            raise ValueError("at least one probe seed is required")
        val_tensor = torch.tensor(val_indices)
        val_latent = {key: value[val_tensor] for key, value in latent.items()}
        shuffled = _infer_latents(
            model, cache, val_indices, device, args.batch_size, shuffle_actions=True
        )
        zero = _infer_latents(
            model, cache, val_indices, device, args.batch_size, zero_actions=True
        )
        matched_indices, matched_action_indices, matched_audit = (
            _matched_action_derangement(records, val_indices)
        )
        if len(matched_indices) < 2:
            raise ValueError("matched shuffled-action control has fewer than two rows")
        matched = _infer_latents(
            model,
            cache,
            matched_indices,
            device,
            args.batch_size,
            action_indices=matched_action_indices,
        )
        matched_tensor = torch.tensor(matched_indices)
        matched_observed = {
            key: value[matched_tensor] for key, value in latent.items()
        }
        matched_positions, _, matched_keys = _next_state_partition(
            records, cache, matched_indices
        )
        matched_position_tensor = torch.tensor(matched_positions)
        matched_target = matched_observed["next_state_target_latent"][
            matched_position_tensor
        ]
        next_state_diagnostics = _build_next_state_diagnostics(
            records=records,
            cache=cache,
            val_indices=val_indices,
            val_latent=val_latent,
            shuffled_latent=shuffled,
            zero_latent=zero,
            device=device,
        )
        next_state_diagnostics.update(
            {
                "matched_action_control": matched_audit,
                "matched_observed_jepa_pred_to_next_state": _prediction_metrics(
                    matched_observed["pred_latent"][matched_position_tensor],
                    matched_target,
                    matched_keys,
                    device,
                ),
                "matched_shuffled_action_jepa_pred_to_next_state": _prediction_metrics(
                    matched["pred_latent"][matched_position_tensor],
                    matched_target,
                    matched_keys,
                    device,
                ),
            }
        )
        result = {
            "schema_version": "openclaw_text_jepa_next_state_diagnostics_v2",
            "diagnostic_only": True,
            "checkpoint": str(args.checkpoint),
            "control_checkpoint": (
                str(args.control_checkpoint) if args.control_checkpoint is not None else None
            ),
            "records_sha256": records_sha256,
            "split": {"train": train_scope, "validation": val_scope},
            "T2b_next_state_latent_prediction": next_state_diagnostics,
            "collapse_gate": collapse,
            "claim_boundary": (
                "observational next-context prediction only; no branched counterfactual actions "
                "or verified execution-status labels"
            ),
        }
        if args.include_tool_probes:
            t1_first, t1_atomic = _build_tool_probe_diagnostics(
                cache=cache,
                latent=latent,
                records=records,
                train_indices=train_indices,
                val_indices=val_indices,
                seeds=seeds,
                epochs=args.probe_epochs,
                device=device,
            )
            result["T1_first_tool_choice"] = t1_first
            result["T1_atomic_tool_choice"] = t1_atomic
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote next-state diagnostics to {args.output}")
        return 0

    seeds = [int(value.strip()) for value in args.probe_seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one probe seed is required")
    t1_first, t1_atomic = _build_tool_probe_diagnostics(
        cache=cache,
        latent=latent,
        records=records,
        train_indices=train_indices,
        val_indices=val_indices,
        seeds=seeds,
        epochs=args.probe_epochs,
        device=device,
    )

    val_tensor = torch.tensor(val_indices)
    val_latent = {key: value[val_tensor] for key, value in latent.items()}
    shuffled = _infer_latents(
        model, cache, val_indices, device, args.batch_size, shuffle_actions=True
    )
    zero = _infer_latents(
        model, cache, val_indices, device, args.batch_size, zero_actions=True
    )
    matched_indices, matched_action_indices, matched_audit = (
        _matched_action_derangement(records, val_indices)
    )
    if len(matched_indices) < 2:
        raise ValueError("matched shuffled-action control has fewer than two rows")
    matched = _infer_latents(
        model,
        cache,
        matched_indices,
        device,
        args.batch_size,
        action_indices=matched_action_indices,
    )
    matched_position = {index: position for position, index in enumerate(val_indices)}
    matched_val_positions = torch.tensor(
        [matched_position[index] for index in matched_indices]
    )
    matched_keys = [_feedback_key(records[index]) for index in matched_indices]
    keys = [_feedback_key(records[index]) for index in val_indices]
    common_state_identity = None
    if model.target_geometry == "frozen_random_orthogonal_v1":
        common_state_identity = model.map_fixed_target_hidden(
            cache["state_hidden"][val_tensor].to(device)
        ).detach().cpu()
    t2 = {
        "scope": "observational_task_group_heldout_multi_positive_retrieval",
        "jepa_pred_to_target": _retrieval_on_device(
            val_latent["pred_latent"], val_latent["target_latent"], keys, device
        ),
        "raw_state_to_raw_target": _retrieval_on_device(
            cache["state_hidden"][val_tensor], cache["target_hidden"][val_tensor], keys, device
        ),
        "raw_action_to_raw_target": _retrieval_on_device(
            cache["action_hidden"][val_tensor], cache["target_hidden"][val_tensor], keys, device
        ),
        "shuffled_action_jepa_pred_to_target": _retrieval_on_device(
            shuffled["pred_latent"], val_latent["target_latent"], keys, device
        ),
        "zero_action_jepa_pred_to_target": _retrieval_on_device(
            zero["pred_latent"], val_latent["target_latent"], keys, device
        ),
        "matched_action_control": matched_audit,
        "matched_observed_jepa_pred_to_target": _retrieval_on_device(
            val_latent["pred_latent"][matched_val_positions],
            val_latent["target_latent"][matched_val_positions],
            matched_keys,
            device,
        ),
        "matched_shuffled_action_jepa_pred_to_target": _retrieval_on_device(
            matched["pred_latent"],
            val_latent["target_latent"][matched_val_positions],
            matched_keys,
            device,
        ),
    }
    if common_state_identity is not None:
        t2["common_geometry_state_identity_to_target"] = _retrieval_on_device(
            common_state_identity,
            val_latent["target_latent"],
            keys,
            device,
        )
    t2_next_state = _build_next_state_diagnostics(
        records=records,
        cache=cache,
        val_indices=val_indices,
        val_latent=val_latent,
        shuffled_latent=shuffled,
        zero_latent=zero,
        device=device,
    )
    result = {
        "schema_version": "openclaw_text_jepa_offline_diagnostics_v2",
        "diagnostic_only": True,
        "checkpoint": str(args.checkpoint),
        "control_checkpoint": (
            str(args.control_checkpoint) if args.control_checkpoint is not None else None
        ),
        "records_sha256": records_sha256,
        "split": {"train": train_scope, "validation": val_scope},
        "T1_first_tool_choice": t1_first,
        "T1_atomic_tool_choice": t1_atomic,
        "T2_execution_result_retrieval": t2,
        "T2b_next_state_latent_prediction": t2_next_state,
        "collapse_gate": collapse,
        "unsupported_claims": {
            "structured_execution_status_accuracy": "blocked_by_data",
            "counterfactual_execution_result_prediction": "blocked_by_data",
            "reason": (
                "records lack verified structured execution labels, candidate_set_id, and "
                "environment snapshot provenance"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote offline diagnostics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
