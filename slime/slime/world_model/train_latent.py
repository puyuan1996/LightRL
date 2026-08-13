from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .cache_text_hidden import _build_cache_integrity_metadata, validate_hidden_cache_integrity
from .hidden_encoder import PolicyHiddenEncoder, hash_hidden_batch
from .metadata import is_verified_execution_reward_contract
from .metrics import effective_rank
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig
from .replay_buffer import TrajectoryReplayBuffer
from .seta_dataset import TerminalTransition, load_terminal_transitions
from .state_view import (
    BELIEF_VIEW_ALLOWLIST,
    BELIEF_VIEW_MAX_CHARS,
    BELIEF_VIEW_POOLING,
    BELIEF_VIEW_V1,
    FULL_CONTEXT_V1,
    STATE_VIEW_CHOICES,
    belief_view_metadata,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_records(path: Path, transitions: Sequence[TerminalTransition]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for transition in transitions:
            handle.write(
                json.dumps(
                    transition.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    return _file_sha256(path)


def _group_value(transition: TerminalTransition, key: str) -> str | None:
    if key in transition.__dataclass_fields__:
        value = getattr(transition, key)
    else:
        value = transition.to_dict().get(key)
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _value_label_contract(
    transitions: Sequence[TerminalTransition],
    indices: Sequence[int],
) -> dict[str, Any]:
    fields = [
        "reward_label_scope",
        "reward_label_source",
        "reward_label_semantics",
        "reward_label_is_execution_outcome",
    ]
    contracts: dict[str, dict[str, Any]] = {}
    labeled_count = 0
    for index in indices:
        row = transitions[index]
        if row.reward is None:
            continue
        labeled_count += 1
        contract = {field: getattr(row, field) for field in fields}
        key = json.dumps(contract, sort_keys=True, default=str)
        contracts[key] = contract
    consistent = len(contracts) == 1
    representative = next(iter(contracts.values())) if consistent else {field: None for field in fields}
    verified = bool(
        labeled_count > 0
        and consistent
        and is_verified_execution_reward_contract(representative)
    )
    return {
        **representative,
        "labeled_count": labeled_count,
        "contract_count": len(contracts),
        "consistent": consistent,
        "verified_execution_outcome": verified,
    }


def _roundtrip_replay(
    transitions: Sequence[TerminalTransition],
    *,
    output_path: Path,
    buffer_size: int,
    seed: int,
) -> tuple[list[TerminalTransition], dict[str, float]]:
    """Exercise verified replay persistence without silently changing the dataset."""

    if buffer_size < len(transitions):
        raise ValueError(
            "offline replay round-trip would evict transitions: "
            f"replay_buffer_size={buffer_size} record_count={len(transitions)}; "
            "increase --replay-buffer-size to at least the loaded record count"
        )
    replay = TrajectoryReplayBuffer(buffer_size, seed=seed)
    admitted = replay.push(transitions, current_step=0)
    if admitted != len(transitions) or len(replay) != len(transitions):
        raise ValueError(
            "offline replay round-trip rejected or deduplicated transitions; "
            "inspect transition IDs before training"
        )
    sampled = replay.sample(len(replay), current_step=0)
    if len(sampled) != len(transitions):
        raise RuntimeError("offline replay round-trip returned an incomplete sample")
    sampled_by_id = {str(record["transition_id"]): record for record in sampled}
    if len(sampled_by_id) != len(transitions):
        raise RuntimeError("offline replay sample contains duplicate transition IDs")
    replay.save(output_path)
    # Preserve canonical dataset order so replay/no-replay comparisons can
    # share one verified hidden cache. Sampling is still exercised and its RNG
    # state is persisted in the replay artifact.
    ordered = [sampled_by_id[row.transition_id] for row in transitions]
    restored = [TerminalTransition.from_dict(record) for record in ordered]
    if [row.to_dict() for row in restored] != [row.to_dict() for row in transitions]:
        raise RuntimeError("offline replay round-trip changed canonical transition contents")
    return restored, replay.stats()


def _loss_curve_summary(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("loss history must not be empty")

    def summarize(key: str) -> dict[str, Any]:
        rows = [
            (int(row["epoch"]), float(row[key]))
            for row in history
            if row.get(key) is not None
        ]
        if not rows:
            return {
                "initial": None,
                "final": None,
                "absolute_reduction": None,
                "relative_reduction": None,
                "best": None,
                "best_epoch": None,
            }
        initial = rows[0][1]
        final = rows[-1][1]
        best_epoch, best = min(rows, key=lambda item: item[1])
        return {
            "initial": initial,
            "final": final,
            "absolute_reduction": initial - final,
            "relative_reduction": (initial - final) / max(abs(initial), 1e-12),
            "best": best,
            "best_epoch": best_epoch,
        }

    return {
        "train": summarize("train_loss"),
        "validation": summarize("val_loss"),
    }


def _split_indices(
    transitions: Sequence[TerminalTransition],
    val_ratio: float,
    seed: int,
    group_key: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    count = len(transitions)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    group_values = [_group_value(row, group_key) for row in transitions]
    complete = bool(group_values) and all(value is not None for value in group_values)
    if val_ratio <= 0.0:
        indices = list(range(count))
        return indices, [], {
            "strategy": "no_validation",
            "group_key": group_key,
            "group_values_complete": complete,
            "group_disjoint": False,
            "confirmatory_group_key": False,
            "train_indices": indices,
            "val_indices": [],
        }
    if not complete:
        raise ValueError(
            f"group-heldout validation requires complete {group_key!r} metadata; "
            "record-level random fallback is intentionally disabled"
        )

    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(group_values):
        groups[str(value)].append(index)
    if len(groups) < 2:
        raise ValueError(
            f"group-heldout validation requires at least two distinct {group_key!r} values"
        )

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    target_count = max(1, min(count - 1, int(round(count * val_ratio))))
    val_indices: list[int] = []
    val_groups: list[str] = []
    for name in group_names[:-1]:
        val_groups.append(name)
        val_indices.extend(groups[name])
        if len(val_indices) >= target_count:
            break
    val_set = set(val_indices)
    train_indices = [index for index in range(count) if index not in val_set]
    if not train_indices or not val_indices:
        raise ValueError("group-heldout split produced an empty train or validation partition")
    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices, {
        "strategy": "group_holdout",
        "group_key": group_key,
        "group_values_complete": True,
        "group_disjoint": True,
        # Even task_cluster_id is only an artifact precondition here. The
        # trainer does not verify a frozen split manifest or an independent
        # test partition, so it must never emit a confirmatory checkpoint.
        "confirmatory_group_key": False,
        "strongest_available_group_key": group_key == "task_cluster_id",
        "group_count": len(groups),
        "train_group_count": len(groups) - len(val_groups),
        "val_group_count": len(val_groups),
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def _split_indices_from_manifest(
    transitions: Sequence[TerminalTransition],
    manifest_path: Path,
    *,
    records_sha256: str,
    group_key: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("--split-manifest must contain a JSON object")
    if value.get("schema_version") != "openclaw_group_kfold_split_v1":
        raise ValueError("unsupported split manifest schema_version")
    count = len(transitions)
    if value.get("records_sha256") != records_sha256:
        raise ValueError("split manifest records_sha256 does not match training records")
    if int(value.get("record_count", -1)) != count:
        raise ValueError("split manifest record_count does not match training records")
    if value.get("group_key") != group_key:
        raise ValueError("split manifest group_key does not match --split-group-key")

    def indices(name: str) -> list[int]:
        raw = value.get(name)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"split manifest {name} must be a non-empty list")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in raw):
            raise ValueError(f"split manifest {name} must contain integer indices")
        if len(raw) != len(set(raw)) or any(index < 0 or index >= count for index in raw):
            raise ValueError(f"split manifest {name} contains duplicate or out-of-range indices")
        return sorted(raw)

    train_indices = indices("train_indices")
    val_indices = indices("val_indices")
    train_set = set(train_indices)
    val_set = set(val_indices)
    if train_set & val_set:
        raise ValueError("split manifest train and validation indices overlap")
    if train_set | val_set != set(range(count)):
        raise ValueError("split manifest indices must partition all records")

    group_values = [_group_value(row, group_key) for row in transitions]
    if not group_values or any(group is None for group in group_values):
        raise ValueError(f"split manifest requires complete {group_key!r} metadata")
    train_groups = {str(group_values[index]) for index in train_indices}
    val_groups = {str(group_values[index]) for index in val_indices}
    if train_groups & val_groups:
        raise ValueError("split manifest is not group-disjoint")

    fold_index = int(value.get("fold_index", -1))
    fold_count = int(value.get("fold_count", -1))
    if fold_count < 2 or not 0 <= fold_index < fold_count:
        raise ValueError("split manifest fold_index/fold_count is invalid")
    assignment_seed = int(value.get("assignment_seed", 0))
    return train_indices, val_indices, {
        "strategy": "group_holdout",
        "source": "frozen_group_kfold_manifest",
        "group_key": group_key,
        "group_values_complete": True,
        "group_disjoint": True,
        "confirmatory_group_key": False,
        "strongest_available_group_key": group_key == "task_cluster_id",
        "group_count": len(train_groups | val_groups),
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "fold_index": fold_index,
        "fold_count": fold_count,
        "seed": assignment_seed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
    }


def _batches(indices: Sequence[int], batch_size: int, *, shuffle: bool, seed: int) -> list[list[int]]:
    values = list(indices)
    if shuffle:
        random.Random(seed).shuffle(values)
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def _select_hidden(
    hidden: dict[str, torch.Tensor],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    return {key: value.index_select(0, index).to(device) for key, value in hidden.items()}


def _cache_hidden(
    transitions: Sequence[TerminalTransition],
    *,
    encoder_kind: str,
    hash_hidden_dim: int,
    policy_encoder: PolicyHiddenEncoder | None,
    batch_size: int,
    state_view: str,
    belief_max_events: int,
) -> dict[str, torch.Tensor]:
    if encoder_kind == "hash":
        return hash_hidden_batch(
            transitions,
            hash_hidden_dim,
            state_view=state_view,
            belief_max_events=belief_max_events,
        )
    if policy_encoder is None:
        raise ValueError("policy_encoder is required for hf-policy encoding")
    rows: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, len(transitions), batch_size):
            batch = policy_encoder(
                transitions[start : start + batch_size],
                include_auxiliary_targets=True,
            )
            for key, value in batch.items():
                rows.setdefault(key, []).append(value.detach().cpu())
    return {key: torch.cat(values, dim=0) for key, values in rows.items()}


def _encoder_config(args: argparse.Namespace, *, final_backbone: bool) -> dict[str, Any]:
    if args.encoder == "hash":
        config = {
            "encoder": "hash",
            "hidden_dim": args.hash_hidden_dim,
            "schema": "openclaw_hash_hidden_v1",
        }
        if args.state_view != FULL_CONTEXT_V1:
            config.update(
                {
                    "state_view": args.state_view,
                    "state_view_pooling": BELIEF_VIEW_POOLING,
                    "state_view_allowlist": list(BELIEF_VIEW_ALLOWLIST),
                    "belief_max_events": args.belief_max_events,
                    "belief_event_max_chars": BELIEF_VIEW_MAX_CHARS,
                }
            )
        return config
    config = {
        "encoder": "hf-policy",
        "model_name_or_path": (
            str(Path(args.output_dir).expanduser() / "updated_llm")
            if final_backbone
            else args.hf_model
        ),
        "dtype": args.hf_dtype,
        "hidden_layer": args.hidden_layer,
        "action_pool": args.action_pool,
        "max_context_tokens": args.max_context_tokens,
        "max_action_tokens": args.max_action_tokens,
        "max_feedback_tokens": args.max_feedback_tokens,
        "encoder_long_text_mode": args.encoder_long_text_mode,
        "chunk_forward_batch_size": args.chunk_forward_batch_size,
        "strict_action_boundary": not args.allow_approximate_action_boundary,
        "local_files_only": args.hf_local_files_only,
        "trust_remote_code": args.hf_trust_remote_code,
        "backbone_updated": final_backbone,
        "llm_train_mode": args.llm_train_mode,
        "lora_merged_for_export": bool(final_backbone and args.llm_train_mode == "lora"),
        "lora_rank": args.lora_rank if args.llm_train_mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.llm_train_mode == "lora" else None,
        "lora_dropout": args.lora_dropout if args.llm_train_mode == "lora" else None,
        "lora_target_modules": (
            args.lora_target_modules if args.llm_train_mode == "lora" else None
        ),
        "fixed_target_backbone": args.fixed_target_backbone,
        "fixed_target_model_name_or_path": (
            args.hf_model if args.fixed_target_backbone else None
        ),
        "schema": "openclaw_policy_hidden_encoder_v2",
    }
    if args.state_view != FULL_CONTEXT_V1:
        config.update(
            {
                "state_view": args.state_view,
                "state_view_pooling": BELIEF_VIEW_POOLING,
                "state_view_allowlist": list(BELIEF_VIEW_ALLOWLIST),
                "belief_max_events": args.belief_max_events,
                "belief_event_max_chars": BELIEF_VIEW_MAX_CHARS,
            }
        )
    return config


def _save_hidden_cache(
    path: Path,
    *,
    hidden: dict[str, torch.Tensor],
    transitions: Sequence[TerminalTransition],
    input_records_sha256: str,
    encoder_config: dict[str, Any],
) -> dict[str, Any]:
    rewards = torch.tensor(
        [0.0 if row.reward is None else float(row.reward) for row in transitions],
        dtype=torch.float32,
    )
    reward_mask = torch.tensor([row.reward is not None for row in transitions], dtype=torch.bool)
    record_metadata = [row.to_dict() for row in transitions]
    if encoder_config.get("state_view") == BELIEF_VIEW_V1:
        max_events = int(encoder_config["belief_max_events"])
        for metadata, row in zip(record_metadata, transitions, strict=True):
            metadata.update(
                belief_view_metadata(
                    row.context_messages,
                    row.next_context_messages,
                    max_events=max_events,
                )
            )
    payload: dict[str, Any] = {
        **{key: value.detach().cpu() for key, value in hidden.items()},
        "record_count": len(transitions),
        "record_metadata": record_metadata,
        "reward": rewards,
        "reward_mask": reward_mask,
    }
    payload["metadata"] = _build_cache_integrity_metadata(
        payload,
        input_records_sha256=input_records_sha256,
        encoder_config=encoder_config,
    )
    validate_hidden_cache_integrity(payload, require_verified=True)
    torch.save(payload, path)
    return payload["metadata"]


def _load_verified_hidden_cache(
    source_path: Path,
    output_path: Path,
    *,
    transitions: Sequence[TerminalTransition],
    input_records_sha256: str,
    expected_encoder_config: dict[str, Any],
    allow_encoder_mismatch: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int]:
    if not source_path.is_file():
        raise FileNotFoundError(f"precomputed hidden cache does not exist: {source_path}")
    payload = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise TypeError("precomputed hidden cache must be a dictionary")
    validate_hidden_cache_integrity(payload, require_verified=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("precomputed hidden cache metadata is missing")
    if metadata.get("input_records_sha256") != input_records_sha256:
        raise ValueError(
            "precomputed hidden cache record digest does not match the current redacted dataset"
        )
    actual_encoder_config = metadata.get("encoder_config")
    if actual_encoder_config != expected_encoder_config:
        volatile_keys = {
            "model_name_or_path",
            "backbone_updated",
            "fixed_target_backbone",
            "fixed_target_model_name_or_path",
        }
        actual_stable = {
            key: value
            for key, value in (actual_encoder_config or {}).items()
            if key not in volatile_keys
        }
        expected_stable = {
            key: value
            for key, value in expected_encoder_config.items()
            if key not in volatile_keys
        }
        if not allow_encoder_mismatch or actual_stable != expected_stable:
            raise ValueError("precomputed hidden cache encoder config does not match this run")
    if int(payload.get("record_count", -1)) != len(transitions):
        raise ValueError("precomputed hidden cache record count does not match the current dataset")
    record_metadata = payload.get("record_metadata")
    if not isinstance(record_metadata, list):
        raise ValueError("precomputed hidden cache record metadata is missing")
    cached_ids = [str(record.get("transition_id")) for record in record_metadata]
    expected_ids = [row.transition_id for row in transitions]
    if cached_ids != expected_ids:
        raise ValueError("precomputed hidden cache transition order does not match the current dataset")
    if expected_encoder_config.get("state_view") == BELIEF_VIEW_V1:
        max_events = int(expected_encoder_config["belief_max_events"])
        expected_view = [
            belief_view_metadata(
                row.context_messages,
                row.next_context_messages,
                max_events=max_events,
            )
            for row in transitions
        ]
        for cached, expected in zip(record_metadata, expected_view, strict=True):
            if any(cached.get(key) != value for key, value in expected.items()):
                raise ValueError("precomputed hidden cache belief_view_v1 provenance mismatch")

    tensor_keys = [
        "state_hidden",
        "action_hidden",
        "target_hidden",
        "next_state_hidden",
        "has_next",
    ]
    missing = [key for key in tensor_keys if not isinstance(payload.get(key), torch.Tensor)]
    if missing:
        raise ValueError(f"precomputed hidden cache is missing tensors: {missing}")
    hidden = {key: payload[key] for key in tensor_keys}
    hidden_dim = int(hidden["state_hidden"].shape[-1])
    for key in ["action_hidden", "target_hidden", "next_state_hidden"]:
        if int(hidden[key].shape[-1]) != hidden_dim:
            raise ValueError("precomputed hidden cache uses inconsistent hidden dimensions")
    _validate_next_state_supervision(hidden, transitions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, output_path)
    except OSError:
        shutil.copy2(source_path, output_path)
    return hidden, dict(metadata), hidden_dim


def _validate_next_state_supervision(
    hidden: Mapping[str, torch.Tensor],
    transitions: Sequence[TerminalTransition],
) -> None:
    has_next = hidden.get("has_next")
    if not isinstance(has_next, torch.Tensor):
        raise ValueError("hidden cache has_next tensor is missing")
    if has_next.dtype != torch.bool or tuple(has_next.shape) != (len(transitions),):
        raise ValueError("hidden cache has_next must be bool with shape (record_count,)")
    expected = torch.tensor([row.has_next for row in transitions], dtype=torch.bool)
    if not torch.equal(has_next.detach().cpu(), expected):
        raise ValueError("hidden cache has_next does not match the redacted transition records")
    if any(row.has_next and not row.next_context_text for row in transitions):
        raise ValueError("has_next=true transition is missing its canonical next context")


def _run_epoch(
    *,
    model: TextLatentWorldModel,
    transitions: Sequence[TerminalTransition],
    indices: Sequence[int],
    cached_hidden: dict[str, torch.Tensor] | None,
    policy_encoder: PolicyHiddenEncoder | None,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    device: torch.device,
    seed: int,
    sigreg_coef: float,
    action_contrast_coef: float,
    alignment_coef: float,
    feedback_aux_coef: float = 0.0,
    value_coef: float,
    pred_loss_type: str = "mse",
    max_batches: int | None = None,
) -> tuple[float, dict[str, float], int, int]:
    training = optimizer is not None
    model.train(training)
    model.reset_microbatch_queue()
    if policy_encoder is not None:
        # Do not call policy_encoder.train(): nn.Module.train() would recurse
        # into the fixed target and put it in train mode.  The online/student
        # backbone follows the epoch mode; the target is immutable and always
        # evaluated without gradients.
        student_training = bool(training and policy_encoder.backprop_to_llm)
        policy_encoder.model.train(student_training)
        if policy_encoder.backprop_to_llm:
            gradient_checkpointing_enable = getattr(
                policy_encoder.model, "gradient_checkpointing_enable", None
            )
            if callable(gradient_checkpointing_enable):
                gradient_checkpointing_enable()
            student_config = getattr(policy_encoder.model, "config", None)
            if student_config is not None and hasattr(student_config, "use_cache"):
                student_config.use_cache = False
        if policy_encoder.target_model is not None:
            policy_encoder.target_model.eval()
            policy_encoder.target_model.requires_grad_(False)
    totals: dict[str, float] = {}
    total_loss = 0.0
    total_count = 0
    optimizer_steps = 0
    value_steps = 0
    grad_context = torch.enable_grad() if training else torch.no_grad()
    rng_context = nullcontext()
    if not training:
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        rng_context = torch.random.fork_rng(devices=cuda_devices)
    with rng_context, grad_context:
        if not training:
            # SIGReg samples random projections. Keep validation projections
            # fixed without perturbing the RNG stream used by the next epoch.
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
        batches = _batches(indices, batch_size, shuffle=training, seed=seed)
        if max_batches is not None:
            batches = batches[:max_batches]
        for batch_indices in batches:
            batch_transitions = [transitions[index] for index in batch_indices]
            if cached_hidden is not None:
                hidden = _select_hidden(cached_hidden, batch_indices, device)
            else:
                if policy_encoder is None:
                    raise RuntimeError("End-to-end training requires a policy hidden encoder")
                if feedback_aux_coef > 0.0:
                    hidden = policy_encoder(
                        batch_transitions,
                        include_auxiliary_targets=True,
                    )
                else:
                    hidden = policy_encoder(batch_transitions)
            rewards = torch.tensor(
                [0.0 if row.reward is None else float(row.reward) for row in batch_transitions],
                dtype=torch.float32,
                device=device,
            )
            reward_mask = torch.tensor(
                [row.reward is not None for row in batch_transitions],
                dtype=torch.bool,
                device=device,
            )
            loss, metrics = model.compute_loss(
                state_hidden=hidden["state_hidden"],
                action_hidden=hidden["action_hidden"],
                target_hidden=hidden["target_hidden"],
                next_state_hidden=hidden["next_state_hidden"],
                has_next=hidden["has_next"],
                reward=rewards,
                reward_mask=reward_mask,
                pred_loss_type=pred_loss_type,
                sigreg_coef=sigreg_coef,
                action_contrast_coef=action_contrast_coef,
                alignment_coef=alignment_coef,
                feedback_aux_coef=feedback_aux_coef,
                value_coef=value_coef,
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite world-model loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.update_target_ema()
                optimizer_steps += 1
                if value_coef > 0.0 and bool(reward_mask.any().item()):
                    value_steps += 1
            count = len(batch_indices)
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
            for key, value in metrics.items():
                metric_value = float(value.detach().cpu())
                if key in {"wm/value_mask_count", "wm/prediction_mask_count"}:
                    totals[key] = totals.get(key, 0.0) + metric_value
                else:
                    totals[key] = totals.get(key, 0.0) + metric_value * count
    averaged = {
        key: (
            value
            if key in {"wm/value_mask_count", "wm/prediction_mask_count"}
            else value / max(total_count, 1)
        )
        for key, value in totals.items()
    }
    return total_loss / max(total_count, 1), averaged, optimizer_steps, value_steps


def _distribution_diagnostics(value: torch.Tensor, prefix: str) -> dict[str, float]:
    value = value.float()
    count, width = value.shape
    variance = value.var(dim=0, unbiased=False)
    centered = value - value.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(count - 1, 1)
    off_diagonal = covariance[~torch.eye(width, dtype=torch.bool)]
    normalized = F.normalize(value, dim=-1)
    if count > 1:
        pairwise_cosine = (
            normalized.sum(dim=0).square().sum() - float(count)
        ) / float(count * (count - 1))
    else:
        pairwise_cosine = torch.ones(())
    return {
        f"wm/{prefix}_effective_rank": float(effective_rank(value)),
        f"wm/{prefix}_variance_mean": float(variance.mean()),
        f"wm/{prefix}_variance_min": float(variance.min()),
        f"wm/{prefix}_offdiag_cov_rms": (
            float(off_diagonal.square().mean().sqrt()) if off_diagonal.numel() else 0.0
        ),
        f"wm/{prefix}_pairwise_cosine": float(pairwise_cosine),
    }


@torch.no_grad()
def _partition_latent_diagnostics(
    *,
    model: TextLatentWorldModel,
    transitions: Sequence[TerminalTransition],
    indices: Sequence[int],
    cached_hidden: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    rows: dict[str, list[torch.Tensor]] = {
        "state": [],
        "pred": [],
        "target": [],
    }
    model.eval()
    for batch_indices in _batches(indices, batch_size, shuffle=False, seed=0):
        hidden = _select_hidden(cached_hidden, batch_indices, device)
        output = model(
            state_hidden=hidden["state_hidden"],
            action_hidden=hidden["action_hidden"],
            target_hidden=hidden["target_hidden"],
            next_state_hidden=hidden["next_state_hidden"],
        )
        if model.config.prediction_target == "next_state":
            mask = hidden["has_next"].to(dtype=torch.bool).view(-1)
            target = output["next_state_target_latent"]
        else:
            mask = torch.ones(len(batch_indices), dtype=torch.bool, device=device)
            target = output["target_latent"]
        rows["state"].append(output["state_latent"][mask].detach().cpu())
        rows["pred"].append(output["pred_latent"][mask].detach().cpu())
        rows["target"].append(target[mask].detach().cpu())
    diagnostics: dict[str, float] = {}
    for prefix, values in rows.items():
        diagnostics.update(_distribution_diagnostics(torch.cat(values, dim=0), prefix))
    # Preserve established metric names, now computed once over the complete
    # heldout partition rather than averaged over minibatches.
    diagnostics["wm/effective_rank"] = diagnostics["wm/state_effective_rank"]
    return diagnostics


@torch.no_grad()
def _write_predictions(
    *,
    path: Path,
    model: TextLatentWorldModel,
    transitions: Sequence[TerminalTransition],
    cached_hidden: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> None:
    model.eval()
    with path.open("w", encoding="utf-8") as handle:
        for indices in _batches(range(len(transitions)), batch_size, shuffle=False, seed=0):
            batch_transitions = [transitions[index] for index in indices]
            hidden = _select_hidden(cached_hidden, indices, device)
            output = model(
                state_hidden=hidden["state_hidden"],
                action_hidden=hidden["action_hidden"],
                target_hidden=hidden["target_hidden"],
                next_state_hidden=hidden["next_state_hidden"],
            )
            feedback_error = (output["pred_latent"] - output["target_latent"]).pow(2).mean(dim=-1)
            next_state_error = (
                (output["pred_latent"] - output["next_state_target_latent"]).pow(2).mean(dim=-1)
            )
            for offset, transition in enumerate(batch_transitions):
                if model.config.prediction_target == "next_state":
                    prediction_target_latent = (
                        output["next_state_target_latent"][offset] if transition.has_next else None
                    )
                    prediction_error = next_state_error[offset] if transition.has_next else None
                else:
                    prediction_target_latent = output["target_latent"][offset]
                    prediction_error = feedback_error[offset]
                row = {
                    "transition_id": transition.transition_id,
                    "trajectory_id": transition.trajectory_id,
                    "task_name": transition.task_name,
                    "turn_idx": transition.turn_idx,
                    "done": transition.done,
                    "has_next": transition.has_next,
                    "reward": transition.reward,
                    # Preserve the established feedback fields for downstream
                    # readers, and add explicit selected-target fields.
                    "latent_mse": float(feedback_error[offset].cpu()),
                    "pred_latent": output["pred_latent"][offset].cpu().tolist(),
                    "target_latent": output["target_latent"][offset].cpu().tolist(),
                    "prediction_target": model.config.prediction_target,
                    "prediction_target_mse": (
                        None if prediction_error is None else float(prediction_error.cpu())
                    ),
                    "prediction_target_latent": (
                        None
                        if prediction_target_latent is None
                        else prediction_target_latent.cpu().tolist()
                    ),
                    "next_state_latent_mse": (
                        float(next_state_error[offset].cpu()) if transition.has_next else None
                    ),
                    "value": None if output["value"] is None else float(output["value"][offset].cpu()),
                    "uncertainty": None,
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an action-conditioned latent world model on SETA replay.")
    parser.add_argument("--input", required=True, help="SETA directory, redacted records JSONL, or replay .pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", choices=["hash", "hf-policy"], default="hash")
    parser.add_argument("--hash-hidden-dim", type=int, default=256)
    parser.add_argument("--hf-model", default=None)
    hf_access = parser.add_mutually_exclusive_group()
    hf_access.add_argument("--hf-local-files-only", dest="hf_local_files_only", action="store_true")
    hf_access.add_argument("--hf-allow-download", dest="hf_local_files_only", action="store_false")
    parser.set_defaults(hf_local_files_only=True)
    parser.add_argument("--hf-trust-remote-code", action="store_true")
    parser.add_argument("--hf-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--action-pool", choices=["mean", "last"], default="mean")
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--max-feedback-tokens", type=int, default=512)
    parser.add_argument(
        "--encoder-long-text-mode",
        choices=["tail_v1", "hierarchical_chunks_v1"],
        default="tail_v1",
    )
    parser.add_argument("--chunk-forward-batch-size", type=int, default=16)
    parser.add_argument(
        "--state-view",
        choices=STATE_VIEW_CHOICES,
        default=FULL_CONTEXT_V1,
        help="Encode the full transcript or the versioned dynamic belief view.",
    )
    parser.add_argument("--belief-max-events", type=int, default=3)
    parser.add_argument("--allow-approximate-action-boundary", action="store_true")
    parser.add_argument(
        "--backprop-to-llm",
        "--world-model-backprop-to-llm",
        dest="backprop_to_llm",
        action="store_true",
        help="Update the policy backbone; requires --save-updated-llm.",
    )
    parser.add_argument("--save-updated-llm", action="store_true")
    parser.add_argument(
        "--llm-train-mode",
        choices=["full", "lora"],
        default="full",
        help="Update all backbone parameters or a PEFT LoRA adapter.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--fixed-target-backbone",
        action="store_true",
        help=(
            "Use a separate frozen copy of the initial policy backbone for feedback/next-state "
            "targets while the online backbone is updated."
        ),
    )
    parser.add_argument(
        "--use-dapo-replay-buffer",
        "--world-model-use-dapo-replay-buffer",
        dest="use_dapo_replay_buffer",
        action="store_true",
        help="Round-trip loaded transitions through the verified bounded replay buffer.",
    )
    parser.add_argument("--replay-buffer-size", type=int, default=2048)
    parser.add_argument("--allow-unverified-replay", action="store_true")
    parser.add_argument(
        "--allow-unverified-world-model-records",
        action="store_true",
        help=(
            "Allow loading non-canonical tool-call/result-only world-model records by"
            " treating them as unverified and non-canonical."
        ),
    )
    parser.add_argument(
        "--hidden-cache-input",
        default=None,
        help="Reuse a verified cache whose records and encoder config exactly match this run.",
    )
    parser.add_argument(
        "--allow-cache-encoder-mismatch",
        action="store_true",
        help=(
            "Allow intentional reuse of a cache with a different backbone path/update state; "
            "record digest, cache provenance, and view configuration remain fail-closed."
        ),
    )
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--max-text-chars", type=int, default=4096)
    parser.add_argument("--require-tool-feedback", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--predictor-type", choices=["adaln", "mlp"], default="adaln")
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-num-heads", type=int, default=4)
    parser.add_argument("--predictor-mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--prediction-target",
        choices=["feedback", "next_state"],
        default="feedback",
        help="Supervise the predictor with observed feedback or the direct future state latent.",
    )
    parser.add_argument(
        "--predictor-input-mode",
        choices=["observed", "state_only", "action_only"],
        default="observed",
        help="Use both inputs or a parameter-matched state-only/action-only baseline.",
    )
    parser.add_argument(
        "--prediction-form",
        choices=["direct", "residual"],
        default="direct",
        help="Predict the next latent directly or as an update to the current state latent.",
    )
    parser.add_argument(
        "--pred-loss-type",
        choices=["mse", "scaled_mse", "cosine", "smooth_l1"],
        default="mse",
    )
    parser.add_argument(
        "--objective-population",
        choices=["all", "has_next"],
        default="all",
        help="Use all split records or the matched has_next=true population.",
    )
    parser.add_argument("--stop-grad-target", action="store_true")
    parser.add_argument("--target-ema-decay", type=float, default=0.996)
    parser.add_argument(
        "--target-geometry",
        choices=["learned_shared_v2", "frozen_random_orthogonal_v1"],
        default="learned_shared_v2",
    )
    parser.add_argument("--fixed-target-seed", type=int, default=20260731)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--train-batches-per-epoch",
        type=int,
        default=0,
        help="Bound each training chunk; zero preserves full-dataset epoch behavior.",
    )
    parser.add_argument(
        "--validation-batches-per-epoch",
        type=int,
        default=0,
        help="Bound validation work per validation chunk; zero evaluates the full split.",
    )
    parser.add_argument(
        "--min-train-seconds",
        type=float,
        default=0.0,
        help="Minimum cumulative time spent in training epochs (validation and export excluded).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Fail closed if this epoch ceiling is reached before the minimum duration contract.",
    )
    parser.add_argument("--validation-interval-epochs", type=int, default=1)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--start-file", type=Path, default=None)
    parser.add_argument("--phase-file", type=Path, default=None)
    parser.add_argument("--barrier-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument(
        "--microbatch-queue-size",
        type=int,
        default=0,
        help="Detached latent queue used by SIGReg and action contrast for batch_size=1.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["final_epoch", "best_validation"],
        default="final_epoch",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--llm-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sigreg-coef", type=float, default=0.09)
    parser.add_argument(
        "--sigreg-scope",
        choices=["state", "pred", "state_pred"],
        default="state",
        help="Apply SIGReg to the state latent, prediction latent, or both.",
    )
    parser.add_argument("--action-contrast-coef", type=float, default=0.1)
    parser.add_argument("--alignment-coef", type=float, default=0.1)
    parser.add_argument(
        "--feedback-aux-coef",
        type=float,
        default=0.0,
        help=(
            "For a next-state objective, also align the predicted next-belief latent "
            "to the observed feedback latent."
        ),
    )
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--allow-unverified-value-labels", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-group-key", default="trajectory_id")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional frozen group-kfold partition bound to the exact records digest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Holdout assignment seed; defaults to --seed for backward compatibility.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.encoder == "hf-policy" and not args.hf_model:
        raise ValueError("--hf-model is required when --encoder hf-policy")
    if args.encoder == "hash" and args.backprop_to_llm:
        raise ValueError("--backprop-to-llm requires --encoder hf-policy")
    if args.hidden_cache_input and args.backprop_to_llm:
        raise ValueError("--hidden-cache-input is incompatible with --backprop-to-llm")
    if args.backprop_to_llm and not args.save_updated_llm:
        raise ValueError("--backprop-to-llm requires --save-updated-llm so backbone updates are not discarded")
    if args.save_updated_llm and not args.backprop_to_llm:
        raise ValueError("--save-updated-llm requires --backprop-to-llm")
    if args.llm_train_mode == "lora" and not args.backprop_to_llm:
        raise ValueError("--llm-train-mode lora requires --backprop-to-llm")
    if args.llm_train_mode == "lora":
        if args.lora_rank <= 0 or args.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= args.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not [value.strip() for value in args.lora_target_modules.split(",") if value.strip()]:
            raise ValueError("LoRA target modules cannot be empty")
    if args.fixed_target_backbone and not args.backprop_to_llm:
        raise ValueError("--fixed-target-backbone requires --backprop-to-llm")
    if (
        args.backprop_to_llm
        and args.prediction_target == "next_state"
        and not args.fixed_target_backbone
    ):
        raise ValueError(
            "end-to-end next-state training requires --fixed-target-backbone to prevent target drift"
        )
    if (args.ready_file is None) != (args.start_file is None):
        raise ValueError("--ready-file and --start-file must be supplied together")
    if args.latent_dim % args.predictor_num_heads != 0 and args.predictor_type == "adaln":
        raise ValueError("--latent-dim must be divisible by --predictor-num-heads")
    for name in [
        "epochs",
        "batch_size",
        "encode_batch_size",
        "hash_hidden_dim",
        "replay_buffer_size",
        "max_text_chars",
        "belief_max_events",
        "chunk_forward_batch_size",
        "validation_interval_epochs",
    ]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.microbatch_queue_size) < 0:
        raise ValueError("--microbatch-queue-size must be non-negative")
    for name in ["train_batches_per_epoch", "validation_batches_per_epoch"]:
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ["lr", "llm_lr"]:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in [
        "weight_decay",
        "sigreg_coef",
        "action_contrast_coef",
        "alignment_coef",
        "feedback_aux_coef",
        "value_coef",
    ]:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.feedback_aux_coef > 0.0 and args.prediction_target != "next_state":
        raise ValueError("--feedback-aux-coef requires --prediction-target next_state")
    if not 0.0 <= float(args.target_ema_decay) < 1.0:
        raise ValueError("--target-ema-decay must be in [0, 1)")
    if not math.isfinite(args.min_train_seconds) or args.min_train_seconds < 0.0:
        raise ValueError("--min-train-seconds must be finite and non-negative")
    if not math.isfinite(args.barrier_timeout_seconds) or args.barrier_timeout_seconds <= 0.0:
        raise ValueError("--barrier-timeout-seconds must be finite and positive")
    if args.max_epochs is not None and args.max_epochs < args.epochs:
        raise ValueError("--max-epochs must be at least --epochs")


def _clone_state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in module.state_dict().items()
    }


def _clone_trainable_state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu", copy=True)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state_dict(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    current = dict(module.named_parameters())
    if not state.keys() <= current.keys():
        raise RuntimeError("trainable backbone parameter keys changed")
    with torch.no_grad():
        for name, value in state.items():
            if not current[name].requires_grad:
                raise RuntimeError(f"backbone parameter is no longer trainable: {name}")
            current[name].copy_(
                value.to(device=current[name].device, dtype=current[name].dtype)
            )


def _write_phase(path: Path | None, phase: str, mode: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"{phase} {mode}\n", encoding="utf-8")
    temporary.replace(path)


def _barrier_wait(args: argparse.Namespace, output_dir: Path) -> None:
    if args.ready_file is None or args.start_file is None:
        return
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_dir": str(output_dir),
        "pid": os.getpid(),
        "ready_at": datetime.now(timezone.utc).isoformat(),
        "prediction_target": args.prediction_target,
        "backprop_to_llm": args.backprop_to_llm,
        "fixed_target_backbone": args.fixed_target_backbone,
    }
    temporary = args.ready_file.with_suffix(args.ready_file.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.ready_file)
    deadline = time.monotonic() + args.barrier_timeout_seconds
    while not args.start_file.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for synchronized start: {args.start_file}")
        time.sleep(0.5)


def _training_contract_pending(
    *,
    completed_epochs: int,
    minimum_epochs: int,
    cumulative_train_seconds: float,
    minimum_train_seconds: float,
) -> bool:
    """Return whether either independently required training floor is unmet."""

    return (
        completed_epochs < minimum_epochs
        or cumulative_train_seconds < minimum_train_seconds
    )


def _parameter_delta_l2(
    parameters: Sequence[torch.nn.Parameter],
    initial: Sequence[torch.Tensor],
) -> float | None:
    if not parameters:
        return None
    squared = 0.0
    for parameter, baseline in zip(parameters, initial, strict=True):
        delta = parameter.detach().cpu().float() - baseline.float()
        squared += float(delta.square().sum().item())
    return math.sqrt(squared)


@torch.no_grad()
def _module_parameter_delta_l2(
    updated: torch.nn.Module,
    reference: torch.nn.Module,
    *,
    chunk_elements: int = 1_048_576,
) -> float:
    updated_parameters = dict(updated.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    if updated_parameters.keys() != reference_parameters.keys():
        raise RuntimeError("online and fixed target backbone parameter keys differ")
    squared = 0.0
    for name, value in updated_parameters.items():
        baseline = reference_parameters[name]
        if value.shape != baseline.shape:
            raise RuntimeError(f"backbone parameter shape mismatch: {name}")
        value_flat = value.detach().reshape(-1)
        baseline_flat = baseline.detach().reshape(-1)
        for start in range(0, value_flat.numel(), chunk_elements):
            delta = (
                value_flat[start : start + chunk_elements].float()
                - baseline_flat[start : start + chunk_elements].float()
            )
            squared += float(delta.square().sum().item())
    return math.sqrt(squared)


def main() -> None:
    args = _build_parser().parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in ["latent_world_model.pt", "hidden_cache.pt", "records.jsonl"]:
        if (output_dir / artifact).exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output_dir / artifact}")

    transitions = load_terminal_transitions(
        args.input,
        max_trajectories=args.max_trajectories,
        max_transitions=args.max_transitions,
        require_tool_feedback=args.require_tool_feedback,
        allow_unverified_replay=args.allow_unverified_replay,
        allow_unverified_world_model_views=args.allow_unverified_world_model_records,
        max_text_chars=args.max_text_chars,
    )
    if not transitions:
        raise ValueError(f"No valid transitions found in {args.input}")

    replay_stats = None
    if args.use_dapo_replay_buffer:
        transitions, replay_stats = _roundtrip_replay(
            transitions,
            output_path=output_dir / "dapo_replay.pt",
            buffer_size=args.replay_buffer_size,
            seed=args.seed,
        )

    records_path = output_dir / "records.jsonl"
    input_records_sha256 = _write_records(records_path, transitions)
    if args.split_manifest is not None:
        train_indices, val_indices, split_metadata = _split_indices_from_manifest(
            transitions,
            args.split_manifest,
            records_sha256=input_records_sha256,
            group_key=args.split_group_key,
        )
        split_seed = int(split_metadata["seed"])
    else:
        split_seed = args.seed if args.split_seed is None else args.split_seed
        train_indices, val_indices, split_metadata = _split_indices(
            transitions,
            args.val_ratio,
            split_seed,
            args.split_group_key,
        )
        split_metadata["seed"] = split_seed
    if args.prediction_target == "next_state" or args.objective_population == "has_next":
        complete_train_indices = train_indices
        complete_val_indices = val_indices
        train_indices = [index for index in train_indices if transitions[index].has_next]
        val_indices = [index for index in val_indices if transitions[index].has_next]
        if not train_indices:
            raise ValueError(
                "the selected objective population has no has_next=true train transitions"
            )
        if complete_val_indices and not val_indices:
            raise ValueError(
                "the selected objective population has no has_next=true validation transitions"
            )
        split_metadata["prediction_target_filter"] = {
            "target": args.prediction_target,
            "population": "has_next",
            "requires_has_next": True,
            "pre_filter_train_count": len(complete_train_indices),
            "pre_filter_val_count": len(complete_val_indices),
            "train_count": len(train_indices),
            "val_count": len(val_indices),
            "objective_train_indices": train_indices,
            "objective_val_indices": val_indices,
        }
    train_value_contract = _value_label_contract(transitions, train_indices)
    val_value_contract = _value_label_contract(transitions, val_indices) if val_indices else None
    train_reward_label_count = int(train_value_contract["labeled_count"])
    verified_value_labels = bool(train_value_contract["verified_execution_outcome"])
    if args.value_coef > 0.0 and train_reward_label_count <= 0:
        raise ValueError("--value-coef is positive but the train split has no valid reward labels")
    if args.value_coef > 0.0 and not train_value_contract["consistent"]:
        raise ValueError("value supervision mixes incompatible reward label contracts")
    if args.value_coef > 0.0 and val_value_contract is not None:
        if int(val_value_contract["labeled_count"]) <= 0:
            raise ValueError("--value-coef is positive but the validation split has no valid reward labels")
        if not val_value_contract["consistent"]:
            raise ValueError("validation value supervision mixes incompatible reward label contracts")
        contract_fields = (
            "reward_label_scope",
            "reward_label_source",
            "reward_label_semantics",
            "reward_label_is_execution_outcome",
        )
        if any(
            train_value_contract.get(field) != val_value_contract.get(field)
            for field in contract_fields
        ):
            raise ValueError("train and validation value label contracts do not match")
        verified_value_labels = bool(
            verified_value_labels and val_value_contract["verified_execution_outcome"]
        )
    if args.value_coef > 0.0 and not verified_value_labels and not args.allow_unverified_value_labels:
        raise ValueError(
            "value labels are not verified execution outcomes; pass --allow-unverified-value-labels "
            "only for an explicitly diagnostic run"
        )

    device = _device(args.device)
    policy_encoder: PolicyHiddenEncoder | None = None
    cached_hidden: dict[str, torch.Tensor] | None = None
    cache_metadata: dict[str, Any] | None = None
    if args.hidden_cache_input:
        cached_hidden, cache_metadata, hidden_dim = _load_verified_hidden_cache(
            Path(args.hidden_cache_input).expanduser(),
            output_dir / "hidden_cache.pt",
            transitions=transitions,
            input_records_sha256=input_records_sha256,
            expected_encoder_config=_encoder_config(args, final_backbone=False),
            allow_encoder_mismatch=args.allow_cache_encoder_mismatch,
        )
    elif args.encoder == "hf-policy":
        policy_encoder = PolicyHiddenEncoder.from_pretrained(
            args.hf_model,
            device=str(device),
            dtype=args.hf_dtype,
            local_files_only=args.hf_local_files_only,
            trust_remote_code=args.hf_trust_remote_code,
            hidden_layer=args.hidden_layer,
            action_pool=args.action_pool,
            max_context_tokens=args.max_context_tokens,
            max_action_tokens=args.max_action_tokens,
            max_feedback_tokens=args.max_feedback_tokens,
            backprop_to_llm=args.backprop_to_llm,
            llm_train_mode=args.llm_train_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            fixed_target_backbone=args.fixed_target_backbone,
            strict_action_boundary=not args.allow_approximate_action_boundary,
            state_view=args.state_view,
            belief_max_events=args.belief_max_events,
            encoder_long_text_mode=args.encoder_long_text_mode,
            chunk_forward_batch_size=args.chunk_forward_batch_size,
            prediction_target=args.prediction_target,
        )
        hidden_dim = policy_encoder.hidden_size
    else:
        hidden_dim = args.hash_hidden_dim

    if not args.backprop_to_llm and cached_hidden is None:
        cached_hidden = _cache_hidden(
            transitions,
            encoder_kind=args.encoder,
            hash_hidden_dim=args.hash_hidden_dim,
            policy_encoder=policy_encoder,
            batch_size=args.encode_batch_size,
            state_view=args.state_view,
            belief_max_events=args.belief_max_events,
        )
        cache_metadata = _save_hidden_cache(
            output_dir / "hidden_cache.pt",
            hidden=cached_hidden,
            transitions=transitions,
            input_records_sha256=input_records_sha256,
            encoder_config=_encoder_config(args, final_backbone=False),
        )
        if policy_encoder is not None:
            del policy_encoder
            policy_encoder = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if cached_hidden is not None:
        _validate_next_state_supervision(cached_hidden, transitions)

    config = TextLatentWorldModelConfig(
        state_hidden_dim=hidden_dim,
        action_hidden_dim=hidden_dim,
        target_hidden_dim=hidden_dim,
        latent_dim=args.latent_dim,
        adapter_dim=args.adapter_dim,
        predictor_type=args.predictor_type,
        architecture_version="shared_latent_v2",
        prediction_target=args.prediction_target,
        predictor_input_mode=args.predictor_input_mode,
        prediction_form=args.prediction_form,
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        predictor_mlp_ratio=args.predictor_mlp_ratio,
        value_head=args.value_coef > 0.0,
        uncertainty_head=False,
        stop_grad_target=args.stop_grad_target,
        target_ema_decay=args.target_ema_decay,
        target_geometry=args.target_geometry,
        fixed_target_seed=args.fixed_target_seed,
        sigreg_scope=args.sigreg_scope,
        microbatch_queue_size=args.microbatch_queue_size,
    )
    model = TextLatentWorldModel(config).to(device)
    initial_model_state = _clone_state_dict_cpu(model)
    value_parameters = list(model.value_head.parameters()) if model.value_head is not None else []
    initial_value_parameters = [parameter.detach().cpu().clone() for parameter in value_parameters]
    parameter_groups: list[dict[str, Any]] = [
        {"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": args.lr}
    ]
    backbone_total_parameter_count = 0
    backbone_trainable_parameter_count = 0
    if args.backprop_to_llm:
        if policy_encoder is None:
            raise RuntimeError("policy encoder was not initialized")
        backbone_parameters = [
            parameter
            for parameter in policy_encoder.model.parameters()
            if parameter.requires_grad
        ]
        if not backbone_parameters:
            raise RuntimeError("backbone training requested without trainable parameters")
        backbone_total_parameter_count = sum(
            parameter.numel() for parameter in policy_encoder.model.parameters()
        )
        backbone_trainable_parameter_count = sum(
            parameter.numel() for parameter in backbone_parameters
        )
        parameter_groups.append({"params": backbone_parameters, "lr": args.llm_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    optimizer_step_count = 0
    value_update_step_count = 0
    cumulative_train_seconds = 0.0
    best_epoch = 0
    best_selection_loss = math.inf
    best_model_state: dict[str, torch.Tensor] | None = None
    best_backbone_state: dict[str, torch.Tensor] | None = None
    _write_phase(args.phase_file, "model_ready", "audit")
    _barrier_wait(args, output_dir)
    _write_phase(args.phase_file, "backbone_train", "critical")
    epoch = 0
    while _training_contract_pending(
        completed_epochs=epoch,
        minimum_epochs=args.epochs,
        cumulative_train_seconds=cumulative_train_seconds,
        minimum_train_seconds=args.min_train_seconds,
    ):
        if args.max_epochs is not None and epoch >= args.max_epochs:
            raise RuntimeError(
                "maximum epoch ceiling reached before satisfying the minimum training duration: "
                f"epochs={epoch} train_seconds={cumulative_train_seconds:.3f} "
                f"required={args.min_train_seconds:.3f}"
            )
        train_started = time.monotonic()
        train_loss, train_metrics, optimizer_steps, value_steps = _run_epoch(
            model=model,
            transitions=transitions,
            indices=train_indices,
            cached_hidden=cached_hidden,
            policy_encoder=policy_encoder,
            optimizer=optimizer,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + epoch,
            sigreg_coef=args.sigreg_coef,
            action_contrast_coef=args.action_contrast_coef,
            alignment_coef=args.alignment_coef,
            feedback_aux_coef=args.feedback_aux_coef,
            value_coef=args.value_coef,
            pred_loss_type=args.pred_loss_type,
            max_batches=(args.train_batches_per_epoch or None),
        )
        cumulative_train_seconds += time.monotonic() - train_started
        optimizer_step_count += optimizer_steps
        value_update_step_count += value_steps
        val_loss = None
        val_metrics: dict[str, float] = {}
        duration_will_be_satisfied = cumulative_train_seconds >= args.min_train_seconds
        minimum_epochs_will_be_satisfied = epoch + 1 >= args.epochs
        should_validate = (
            (epoch + 1) % args.validation_interval_epochs == 0
            or (duration_will_be_satisfied and minimum_epochs_will_be_satisfied)
        )
        if val_indices and should_validate:
            val_loss, val_metrics, _, _ = _run_epoch(
                model=model,
                transitions=transitions,
                indices=val_indices,
                cached_hidden=cached_hidden,
                policy_encoder=policy_encoder,
                optimizer=None,
                batch_size=args.batch_size,
                device=device,
                seed=split_seed,
                sigreg_coef=args.sigreg_coef,
                action_contrast_coef=args.action_contrast_coef,
                alignment_coef=args.alignment_coef,
                feedback_aux_coef=args.feedback_aux_coef,
                value_coef=args.value_coef,
                pred_loss_type=args.pred_loss_type,
                max_batches=(args.validation_batches_per_epoch or None),
            )
            if cached_hidden is not None:
                val_metrics.update(
                    _partition_latent_diagnostics(
                        model=model,
                        transitions=transitions,
                        indices=val_indices,
                        cached_hidden=cached_hidden,
                        batch_size=args.batch_size,
                        device=device,
                    )
                )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        selection_loss = train_loss if not val_indices else val_loss
        if selection_loss is not None and selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_epoch = epoch + 1
            if args.checkpoint_selection == "best_validation":
                best_model_state = _clone_state_dict_cpu(model)
                if policy_encoder is not None and args.backprop_to_llm:
                    best_backbone_state = _clone_trainable_state_dict_cpu(
                        policy_encoder.model
                    )
        epoch += 1
    _write_phase(args.phase_file, "backbone_train_complete", "audit")

    if best_epoch <= 0:
        raise RuntimeError("training completed without observing a finite selection metric")

    # The final cache is an inference-only export.  Release AdamW state and
    # training gradients before running another full Qwen forward pass; on an
    # 8B student plus frozen target this is the difference between fitting the
    # training loop and exhausting the H200 during cache export.
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    if policy_encoder is not None:
        policy_encoder.model.eval()
        if policy_encoder.target_model is not None:
            policy_encoder.target_model.eval()
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.checkpoint_selection == "best_validation":
        if best_model_state is None:
            raise RuntimeError("best-validation checkpoint state was not captured")
        model.load_state_dict(best_model_state, strict=True)
        if policy_encoder is not None and args.backprop_to_llm:
            if best_backbone_state is None:
                raise RuntimeError("best-validation backbone state was not captured")
            _restore_trainable_state_dict(policy_encoder.model, best_backbone_state)
        del best_model_state, best_backbone_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    backbone_parameter_delta_l2 = None
    if args.backprop_to_llm:
        if policy_encoder is None:
            raise RuntimeError("policy encoder disappeared before final cache export")
        if args.fixed_target_backbone:
            if policy_encoder.target_model is None:
                raise RuntimeError("fixed target backbone disappeared before export")
            if args.llm_train_mode == "lora":
                policy_encoder.merge_lora_for_export()
            backbone_parameter_delta_l2 = _module_parameter_delta_l2(
                policy_encoder.model, policy_encoder.target_model
            )
            if backbone_parameter_delta_l2 <= 0.0:
                raise RuntimeError("backpropagation produced no measurable backbone parameter update")
        cached_hidden = _cache_hidden(
            transitions,
            encoder_kind=args.encoder,
            hash_hidden_dim=args.hash_hidden_dim,
            policy_encoder=policy_encoder,
            batch_size=args.encode_batch_size,
            state_view=args.state_view,
            belief_max_events=args.belief_max_events,
        )
        _validate_next_state_supervision(cached_hidden, transitions)
        cache_metadata = _save_hidden_cache(
            output_dir / "hidden_cache.pt",
            hidden=cached_hidden,
            transitions=transitions,
            input_records_sha256=input_records_sha256,
            encoder_config=_encoder_config(args, final_backbone=True),
        )
        # Keep the large optional backbone export after the cache has been
        # verified and persisted; a failed cache export should not leave an
        # unusable multi-gigabyte directory behind.
        updated_llm = output_dir / "updated_llm"
        policy_encoder.model.save_pretrained(updated_llm)
        policy_encoder.tokenizer.save_pretrained(updated_llm)
    if cached_hidden is None or cache_metadata is None:
        raise RuntimeError("verified hidden cache was not produced")

    final = history[-1]
    selected = history[best_epoch - 1] if args.checkpoint_selection == "best_validation" else final
    selected["val_metrics"].update(
        _partition_latent_diagnostics(
            model=model,
            transitions=transitions,
            indices=val_indices or train_indices,
            cached_hidden=cached_hidden,
            batch_size=args.batch_size,
            device=device,
        )
    )
    loss_curve = _loss_curve_summary(history)
    value_head_parameter_delta_l2 = _parameter_delta_l2(
        value_parameters,
        initial_value_parameters,
    )
    metadata = {
        "schema_version": "openclaw_terminal_latent_wm_checkpoint_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).expanduser()),
        "records_path": str(records_path),
        "records_sha256": input_records_sha256,
        "cache_metadata": cache_metadata,
        "record_count": len(transitions),
        "has_reward": train_reward_label_count > 0,
        "train_count": len(train_indices),
        "train_reward_label_count": train_reward_label_count,
        "val_count": len(val_indices),
        "split": split_metadata,
        "optimizer_step_count": optimizer_step_count,
        "completed_epochs": len(history),
        "checkpoint_selection": args.checkpoint_selection,
        "best_epoch": best_epoch,
        "best_selection_loss": best_selection_loss,
        "cumulative_train_seconds": cumulative_train_seconds,
        "minimum_train_seconds_satisfied": cumulative_train_seconds >= args.min_train_seconds,
        "backbone_parameter_delta_l2": backbone_parameter_delta_l2,
        "backbone_total_parameter_count": backbone_total_parameter_count,
        "backbone_trainable_parameter_count": backbone_trainable_parameter_count,
        "value_update_step_count": value_update_step_count,
        "value_head_parameter_delta_l2": value_head_parameter_delta_l2,
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "selected_epoch": selected["epoch"],
        "selected_train_loss": selected["train_loss"],
        "selected_val_loss": selected["val_loss"],
        "loss_curve": loss_curve,
        "value_labels_verified_execution_outcomes": verified_value_labels,
        "train_value_label_contract": train_value_contract,
        "val_value_label_contract": val_value_contract,
        "diagnostic_only": bool(
            args.encoder == "hash"
            or args.allow_unverified_replay
            or split_metadata.get("strategy") != "group_holdout"
            or not split_metadata.get("confirmatory_group_key", False)
            or (args.value_coef > 0.0 and not verified_value_labels)
        ),
        "replay_stats": replay_stats,
        "hidden_cache_input": (
            str(Path(args.hidden_cache_input).expanduser()) if args.hidden_cache_input else None
        ),
        "hyperparameters": {
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "microbatch_queue_size": args.microbatch_queue_size,
            "checkpoint_selection": args.checkpoint_selection,
            "epochs": args.epochs,
            "max_epochs": args.max_epochs,
            "min_train_seconds": args.min_train_seconds,
            "validation_interval_epochs": args.validation_interval_epochs,
            "train_batches_per_epoch": args.train_batches_per_epoch,
            "validation_batches_per_epoch": args.validation_batches_per_epoch,
            "lr": args.lr,
            "llm_lr": args.llm_lr,
            "llm_train_mode": args.llm_train_mode,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "sigreg_coef": args.sigreg_coef,
            "sigreg_scope": args.sigreg_scope,
            "prediction_target": args.prediction_target,
            "predictor_input_mode": args.predictor_input_mode,
            "prediction_form": args.prediction_form,
            "pred_loss_type": args.pred_loss_type,
            "objective_population": (
                "has_next" if args.prediction_target == "next_state" else args.objective_population
            ),
            "action_contrast_coef": args.action_contrast_coef,
            "alignment_coef": args.alignment_coef,
            "feedback_aux_coef": args.feedback_aux_coef,
            "value_coef": args.value_coef,
            "target_ema_decay": args.target_ema_decay,
            "target_geometry": args.target_geometry,
            "fixed_target_seed": args.fixed_target_seed,
            "val_ratio": args.val_ratio,
            "split_group_key": args.split_group_key,
            "split_seed": split_seed,
            "split_manifest": (
                str(args.split_manifest.expanduser().resolve())
                if args.split_manifest is not None
                else None
            ),
            "seed": args.seed,
        },
    }
    checkpoint = {
        "config": config.__dict__,
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "history": history,
        "runtime": vars(args),
    }
    torch.save(checkpoint, output_dir / "latent_world_model.pt")
    control_model = TextLatentWorldModel(config).to(device)
    control_model.load_state_dict(initial_model_state, strict=True)
    control_val_metrics = _partition_latent_diagnostics(
        model=control_model,
        transitions=transitions,
        indices=val_indices or train_indices,
        cached_hidden=cached_hidden,
        batch_size=args.batch_size,
        device=device,
    )
    control_metadata = {
        **metadata,
        "schema_version": "openclaw_terminal_latent_wm_initial_control_v1",
        "checkpoint_selection": "initialization",
        "control_for_checkpoint": str(output_dir / "latent_world_model.pt"),
        "optimizer_step_count": 0,
        "completed_epochs": 0,
        "best_epoch": 0,
        "selected_epoch": 0,
        "selected_train_loss": None,
        "selected_val_loss": None,
        "cumulative_train_seconds": 0.0,
    }
    torch.save(
        {
            "config": config.__dict__,
            "state_dict": initial_model_state,
            "metadata": control_metadata,
            "history": [],
            "final": {"val_metrics": control_val_metrics},
            "runtime": vars(args),
        },
        output_dir / "initial_latent_world_model.pt",
    )
    del control_model
    _write_predictions(
        path=output_dir / "predictions.jsonl",
        model=model,
        transitions=transitions,
        cached_hidden=cached_hidden,
        batch_size=args.batch_size,
        device=device,
    )
    summary = {
        "schema_version": "openclaw_terminal_latent_wm_run_summary_v3",
        "checkpoint": str(output_dir / "latent_world_model.pt"),
        "hidden_cache": str(output_dir / "hidden_cache.pt"),
        "records": str(records_path),
        "record_count": len(transitions),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "split": split_metadata,
        "encoder": args.encoder,
        "backprop_to_llm": args.backprop_to_llm,
        "fixed_target_backbone": args.fixed_target_backbone,
        "llm_train_mode": args.llm_train_mode,
        "checkpoint_selection": args.checkpoint_selection,
        "updated_backbone_selected_at_final_epoch": bool(
            args.backprop_to_llm and args.checkpoint_selection == "final_epoch"
        ),
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "optimizer_step_count": optimizer_step_count,
        "cumulative_train_seconds": cumulative_train_seconds,
        "backbone_parameter_delta_l2": backbone_parameter_delta_l2,
        "backbone_total_parameter_count": backbone_total_parameter_count,
        "backbone_trainable_parameter_count": backbone_trainable_parameter_count,
        "minimum_train_seconds_satisfied": cumulative_train_seconds >= args.min_train_seconds,
        "use_dapo_replay_buffer": args.use_dapo_replay_buffer,
        "hidden_cache_reused": bool(args.hidden_cache_input),
        "replay_stats": replay_stats,
        "diagnostic_only": metadata["diagnostic_only"],
        "model_config": config.__dict__,
        "hyperparameters": metadata["hyperparameters"],
        "history": history,
        "loss_curve": loss_curve,
        "final": final,
        "selected": selected,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved latent world-model outputs to {output_dir}")


if __name__ == "__main__":
    main()
