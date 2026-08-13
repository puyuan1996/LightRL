"""Fit a fold-local probe from frozen next-state JEPA latents to result latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .cache_text_hidden import validate_hidden_cache_integrity
from .checkpoint import select_evaluation_indices, validate_cache_encoder
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig
from .offline_diagnostics import (
    _feedback_key,
    _infer_latents,
    _matched_action_derangement,
    _prediction_metrics,
    _read_jsonl,
    _validate_alignment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().float().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _indices_sha256(indices: Sequence[int]) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class LatentResultProbe(nn.Module):
    """Low-capacity map from a frozen dynamics latent to feedback latent space."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.input_norm = nn.LayerNorm(latent_dim)
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.input_norm(latent.float())), dim=-1)


def _eligible(indices: Sequence[int], has_next: torch.Tensor) -> list[int]:
    return [index for index in indices if bool(has_next[index].item())]


def _batches(
    indices: Sequence[int], batch_size: int, *, seed: int, shuffle: bool
) -> list[list[int]]:
    values = list(indices)
    if shuffle:
        random.Random(seed).shuffle(values)
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return F.mse_loss(prediction, target) * prediction.shape[-1]


def _run_epoch(
    probe: LatentResultProbe,
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> float:
    probe.train(optimizer is not None)
    total = 0.0
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in _batches(
            indices, batch_size, seed=seed, shuffle=optimizer is not None
        ):
            index = torch.tensor(batch)
            prediction = probe(features[index].to(device))
            loss = _loss(prediction, targets[index].to(device))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
    return total / max(count, 1)


def _predict(
    probe: LatentResultProbe,
    features: torch.Tensor,
    indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    probe.eval()
    with torch.no_grad():
        for batch in _batches(indices, batch_size, seed=0, shuffle=False):
            index = torch.tensor(batch)
            rows.append(probe(features[index].to(device)).detach().float().cpu())
    return torch.cat(rows)


def _metrics(
    probe: LatentResultProbe,
    features: torch.Tensor,
    targets: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    index = torch.tensor(indices)
    prediction = _predict(
        probe, features, indices, device=device, batch_size=batch_size
    )
    keys = [_feedback_key(records[value]) for value in indices]
    return _prediction_metrics(prediction, targets[index], keys, device)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mrr-margin", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if min(args.epochs, args.batch_size, args.inference_batch_size) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")
    if not 0 <= args.mrr_margin < 1:
        raise ValueError("MRR margin must be in [0, 1)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    records_sha256 = _sha256(args.records)
    records = _read_jsonl(args.records)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(cache, dict):
        raise TypeError("hidden cache must be a dictionary")
    validate_hidden_cache_integrity(cache, require_verified=True)
    _validate_alignment(records, cache, records_sha256)
    cache_metadata = cache.get("metadata")
    if not isinstance(cache_metadata, Mapping):
        raise ValueError("cache metadata is missing")

    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("source checkpoint must be a dictionary")
    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("source checkpoint metadata is missing")
    validate_cache_encoder(checkpoint_metadata, cache_metadata)
    train_indices, train_scope = select_evaluation_indices(
        checkpoint_metadata, cache_metadata, count=len(records), requested_split="train"
    )
    val_indices, val_scope = select_evaluation_indices(
        checkpoint_metadata, cache_metadata, count=len(records), requested_split="val"
    )
    if not val_scope.get("group_disjoint"):
        raise ValueError("result transfer requires a group-disjoint source split")
    has_next = cache.get("has_next")
    if not isinstance(has_next, torch.Tensor) or has_next.dtype != torch.bool:
        raise ValueError("verified bool has_next tensor is required")
    train_indices = _eligible(train_indices, has_next)
    val_indices = _eligible(val_indices, has_next)
    if not train_indices or not val_indices:
        raise ValueError("source split has no eligible has-next train/validation rows")

    config_value = checkpoint.get("config")
    if not isinstance(config_value, Mapping):
        raise ValueError("source checkpoint config is missing")
    config = TextLatentWorldModelConfig(**dict(config_value))
    if config.prediction_target != "next_state":
        raise ValueError("result transfer requires a next-state source checkpoint")
    if config.target_geometry != "frozen_random_orthogonal_v1":
        raise ValueError("result transfer requires frozen_random_orthogonal_v1 geometry")
    fixed_projection = checkpoint.get("state_dict", {}).get("fixed_target_projection")
    if not isinstance(fixed_projection, torch.Tensor):
        raise ValueError("source checkpoint fixed target projection is missing")
    target_projection_sha256 = _tensor_sha256(fixed_projection)
    model = TextLatentWorldModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    source_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.to(device)
    del checkpoint

    all_indices = list(range(len(records)))
    latent = _infer_latents(
        model, cache, all_indices, device, args.inference_batch_size
    )
    features = latent["pred_latent"]
    targets = latent["target_latent"]
    if features.shape != targets.shape or features.ndim != 2:
        raise ValueError("source and result latents must be aligned rank-2 tensors")

    probe = LatentResultProbe(int(features.shape[-1])).to(device)
    probe_parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    initial = _metrics(
        probe,
        features,
        targets,
        records,
        val_indices,
        device=device,
        batch_size=args.inference_batch_size,
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            probe,
            features,
            targets,
            train_indices,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed + epoch - 1,
        )
        val_loss = _run_epoch(
            probe,
            features,
            targets,
            val_indices,
            optimizer=None,
            device=device,
            batch_size=args.batch_size,
            seed=20260723,
        )
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False))

    final = _metrics(
        probe,
        features,
        targets,
        records,
        val_indices,
        device=device,
        batch_size=args.inference_batch_size,
    )
    matched_indices, matched_action_indices, matched_audit = _matched_action_derangement(
        records, val_indices
    )
    if len(matched_indices) < 2:
        raise ValueError("matched shuffled-action control has fewer than two rows")
    matched_shuffled = _infer_latents(
        model,
        cache,
        matched_indices,
        device,
        args.inference_batch_size,
        action_indices=matched_action_indices,
    )["pred_latent"]
    matched_observed = features[torch.tensor(matched_indices)]
    matched_target = targets[torch.tensor(matched_indices)]
    matched_keys = [_feedback_key(records[index]) for index in matched_indices]
    matched_observed_prediction = _predict(
        probe,
        matched_observed,
        list(range(len(matched_indices))),
        device=device,
        batch_size=args.inference_batch_size,
    )
    matched_shuffled_prediction = _predict(
        probe,
        matched_shuffled,
        list(range(len(matched_indices))),
        device=device,
        batch_size=args.inference_batch_size,
    )
    matched = {
        "audit": matched_audit,
        "observed": _prediction_metrics(
            matched_observed_prediction, matched_target, matched_keys, device
        ),
        "shuffled_action": _prediction_metrics(
            matched_shuffled_prediction, matched_target, matched_keys, device
        ),
    }
    del model

    metrics_finite = all(
        math.isfinite(float(value))
        for metric in (initial, final, matched["observed"], matched["shuffled_action"])
        for key, value in metric.items()
        if key in {
            "paired_mse",
            "paired_cosine_similarity",
            "paired_cosine_distance",
            "mrr",
            "top1_accuracy",
            "recall_at_5",
        }
    )
    gates = {
        "validation_loss_reduced_by_10pct": (
            history[-1]["val_loss"] <= history[0]["val_loss"] * 0.9
        ),
        "trained_probe_mrr_beats_initial_by_margin": (
            final["mrr"] >= initial["mrr"] + args.mrr_margin
        ),
        "metrics_finite": metrics_finite,
        "group_disjoint_source_split": True,
        "fold_local_cross_fit": True,
    }
    summary = {
        "schema_version": "openclaw_jepa_result_transfer_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "prediction_target": "observational_feedback_latent",
        "source_prediction_target": "next_state",
        "population": "has_next",
        "seed": args.seed,
        "latent_dim": int(features.shape[-1]),
        "source_parameter_count": source_parameter_count,
        "probe_parameter_count": probe_parameter_count,
        "total_parameter_count": source_parameter_count + probe_parameter_count,
        "records_sha256": records_sha256,
        "cache_fingerprint_sha256": cache_metadata.get("cache_fingerprint_sha256"),
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "target_projection_sha256": target_projection_sha256,
        "source_config": dict(config_value),
        "train_scope": train_scope,
        "val_scope": val_scope,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "train_indices_sha256": _indices_sha256(train_indices),
        "val_indices_sha256": _indices_sha256(val_indices),
        "source_fold_index": checkpoint_metadata.get("split", {}).get("fold_index"),
        "val_result_key_count": len(
            {_feedback_key(records[index]) for index in val_indices}
        ),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "mrr_margin": args.mrr_margin,
        },
        "history": history,
        "initial_diagnostics": initial,
        "final_diagnostics": final,
        "matched_action_control": matched,
        "gates": gates,
        "mechanically_complete": (
            gates["metrics_finite"]
            and gates["group_disjoint_source_split"]
            and gates["fold_local_cross_fit"]
        ),
        "probe_optimization_gate_passed": (
            gates["validation_loss_reduced_by_10pct"]
            and gates["trained_probe_mrr_beats_initial_by_margin"]
        ),
        "claim_boundary": (
            "Fold-local transfer from a frozen next-state JEPA prediction to an "
            "observed feedback-text latent on has-next SETA records. The records do not "
            "contain verified atomic execution status, same-snapshot alternatives, or "
            "an external task-cluster test."
        ),
    }
    torch.save(
        {
            "schema_version": "openclaw_jepa_result_transfer_checkpoint_v1",
            "probe_config": {"latent_dim": int(features.shape[-1])},
            "state_dict": probe.state_dict(),
            "summary": summary,
        },
        args.output_dir / "result_transfer_probe.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved result-transfer probe to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
