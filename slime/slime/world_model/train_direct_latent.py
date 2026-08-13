"""Train a parameter-matched raw-hidden target predictor on a frozen cache."""

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
from .metrics import effective_rank
from .modules import TextLatentWorldModel, TextLatentWorldModelConfig
from .offline_diagnostics import _prediction_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_sha256(indices: Sequence[int]) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_record_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"record line {line_number} is not an object")
            rows.append(
                {
                    "transition_id": value.get("transition_id"),
                    "has_next": value.get("has_next"),
                    "next_context_hash": value.get("next_context_hash"),
                    "next_observation_hash": value.get("next_observation_hash"),
                }
            )
    if not rows:
        raise ValueError("records JSONL is empty")
    return rows


def _nearest_hidden_width(
    hidden_dim: int,
    target_parameters: int,
    output_dim: int | None = None,
) -> int:
    output_dim = int(output_dim or hidden_dim)
    if min(hidden_dim, output_dim, target_parameters) <= 0:
        raise ValueError("hidden_dim, output_dim, and target_parameters must be positive")
    constant = 4 * hidden_dim + 3 * output_dim
    coefficient = 2 * hidden_dim + output_dim + 1
    estimate = max(1, round((target_parameters - constant) / coefficient))
    candidates = range(max(1, estimate - 2), estimate + 3)
    return min(
        candidates,
        key=lambda width: abs(
            _parameter_count_formula(hidden_dim, width, output_dim)
            - target_parameters
        ),
    )


def _parameter_count_formula(
    hidden_dim: int,
    width: int,
    output_dim: int | None = None,
) -> int:
    output_dim = int(output_dim or hidden_dim)
    return (
        (2 * hidden_dim + output_dim + 1) * width
        + 4 * hidden_dim
        + 3 * output_dim
    )


class RawHiddenDirectPredictor(nn.Module):
    """Direct baseline from frozen state/action hidden to raw next-state hidden."""

    def __init__(
        self,
        hidden_dim: int,
        width: int,
        *,
        output_dim: int | None = None,
        input_mode: str = "observed",
    ) -> None:
        super().__init__()
        if input_mode not in {"observed", "state_only", "action_only"}:
            raise ValueError(f"unknown input_mode: {input_mode}")
        self.input_mode = input_mode
        output_dim = int(output_dim or hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, width),
            nn.GELU(),
            nn.Linear(width, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, state_hidden: torch.Tensor, action_hidden: torch.Tensor) -> torch.Tensor:
        state = state_hidden.float()
        action = action_hidden.float()
        if self.input_mode == "state_only":
            action = torch.zeros_like(action)
        elif self.input_mode == "action_only":
            state = torch.zeros_like(state)
        prediction = self.trunk(
            torch.cat([self.state_norm(state), self.action_norm(action)], dim=-1)
        )
        return F.normalize(prediction, dim=-1)


def _batches(
    indices: Sequence[int],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[list[int]]:
    values = list(indices)
    if shuffle:
        random.Random(seed).shuffle(values)
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def _eligible(indices: Sequence[int], has_next: torch.Tensor) -> list[int]:
    return [index for index in indices if bool(has_next[index].item())]


def _target_keys(
    cache: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    prediction_target: str,
) -> list[str]:
    if prediction_target == "feedback":
        keys: list[str] = []
        for index in indices:
            value = records[index].get("next_observation_hash")
            if not value:
                raise ValueError(f"record {index} lacks a feedback equivalence key")
            keys.append(str(value))
        return keys
    if prediction_target != "next_state":
        raise ValueError(f"unknown prediction_target: {prediction_target}")
    cached = cache.get("record_metadata")
    if not isinstance(cached, list) or len(cached) != len(records):
        raise ValueError("cache record_metadata is missing or misaligned")
    keys: list[str] = []
    for index in indices:
        value = cached[index].get("next_state_view_hash")
        value = value or records[index].get("next_context_hash")
        if not value:
            raise ValueError(f"record {index} lacks a next-state equivalence key")
        keys.append(str(value))
    return keys


@torch.no_grad()
def _infer(
    model: RawHiddenDirectPredictor,
    cache: Mapping[str, Any],
    indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    shuffle_actions: bool = False,
    zero_actions: bool = False,
) -> torch.Tensor:
    if shuffle_actions and zero_actions:
        raise ValueError("shuffle_actions and zero_actions are mutually exclusive")
    action_indices = list(indices)
    if shuffle_actions:
        generator = torch.Generator().manual_seed(20260723)
        shift = int(torch.randint(1, len(indices), (1,), generator=generator).item())
        action_indices = action_indices[shift:] + action_indices[:shift]
    rows: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch = list(indices[start : start + batch_size])
        action_batch = action_indices[start : start + batch_size]
        state = cache["state_hidden"][batch].to(device)
        action = cache["action_hidden"][action_batch].to(device)
        if zero_actions:
            action = torch.zeros_like(action)
        rows.append(model(state, action).cpu())
    return torch.cat(rows)


def _geometry(value: torch.Tensor) -> dict[str, float]:
    normalized = F.normalize(value.float(), dim=-1)
    variance = value.float().var(dim=0, unbiased=False)
    count = value.shape[0]
    pairwise = (
        normalized.sum(dim=0).square().sum() - float(count)
    ) / max(count * (count - 1), 1)
    return {
        "effective_rank": float(effective_rank(value.float())),
        "variance_mean": float(variance.mean()),
        "variance_min": float(variance.min()),
        "pairwise_cosine": float(pairwise),
    }


def _map_target_geometry(
    hidden: torch.Tensor,
    projection: torch.Tensor | None,
) -> torch.Tensor:
    hidden = hidden.float()
    if projection is None:
        return F.normalize(hidden, dim=-1)
    normalized = F.layer_norm(hidden, (hidden.size(-1),))
    return F.normalize(normalized @ projection.to(normalized), dim=-1)


@torch.no_grad()
def _evaluate(
    model: RawHiddenDirectPredictor,
    cache: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    prediction_target: str,
    target_projection: torch.Tensor | None,
) -> dict[str, Any]:
    target_tensor = (
        cache["target_hidden"]
        if prediction_target == "feedback"
        else cache["next_state_hidden"]
    )
    target = _map_target_geometry(
        target_tensor[list(indices)],
        target_projection,
    )
    state = _map_target_geometry(
        cache["state_hidden"][list(indices)],
        target_projection,
    )
    keys = _target_keys(
        cache,
        records,
        indices,
        prediction_target=prediction_target,
    )
    prediction = _infer(model, cache, indices, device=device, batch_size=batch_size)
    shuffled = _infer(
        model,
        cache,
        indices,
        device=device,
        batch_size=batch_size,
        shuffle_actions=True,
    )
    zero = _infer(
        model,
        cache,
        indices,
        device=device,
        batch_size=batch_size,
        zero_actions=True,
    )
    return {
        "query_count": len(indices),
        "prediction": _prediction_metrics(prediction, target, keys, device),
        "state_identity": _prediction_metrics(state, target, keys, device),
        "shuffled_action": _prediction_metrics(shuffled, target, keys, device),
        "zero_action": _prediction_metrics(zero, target, keys, device),
        "prediction_geometry": _geometry(prediction),
        "target_geometry": _geometry(target),
    }


def _run_epoch(
    model: RawHiddenDirectPredictor,
    cache: Mapping[str, Any],
    indices: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    seed: int,
    prediction_target: str,
    target_projection: torch.Tensor | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in _batches(indices, batch_size, shuffle=training, seed=seed):
            state = cache["state_hidden"][batch].to(device)
            action = cache["action_hidden"][batch].to(device)
            target_name = (
                "target_hidden"
                if prediction_target == "feedback"
                else "next_state_hidden"
            )
            target = _map_target_geometry(
                cache[target_name][batch].to(device),
                target_projection,
            )
            prediction = model(state, action)
            loss = F.mse_loss(prediction, target) * prediction.shape[-1]
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite direct baseline loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
    return total / max(count, 1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=["observed", "state_only", "action_only"],
        default="observed",
    )
    parser.add_argument(
        "--prediction-target",
        choices=["feedback", "next_state"],
        default="next_state",
    )
    parser.add_argument(
        "--population",
        choices=["auto", "all", "has_next"],
        default="auto",
        help=(
            "Rows eligible for fitting and evaluation. auto preserves the legacy "
            "behavior: has_next for next_state and all rows for feedback."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mrr-margin", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and non-negative")
    if not 0.0 <= args.mrr_margin < 1.0:
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
    records = _read_record_metadata(args.records)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    validate_hidden_cache_integrity(cache, require_verified=True)
    cache_metadata = cache.get("metadata")
    if not isinstance(cache_metadata, Mapping):
        raise ValueError("cache metadata is missing")
    if cache_metadata.get("input_records_sha256") != records_sha256:
        raise ValueError("records/cache digest mismatch")
    if int(cache.get("record_count", -1)) != len(records):
        raise ValueError("records/cache count mismatch")
    has_next = cache.get("has_next")
    target_hidden = cache.get(
        "target_hidden" if args.prediction_target == "feedback" else "next_state_hidden"
    )
    population = args.population
    if population == "auto":
        population = "has_next" if args.prediction_target == "next_state" else "all"
    if args.prediction_target == "next_state" or population == "has_next":
        if not isinstance(has_next, torch.Tensor) or has_next.dtype != torch.bool:
            raise ValueError("verified bool has_next tensor is required")
    if not isinstance(target_hidden, torch.Tensor) or target_hidden.ndim != 2:
        raise ValueError(
            f"aligned rank-2 target tensor is required for {args.prediction_target}"
        )

    reference = torch.load(
        args.reference_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    metadata = reference.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("reference checkpoint metadata is missing")
    validate_cache_encoder(metadata, cache_metadata)
    reference_config = reference.get("config")
    if not isinstance(reference_config, Mapping):
        raise ValueError("reference checkpoint config is missing")
    target_geometry = str(
        reference_config.get("target_geometry", "learned_shared_v2")
    )
    target_projection: torch.Tensor | None = None
    if target_geometry == "frozen_random_orthogonal_v1":
        value = reference.get("state_dict", {}).get("fixed_target_projection")
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError("reference frozen target projection is missing")
        target_projection = value.float()
    elif target_geometry != "learned_shared_v2":
        raise ValueError(f"unsupported reference target geometry: {target_geometry}")
    train_indices, train_scope = select_evaluation_indices(
        metadata,
        cache_metadata,
        count=len(records),
        requested_split="train",
    )
    val_indices, val_scope = select_evaluation_indices(
        metadata,
        cache_metadata,
        count=len(records),
        requested_split="val",
    )
    if population == "has_next":
        train_indices = _eligible(train_indices, has_next)
        val_indices = _eligible(val_indices, has_next)
    if not train_indices or not val_indices:
        raise ValueError(
            f"reference split has no eligible {args.prediction_target} train/validation rows"
        )

    state_hidden = cache.get("state_hidden")
    action_hidden = cache.get("action_hidden")
    if not isinstance(state_hidden, torch.Tensor) or not isinstance(action_hidden, torch.Tensor):
        raise ValueError("state/action hidden tensors are missing")
    hidden_dim = int(state_hidden.shape[-1])
    if action_hidden.shape[-1] != hidden_dim or target_hidden.shape[-1] != hidden_dim:
        raise ValueError("raw direct baseline requires equal state/action/target hidden dimensions")
    reference_model = TextLatentWorldModel(
        TextLatentWorldModelConfig(**dict(reference_config))
    )
    reference_model.load_state_dict(reference["state_dict"], strict=True)
    target_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )
    del reference_model
    output_dim = (
        int(target_projection.shape[1])
        if target_projection is not None
        else hidden_dim
    )
    width = _nearest_hidden_width(
        hidden_dim,
        target_parameters,
        output_dim=output_dim,
    )
    model = RawHiddenDirectPredictor(
        hidden_dim,
        width,
        output_dim=output_dim,
        input_mode=args.input_mode,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_ratio = parameter_count / target_parameters
    if not 0.95 <= parameter_ratio <= 1.05:
        raise ValueError(
            f"direct/reference parameter ratio {parameter_ratio:.6f} is outside [0.95, 1.05]"
        )

    initial = _evaluate(
        model,
        cache,
        records,
        val_indices,
        device=device,
        batch_size=args.batch_size,
        prediction_target=args.prediction_target,
        target_projection=target_projection,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            model,
            cache,
            train_indices,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed + epoch - 1,
            prediction_target=args.prediction_target,
            target_projection=target_projection,
        )
        val_loss = _run_epoch(
            model,
            cache,
            val_indices,
            optimizer=None,
            device=device,
            batch_size=args.batch_size,
            seed=20260723,
            prediction_target=args.prediction_target,
            target_projection=target_projection,
        )
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False))

    final = _evaluate(
        model,
        cache,
        records,
        val_indices,
        device=device,
        batch_size=args.batch_size,
        prediction_target=args.prediction_target,
        target_projection=target_projection,
    )
    prediction = final["prediction"]
    identity = final["state_identity"]
    shuffled = final["shuffled_action"]
    zero = final["zero_action"]
    gates = {
        "validation_loss_reduced_by_10pct": (
            history[-1]["val_loss"] <= history[0]["val_loss"] * 0.9
        ),
        "mrr_beats_state_identity_by_margin": (
            prediction["mrr"] >= identity["mrr"] + args.mrr_margin
        ),
        "mrr_beats_shuffled_action_by_margin": (
            prediction["mrr"] >= shuffled["mrr"] + args.mrr_margin
        ),
        "mrr_beats_zero_action_by_margin": (
            prediction["mrr"] >= zero["mrr"] + args.mrr_margin
        ),
        "cosine_beats_state_identity": (
            prediction["paired_cosine_similarity"] >
            identity["paired_cosine_similarity"]
        ),
        "parameter_budget_within_5pct": 0.95 <= parameter_ratio <= 1.05,
        "metrics_finite": all(
            math.isfinite(value)
            for row in (prediction, identity, shuffled, zero)
            for value in (
                row["mrr"],
                row["top1_accuracy"],
                row["recall_at_5"],
                row["paired_cosine_similarity"],
            )
        ),
    }
    summary = {
        "schema_version": "openclaw_raw_hidden_direct_target_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "input_mode": args.input_mode,
        "prediction_target": args.prediction_target,
        "population": population,
        "target_geometry": target_geometry,
        "target_output_dim": output_dim,
        "seed": args.seed,
        "hidden_dim": hidden_dim,
        "width": width,
        "parameter_count": parameter_count,
        "reference_parameter_count": target_parameters,
        "parameter_ratio": parameter_ratio,
        "records_sha256": records_sha256,
        "cache_fingerprint_sha256": cache_metadata.get("cache_fingerprint_sha256"),
        "reference_checkpoint": str(args.reference_checkpoint),
        "reference_checkpoint_sha256": _sha256(args.reference_checkpoint),
        "train_scope": train_scope,
        "val_scope": val_scope,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "train_indices_sha256": _indices_sha256(train_indices),
        "val_indices_sha256": _indices_sha256(val_indices),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "mrr_margin": args.mrr_margin,
        },
        "history": history,
        "initial_diagnostics": initial,
        "final_diagnostics": final,
        "gates": gates,
        "dev_screen_passed": all(gates.values()),
        "claim_boundary": (
            f"Parameter-matched raw-hidden direct {args.prediction_target} diagnostic on "
            "the reused task_id-heldout development split; not an independent, atomic, "
            "counterfactual, or online execution test."
        ),
    }
    checkpoint = {
        "schema_version": "openclaw_raw_hidden_direct_checkpoint_v1",
        "model_config": {
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "width": width,
            "input_mode": args.input_mode,
            "prediction_target": args.prediction_target,
        },
        "state_dict": model.state_dict(),
        "summary": summary,
    }
    torch.save(checkpoint, args.output_dir / "direct_latent.pt")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved raw-hidden direct baseline to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
