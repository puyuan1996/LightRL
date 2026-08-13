"""Build and merge verified policy-hidden caches across independent GPU shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from .cache_text_hidden import (
    _build_cache_integrity_metadata,
    validate_hidden_cache_integrity,
)
from .hidden_encoder import PolicyHiddenEncoder
from .action_view import ACTION_VIEW_SCHEMA, parse_tool_call_bundle
from .result_view import RESULT_VIEW_SCHEMA, parse_result_only_view
from .seta_dataset import load_terminal_transitions
from .state_view import (
    BELIEF_VIEW_ALLOWLIST,
    BELIEF_VIEW_MAX_CHARS,
    BELIEF_VIEW_POOLING,
    BELIEF_VIEW_V1,
)
from .train_latent import _cache_hidden, _save_hidden_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def split_records(input_path: Path, output_dir: Path, shard_count: int) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing shard directory: {output_dir}")
    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError("input records are empty")
    for line_number, line in enumerate(lines, 1):
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("transition_id"):
            raise ValueError(f"record line {line_number} lacks transition_id")
    output_dir.mkdir(parents=True)
    shards: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        start = len(lines) * shard_index // shard_count
        end = len(lines) * (shard_index + 1) // shard_count
        path = output_dir / f"records_shard_{shard_index:02d}.jsonl"
        path.write_text("\n".join(lines[start:end]) + "\n", encoding="utf-8")
        shards.append(
            {
                "index": shard_index,
                "start": start,
                "end": end,
                "count": end - start,
                "records": str(path),
                "records_sha256": _sha256(path),
                "cache": str(output_dir / f"hidden_cache_shard_{shard_index:02d}.pt"),
            }
        )
    manifest = {
        "schema_version": "openclaw_hidden_cache_shards_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path),
            "count": len(lines),
            "sha256": _sha256(input_path),
        },
        "shard_count": shard_count,
        "shards": shards,
    }
    _write_json(output_dir / "shard_manifest.json", manifest)
    return manifest


def validate_canonical_records(
    records_path: Path,
    output_path: Path,
    *,
    expected_count: int | None = None,
    expected_calls: int | None = None,
) -> dict[str, Any]:
    transitions = load_terminal_transitions(
        records_path,
        require_tool_feedback=True,
        max_text_chars=sys.maxsize,
    )
    calls = 0
    results = 0
    with records_path.open(encoding="utf-8") as handle:
        source_rows = (
            json.loads(line) for line in handle if line.strip()
        )
        for index, (source, transition) in enumerate(
            zip_longest(source_rows, transitions),
            1,
        ):
            if source is None or transition is None:
                raise ValueError("canonical loader changed the record count")
            if source.get("action_view_schema") != ACTION_VIEW_SCHEMA:
                raise ValueError(
                    f"record {index} lacks canonical action-view provenance"
                )
            if source.get("feedback_source") != RESULT_VIEW_SCHEMA:
                raise ValueError(
                    f"record {index} lacks canonical result-view provenance"
                )
            calls += len(parse_tool_call_bundle(transition.action_text))
            results += len(parse_result_only_view(transition.feedback_text))
            serialized = transition.to_dict()
            for key in (
                "action_text",
                "next_observation_text",
                "transition_id",
            ):
                if serialized.get(key) != source.get(key):
                    raise ValueError(
                        f"canonical loader round-trip changed {key} at record {index}"
                    )
    if calls != results:
        raise ValueError("canonical action/result block counts differ")
    if expected_count is not None and len(transitions) != expected_count:
        raise ValueError(
            f"canonical record count mismatch: {len(transitions)} != {expected_count}"
        )
    if expected_calls is not None and calls != expected_calls:
        raise ValueError(
            f"canonical call count mismatch: {calls} != {expected_calls}"
        )
    summary = {
        "schema_version": "openclaw_canonical_loader_audit_v1",
        "records": len(transitions),
        "calls": calls,
        "results": results,
        "action_view_schema": ACTION_VIEW_SCHEMA,
        "feedback_source": RESULT_VIEW_SCHEMA,
        "roundtrip_mismatch_count": 0,
        "records_sha256": _sha256(records_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, summary)
    return summary


def _encoder_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "encoder": "hf-policy",
        "model_name_or_path": args.hf_model,
        "dtype": args.dtype,
        "hidden_layer": args.hidden_layer,
        "action_pool": args.action_pool,
        "max_context_tokens": args.max_context_tokens,
        "max_action_tokens": args.max_action_tokens,
        "max_feedback_tokens": args.max_feedback_tokens,
        "encoder_long_text_mode": args.encoder_long_text_mode,
        "chunk_forward_batch_size": args.chunk_forward_batch_size,
        "strict_action_boundary": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "backbone_updated": False,
        "schema": "openclaw_policy_hidden_encoder_v2",
    }
    if args.state_view == BELIEF_VIEW_V1:
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


def wait_for_start_barrier(
    ready_file: Path,
    start_file: Path,
    *,
    timeout_seconds: float,
) -> None:
    if timeout_seconds <= 0:
        raise ValueError("barrier timeout must be positive")
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready_file.with_suffix(
        ready_file.suffix + f".{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "ready_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(ready_file)
    deadline = time.monotonic() + timeout_seconds
    while not start_file.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for encoder start barrier: {start_file}"
            )
        time.sleep(min(1.0, max(0.01, timeout_seconds / 10.0)))


def encode_shard(args: argparse.Namespace) -> None:
    transitions = load_terminal_transitions(
        args.records,
        require_tool_feedback=True,
        max_text_chars=sys.maxsize,
    )
    if not transitions:
        raise ValueError("shard contains no transitions")
    encoder = PolicyHiddenEncoder.from_pretrained(
        args.hf_model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
        trust_remote_code=False,
        hidden_layer=args.hidden_layer,
        action_pool=args.action_pool,
        max_context_tokens=args.max_context_tokens,
        max_action_tokens=args.max_action_tokens,
        max_feedback_tokens=args.max_feedback_tokens,
        backprop_to_llm=False,
        strict_action_boundary=True,
        state_view=args.state_view,
        belief_max_events=args.belief_max_events,
        encoder_long_text_mode=args.encoder_long_text_mode,
        chunk_forward_batch_size=args.chunk_forward_batch_size,
    )
    if (args.ready_file is None) != (args.start_file is None):
        raise ValueError("--ready-file and --start-file must be provided together")
    if args.ready_file is not None and args.start_file is not None:
        wait_for_start_barrier(
            args.ready_file,
            args.start_file,
            timeout_seconds=args.barrier_timeout_seconds,
        )
    hidden = _cache_hidden(
        transitions,
        encoder_kind="hf-policy",
        hash_hidden_dim=0,
        policy_encoder=encoder,
        batch_size=args.encode_batch_size,
        state_view=args.state_view,
        belief_max_events=args.belief_max_events,
    )
    if args.compute_done_file is not None:
        args.compute_done_file.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.compute_done_file,
            {
                "pid": os.getpid(),
                "compute_done_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(transitions),
            },
        )
    metadata = _save_hidden_cache(
        args.output,
        hidden=hidden,
        transitions=transitions,
        input_records_sha256=_sha256(args.records),
        encoder_config=_encoder_config(args),
    )
    print(
        json.dumps(
            {
                "records": len(transitions),
                "cache": str(args.output),
                "cache_fingerprint_sha256": metadata[
                    "cache_fingerprint_sha256"
                ],
            },
            sort_keys=True,
        )
    )


def merge_shards(
    manifest_path: Path,
    full_records: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "openclaw_hidden_cache_shards_v1":
        raise ValueError("unsupported shard manifest")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("shard manifest source is missing")
    if source.get("sha256") != _sha256(full_records):
        raise ValueError("full records digest does not match shard manifest")

    tensor_keys = (
        "state_hidden",
        "action_hidden",
        "target_hidden",
        "next_state_hidden",
        "has_next",
        "reward",
        "reward_mask",
    )
    tensors: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
    record_metadata: list[dict[str, Any]] = []
    encoder_config: dict[str, Any] | None = None
    for shard in manifest.get("shards") or []:
        cache_path = Path(str(shard["cache"]))
        payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
        validate_hidden_cache_integrity(payload, require_verified=True)
        if int(payload["record_count"]) != int(shard["count"]):
            raise ValueError("shard cache count mismatch")
        current_config = payload["metadata"]["encoder_config"]
        if encoder_config is None:
            encoder_config = dict(current_config)
        elif current_config != encoder_config:
            raise ValueError("shard encoder configs differ")
        for key in tensor_keys:
            value = payload.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"shard cache tensor {key!r} is missing")
            tensors[key].append(value)
        current_records = payload.get("record_metadata")
        if not isinstance(current_records, list):
            raise ValueError("shard record metadata is missing")
        record_metadata.extend(current_records)
    if encoder_config is None:
        raise ValueError("shard manifest contains no caches")

    expected_ids: list[str] = []
    with full_records.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected_ids.append(str(json.loads(line).get("transition_id")))
    actual_ids = [str(row.get("transition_id")) for row in record_metadata]
    if actual_ids != expected_ids:
        raise ValueError("merged shard transition order differs from full records")
    payload = {
        key: torch.cat(values, dim=0) for key, values in tensors.items()
    }
    payload.update(
        {
            "record_count": len(record_metadata),
            "record_metadata": record_metadata,
        }
    )
    payload["metadata"] = {
        **_build_cache_integrity_metadata(
            payload,
            input_records_sha256=_sha256(full_records),
            encoder_config=encoder_config,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shard_manifest": str(manifest_path),
        "shard_count": int(manifest["shard_count"]),
    }
    validate_hidden_cache_integrity(payload, require_verified=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    summary = {
        "schema_version": "openclaw_merged_hidden_cache_v1",
        "record_count": len(record_metadata),
        "cache": str(output),
        "cache_sha256": _sha256(output),
        "cache_fingerprint_sha256": payload["metadata"][
            "cache_fingerprint_sha256"
        ],
        "encoder_config": encoder_config,
    }
    _write_json(output.with_suffix(".summary.json"), summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--shards", type=int, default=4)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--records", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--expected-count", type=int)
    validate.add_argument("--expected-calls", type=int)

    encode = subparsers.add_parser("encode")
    encode.add_argument("--records", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--hf-model", required=True)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--dtype", default="bfloat16")
    encode.add_argument("--hidden-layer", type=int, default=-1)
    encode.add_argument("--action-pool", choices=["mean", "last"], default="mean")
    encode.add_argument("--max-context-tokens", type=int, default=1536)
    encode.add_argument("--max-action-tokens", type=int, default=512)
    encode.add_argument("--max-feedback-tokens", type=int, default=512)
    encode.add_argument("--state-view", default="belief_view_v1")
    encode.add_argument("--belief-max-events", type=int, default=3)
    encode.add_argument(
        "--encoder-long-text-mode",
        choices=["tail_v1", "hierarchical_chunks_v1"],
        default="hierarchical_chunks_v1",
    )
    encode.add_argument("--chunk-forward-batch-size", type=int, default=16)
    encode.add_argument("--encode-batch-size", type=int, default=4)
    encode.add_argument("--ready-file", type=Path)
    encode.add_argument("--start-file", type=Path)
    encode.add_argument("--compute-done-file", type=Path)
    encode.add_argument("--barrier-timeout-seconds", type=float, default=1800.0)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--records", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "split":
        print(
            json.dumps(
                split_records(args.input, args.output_dir, args.shards),
                sort_keys=True,
            )
        )
    elif args.command == "validate":
        print(
            json.dumps(
                validate_canonical_records(
                    args.records,
                    args.output,
                    expected_count=args.expected_count,
                    expected_calls=args.expected_calls,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "encode":
        encode_shard(args)
    else:
        print(
            json.dumps(
                merge_shards(args.manifest, args.records, args.output),
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
