"""Create a provenance-preserving subset of verified world-model records/cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .cache_text_hidden import (
    _build_cache_integrity_metadata,
    validate_hidden_cache_integrity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"record line {line_number} is not an object")
            rows.append(value)
            lines.append(line if line.endswith("\n") else line + "\n")
    if not rows:
        raise ValueError("records are empty")
    return rows, lines


def _atomic_write_lines(path: Path, lines: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
    temporary.replace(path)


def _canonicalize_encoder_config(
    encoder_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical = dict(encoder_config)
    migrations: list[dict[str, Any]] = []
    if canonical.get("encoder") != "hf-policy":
        return canonical, migrations
    for field, default in (
        ("fixed_target_backbone", False),
        ("fixed_target_model_name_or_path", None),
    ):
        if field not in canonical:
            canonical[field] = default
            migrations.append(
                {
                    "field": field,
                    "value": default,
                    "reason": "legacy_missing_field_uses_current_default",
                }
            )
    return canonical, migrations


def build_single_call_subset(
    records_path: Path,
    cache_path: Path,
    output_dir: Path,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    records_path = records_path.expanduser().resolve()
    cache_path = cache_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    records, source_lines = _read_records(records_path)
    source_records_sha256 = _sha256(records_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    validate_hidden_cache_integrity(payload, require_verified=True)
    count = len(records)
    if int(payload.get("record_count", -1)) != count:
        raise ValueError("records/cache count mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("cache metadata is missing")
    if metadata.get("input_records_sha256") != source_records_sha256:
        raise ValueError("records/cache source digest mismatch")

    indices = [
        index
        for index, row in enumerate(records)
        if isinstance(row.get("tool_names"), list) and len(row["tool_names"]) == 1
    ]
    if not indices:
        raise ValueError("no single-call records were found")
    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(
            f"single-call count mismatch: expected={expected_count} actual={len(indices)}"
        )

    output_dir.mkdir(parents=True)
    output_records = output_dir / "result_only_records.jsonl"
    _atomic_write_lines(output_records, [source_lines[index] for index in indices])
    output_records_sha256 = _sha256(output_records)

    index_tensor = torch.tensor(indices, dtype=torch.long)
    subset: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == count:
            subset[key] = value[index_tensor].contiguous()
        elif key == "record_metadata" and isinstance(value, list) and len(value) == count:
            subset[key] = [value[index] for index in indices]
        elif key != "metadata":
            subset[key] = value
    subset["record_count"] = len(indices)
    source_encoder_config = metadata.get("encoder_config")
    if not isinstance(source_encoder_config, dict):
        raise ValueError("cache encoder_config is missing")
    encoder_config, encoder_config_migrations = _canonicalize_encoder_config(
        source_encoder_config
    )
    subset_metadata = _build_cache_integrity_metadata(
        subset,
        input_records_sha256=output_records_sha256,
        encoder_config=encoder_config,
    )
    subset_metadata["subset_provenance"] = {
        "filter": "single_call_tool_names_v1",
        "source_cache_fingerprint_sha256": metadata.get(
            "cache_fingerprint_sha256"
        ),
        "source_records_sha256": source_records_sha256,
        "source_record_count": count,
        "selected_record_count": len(indices),
        "selected_indices_sha256": hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "encoder_config_migrations": encoder_config_migrations,
    }
    subset["metadata"] = subset_metadata
    validate_hidden_cache_integrity(subset, require_verified=True)

    output_cache = output_dir.parent / "hierarchical_hidden_cache.pt"
    temporary_cache = output_cache.with_suffix(output_cache.suffix + ".tmp")
    torch.save(subset, temporary_cache)
    temporary_cache.replace(output_cache)
    reloaded = torch.load(
        output_cache, map_location="cpu", weights_only=True, mmap=True
    )
    validate_hidden_cache_integrity(reloaded, require_verified=True)

    manifest = {
        "schema_version": "openclaw_verified_hidden_cache_subset_v1",
        "filter": "single_call_tool_names_v1",
        "source": {
            "records": str(records_path),
            "records_sha256": source_records_sha256,
            "cache": str(cache_path),
            "cache_fingerprint_sha256": metadata.get(
                "cache_fingerprint_sha256"
            ),
            "record_count": count,
        },
        "output": {
            "records": str(output_records),
            "records_sha256": output_records_sha256,
            "cache": str(output_cache),
            "cache_fingerprint_sha256": subset_metadata[
                "cache_fingerprint_sha256"
            ],
            "record_count": len(indices),
        },
        "claim_boundary": (
            "Single-call observational turn subset only; this does not create "
            "verified atomic execution labels or counterfactual candidate outcomes."
        ),
    }
    manifest_path = output_dir.parent / "subset_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    manifest = build_single_call_subset(
        args.records,
        args.cache,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(manifest["output"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
