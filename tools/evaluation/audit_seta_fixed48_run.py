#!/usr/bin/env python3
"""Fail-closed audit for formal SETA fixed48 runs and paired eval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMS = {"seta_fixed48_exploit": 1, "seta_fixed48_explore": 8}
FAIR_CONFIG_KEYS = (
    "hf_ckpt",
    "ref_load",
    "prompt_data",
    "eval_protocol",
    "eval_config",
    "eval_seed",
    "eval_steps",
    "eval_manifest_sha256",
    "eval_set_sha256",
    "train_set_sha256",
    "num_rollout",
    "rollout_batch_size",
    "n_samples",
    "max_turn",
    "rollout_max_response_len",
    "rollout_max_context_len",
    "worker_urls",
    "env_remote_max_active_tasks",
    "env_remote_max_active_runs",
    "env_remote_max_runs_per_task",
    "eval_rollout_max_concurrency",
    "num_gpus",
    "actor_gpus",
    "rollout_gpus",
    "tp_size",
    "rollout_engine_gpus",
    "save_interval",
    "max_ckpt_keep",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path}")
        return {}
    return value


def _check_hash(path: Path, expected: Any, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return
    actual = _sha256(path)
    if actual != expected:
        errors.append(f"{label} sha256 mismatch: {actual} != {expected}")


def _load_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing JSONL: {path}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL {path}:{line_number}: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"non-object JSONL row {path}:{line_number}")
            continue
        rows.append(row)
    return rows


def _close(left: Any, right: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}: expected finite numeric value, got {value!r}")
        return math.nan
    if not math.isfinite(result):
        errors.append(f"{label}: expected finite numeric value, got {value!r}")
        return math.nan
    return result


def _audit_eval_artifact(
    run_dir: Path,
    stream: str,
    step: int,
    expected_k: int,
    errors: list[str],
) -> dict[int, dict[str, Any]]:
    artifact_dir = run_dir / "evaluations" / stream / f"step_{step:04d}"
    summary = _read_json(artifact_dir / "summary.json", errors)
    tasks = _load_jsonl(artifact_dir / "tasks.jsonl", errors)
    if summary.get("dataset") != stream:
        errors.append(f"{stream} step {step}: summary dataset mismatch")
    if summary.get("global_step") != step:
        errors.append(f"{stream} step {step}: summary global_step mismatch")
    if summary.get("eval/task_count") != 48:
        errors.append(f"{stream} step {step}: task_count={summary.get('eval/task_count')} != 48")
    if summary.get("eval/k") != expected_k:
        errors.append(f"{stream} step {step}: summary k={summary.get('eval/k')} != {expected_k}")
    if len(tasks) != 48:
        errors.append(f"{stream} step {step}: tasks rows={len(tasks)} != 48")

    by_prompt: dict[int, dict[str, Any]] = {}
    pass_values: list[float] = []
    best_values: list[float] = []
    unique_values: list[float] = []
    distance_values: list[float] = []
    for row_index, row in enumerate(tasks):
        prefix = f"{stream} step {step} row {row_index}"
        prompt_index = row.get("prompt_index")
        if not isinstance(prompt_index, int):
            errors.append(f"{prefix}: non-integer prompt_index")
            continue
        if prompt_index in by_prompt:
            errors.append(f"{prefix}: duplicate prompt_index={prompt_index}")
        by_prompt[prompt_index] = row
        if row.get("k") != expected_k:
            errors.append(f"{prefix}: k={row.get('k')} != {expected_k}")
        rewards = row.get("rewards")
        successes = row.get("successes")
        statuses = row.get("statuses")
        seeds = row.get("sampling_seeds")
        responses = row.get("responses")
        hashes = row.get("response_sha256")
        fields = {
            "rewards": rewards,
            "successes": successes,
            "statuses": statuses,
            "sampling_seeds": seeds,
            "responses": responses,
            "response_sha256": hashes,
        }
        if any(not isinstance(value, list) or len(value) != expected_k for value in fields.values()):
            errors.append(f"{prefix}: one or more sample arrays do not have k={expected_k}")
            continue
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in rewards):
            errors.append(f"{prefix}: rewards contain non-finite/non-numeric values")
            continue
        expected_successes = [float(value) > 0.0 for value in rewards]
        if successes != expected_successes:
            errors.append(f"{prefix}: successes disagree with rewards")
        if any(not isinstance(seed, int) for seed in seeds):
            errors.append(f"{prefix}: missing/non-integer deterministic seed")
        if len(set(seeds)) != expected_k:
            errors.append(f"{prefix}: sampling seeds are not unique within task")
        expected_hashes = [hashlib.sha256(str(value).encode()).hexdigest() for value in responses]
        if hashes != expected_hashes:
            errors.append(f"{prefix}: response hashes disagree with responses")
        expected_pass = float(any(expected_successes))
        expected_best = max(float(value) for value in rewards)
        if not _close(row.get("pass_at_k"), expected_pass):
            errors.append(f"{prefix}: pass_at_k disagrees with rewards")
        if not _close(row.get("best_reward_at_k"), expected_best):
            errors.append(f"{prefix}: best_reward_at_k disagrees with rewards")
        pass_values.append(expected_pass)
        best_values.append(expected_best)
        unique_values.append(
            _finite_float(row.get("response_unique_ratio"), f"{prefix} unique ratio", errors)
        )
        distance_values.append(
            _finite_float(
                row.get("response_pairwise_jaccard_distance"),
                f"{prefix} pairwise distance",
                errors,
            )
        )

    if set(by_prompt) != set(range(48)):
        errors.append(f"{stream} step {step}: prompt indices are not exactly 0..47")
    aggregates = {
        "eval/pass_at_k": sum(pass_values) / 48 if len(pass_values) == 48 else math.nan,
        "eval/reward_best_at_k": sum(best_values) / 48 if len(best_values) == 48 else math.nan,
        "eval/response_unique_ratio": sum(unique_values) / 48 if len(unique_values) == 48 else math.nan,
        "eval/response_pairwise_jaccard_distance": (
            sum(distance_values) / 48 if len(distance_values) == 48 else math.nan
        ),
    }
    for key, expected in aggregates.items():
        if math.isfinite(expected) and not _close(summary.get(key), expected):
            errors.append(f"{stream} step {step}: summary {key} disagrees with task rows")
    return by_prompt


def audit_run(
    run_dir: Path,
    require_eval_through: int,
    require_train_through: int,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[int, dict[str, Any]]]]:
    run_dir = run_dir.resolve()
    errors: list[str] = []
    config = _read_json(run_dir / "config" / "run_config.json", errors)
    meta = _read_json(run_dir / "meta.json", errors)
    if config.get("eval_protocol") != "seta_fixed48_v2":
        errors.append(f"eval_protocol={config.get('eval_protocol')} != seta_fixed48_v2")
    for key, expected in {
        "eval_seed": "20260808",
        "num_rollout": 1000,
        "rollout_batch_size": 4,
        "n_samples": 8,
        "rollout_max_response_len": 8192,
        "rollout_max_context_len": 16384,
    }.items():
        if config.get(key) != expected:
            errors.append(f"run_config {key}={config.get(key)!r} != {expected!r}")

    _check_hash(Path(str(config.get("prompt_data", ""))), config.get("train_set_sha256"), "train set", errors)
    _check_hash(
        REPO_ROOT / "benchmarks/seta_env_convert/eval_fixed48_v2.jsonl",
        config.get("eval_set_sha256"),
        "eval set",
        errors,
    )
    _check_hash(
        REPO_ROOT / "benchmarks/seta_env_convert/eval_fixed48_v2.manifest.json",
        config.get("eval_manifest_sha256"),
        "eval manifest",
        errors,
    )

    source_dir = run_dir / "reproducibility" / "source_state"
    source_manifest = _read_json(source_dir / "manifest.json", errors)
    source_commit = str(source_manifest.get("base_commit") or "")
    meta_commit = str(meta.get("git_commit") or "")
    if not source_commit or not meta_commit or not source_commit.startswith(meta_commit):
        errors.append("source snapshot base commit disagrees with run meta")
    _check_hash(
        source_dir / "tracked.patch",
        source_manifest.get("tracked_patch_sha256"),
        "tracked source patch",
        errors,
    )
    _check_hash(
        source_dir / "untracked-and-ignored-inputs.tar.gz",
        source_manifest.get("inputs_archive_sha256"),
        "source inputs archive",
        errors,
    )

    worker_ref = _read_json(run_dir / "reproducibility" / "worker-state.json", errors)
    worker_dir = (run_dir / "reproducibility" / str(worker_ref.get("worker_snapshot", ""))).resolve()
    worker_manifest = _read_json(worker_dir / "manifest.json", errors)
    worker_hash = worker_ref.get("source_archive_sha256")
    if worker_hash != worker_manifest.get("source_archive_sha256"):
        errors.append("worker snapshot reference disagrees with canonical manifest")
    _check_hash(worker_dir / "worker-source.tar.gz", worker_hash, "worker source archive", errors)

    try:
        schedule = [int(value) for value in str(config.get("eval_steps", "")).split()]
    except ValueError:
        schedule = []
        errors.append("invalid eval_steps in run_config")
    required_steps = [step for step in schedule if 0 <= step <= require_eval_through]
    if require_eval_through >= 0 and require_eval_through not in schedule:
        errors.append(f"required eval step {require_eval_through} is absent from schedule")

    metrics_path = run_dir / "logs" / "metrics.jsonl"
    metrics = _load_jsonl(metrics_path, errors) if (required_steps or require_train_through) else []
    eval_metric_keys = {
        (str(row.get("dataset")), row.get("global_step"))
        for row in metrics
        if row.get("phase") == "eval"
    }
    train_steps = {
        int(row["global_step"])
        for row in metrics
        if row.get("phase") == "train" and isinstance(row.get("global_step"), int)
    }
    if require_train_through:
        missing_train = sorted(set(range(1, require_train_through + 1)) - train_steps)
        if missing_train:
            errors.append(f"missing train global steps: {missing_train[:20]}")

    artifacts: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for step in required_steps:
        for stream, expected_k in STREAMS.items():
            artifacts[(stream, step)] = _audit_eval_artifact(
                run_dir, stream, step, expected_k, errors
            )
            if (stream, step) not in eval_metric_keys:
                errors.append(f"missing eval metrics row for {stream} step {step}")

    return (
        {
            "run_dir": str(run_dir),
            "ok": not errors,
            "errors": errors,
            "required_eval_steps": required_steps,
            "required_train_through": require_train_through,
            "observed_train_max": max(train_steps, default=0),
            "source_fingerprint_sha256": source_manifest.get("source_fingerprint_sha256"),
            "worker_source_archive_sha256": worker_hash,
            "config": config,
        },
        artifacts,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--peer-run-dir", type=Path)
    parser.add_argument("--require-eval-through", type=int, default=-1)
    parser.add_argument("--require-train-through", type=int, default=0)
    args = parser.parse_args()

    report, artifacts = audit_run(
        args.run_dir, args.require_eval_through, args.require_train_through
    )
    output: dict[str, Any] = {"run": {key: value for key, value in report.items() if key != "config"}}
    if args.peer_run_dir:
        peer, peer_artifacts = audit_run(
            args.peer_run_dir, args.require_eval_through, args.require_train_through
        )
        peer_errors = peer["errors"]
        fairness_mismatches = {
            key: [report["config"].get(key), peer["config"].get(key)]
            for key in FAIR_CONFIG_KEYS
            if report["config"].get(key) != peer["config"].get(key)
        }
        if fairness_mismatches:
            report["errors"].append(f"paired run fairness mismatches: {fairness_mismatches}")
        for artifact_key, rows in artifacts.items():
            peer_rows = peer_artifacts.get(artifact_key, {})
            for prompt_index, row in rows.items():
                peer_row = peer_rows.get(prompt_index)
                if peer_row is None:
                    continue
                if row.get("task_id") != peer_row.get("task_id"):
                    report["errors"].append(
                        f"paired task mismatch {artifact_key} prompt {prompt_index}"
                    )
                if row.get("sampling_seeds") != peer_row.get("sampling_seeds"):
                    report["errors"].append(
                        f"paired seed mismatch {artifact_key} prompt {prompt_index}"
                    )
        report["ok"] = not report["errors"]
        output["run"] = {key: value for key, value in report.items() if key != "config"}
        output["peer"] = {key: value for key, value in peer.items() if key != "config"}
        output["paired_fair_config_mismatches"] = fairness_mismatches
        output["paired_ok"] = report["ok"] and not peer_errors
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output.get("paired_ok", report["ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
