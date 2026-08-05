from __future__ import annotations

from typing import Any

from agentic_rl.platform.registry import REGISTRY


_TOP_LEVEL_SECTIONS = {
    "harness",
    "model",
    "algorithm",
    "environment",
    "backend",
    "cluster",
    "runtime",
    "experiment",
}


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def validate_config(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - _TOP_LEVEL_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown top-level config fields: {', '.join(unknown)}")

    harness = _mapping(config, "harness")
    model = _mapping(config, "model")
    algorithm = _mapping(config, "algorithm")
    environment = _mapping(config, "environment")
    backend = _mapping(config, "backend")
    cluster = _mapping(config, "cluster")
    runtime = _mapping(config, "runtime")

    REGISTRY.spec("harnesses", str(harness.get("name", "")))
    REGISTRY.spec("models", str(model.get("name", "")))
    REGISTRY.spec("algorithms", str(algorithm.get("name", "")))

    if not str(environment.get("name", "")).strip():
        raise ValueError("environment.name is required")
    if not str(backend.get("name", "")).strip():
        raise ValueError("backend.name is required")

    num_gpus = cluster.get("num_gpus")
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus <= 0:
        raise ValueError("cluster.num_gpus must be a positive integer")

    if not isinstance(runtime.get("env", {}), dict):
        raise ValueError("runtime.env must be a mapping")
