from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from agentic_rl.config.schema import LightRLConfig


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    parent = data.pop("extends", None)
    if not parent:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    return _deep_merge(_load_yaml(parent_path, (*stack, config_path)), data)


def _set_override(target: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    keys = dotted_key.split(".")
    cursor = target
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override through scalar key: {key}")
        cursor = child
    cursor[keys[-1]] = yaml.safe_load(raw_value)


def compose_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    default_config = asdict(LightRLConfig())
    merged = _deep_merge(default_config, _load_yaml(Path(config_path)))
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        key, value = override.split("=", 1)
        _set_override(merged, key, value)
    return merged


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    return compose_config(config_path, overrides)
