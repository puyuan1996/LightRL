from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from agentic_rl.config.schema import LightRLConfig
from agentic_rl.config.validation import validate_config


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if candidate.name == "configs":
            return candidate
    raise ValueError(f"Config must live under a configs directory: {path}")


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _default_fragment(
    item: Any,
    *,
    config_root: Path,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    if isinstance(item, str):
        return _load_yaml((config_root / item).with_suffix(".yaml"), stack)
    if not isinstance(item, dict) or len(item) != 1:
        raise ValueError(f"Invalid defaults entry: {item!r}")
    group, name = next(iter(item.items()))
    fragment_path = config_root / str(group) / f"{name}.yaml"
    fragment = _load_yaml(fragment_path, stack)
    return {str(group): fragment}


def _load_yaml(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Cyclic config inheritance: {chain}")

    data = _load_mapping(config_path)
    next_stack = (*stack, config_path)
    composed: dict[str, Any] = {}

    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        composed = _deep_merge(composed, _load_yaml(parent_path, next_stack))

    defaults = data.pop("defaults", [])
    if defaults:
        if not isinstance(defaults, list):
            raise ValueError(f"defaults must be a list: {config_path}")
        root = _config_root(config_path)
        for item in defaults:
            composed = _deep_merge(
                composed,
                _default_fragment(item, config_root=root, stack=next_stack),
            )

    return _deep_merge(composed, data)


def _set_override(target: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    keys = dotted_key.split(".")
    if not all(keys):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
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
    merged = _deep_merge(asdict(LightRLConfig()), _load_yaml(Path(config_path)))
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        key, value = override.split("=", 1)
        _set_override(merged, key, value)
    validate_config(merged)
    return merged


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    return compose_config(config_path, overrides)
