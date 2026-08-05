from __future__ import annotations

from typing import Any

from agentic_rl.platform.registry import REGISTRY


def resolve_plugin(group: str, name: str) -> Any:
    return REGISTRY.load(group, name)
