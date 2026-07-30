from __future__ import annotations

import inspect
from typing import Any

from agentic_rl.core.registry import REGISTRY


def normalize_harness_name(value: str | None) -> str:
    requested = value or "camel_agent"
    try:
        return REGISTRY.canonical_name("harnesses", requested)
    except KeyError as exc:
        choices = ", ".join(REGISTRY.names("harnesses"))
        raise ValueError(
            f"Unsupported harness option: {value!r}. Available: {choices}"
        ) from exc


def create_harness(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered harness without coupling rollout core to it."""
    factory = REGISTRY.load("harnesses", name)
    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return factory(**kwargs)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return factory(**accepted)
