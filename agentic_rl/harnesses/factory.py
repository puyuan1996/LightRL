from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

_HARNESS_ALIASES = {
    "camel_agent": "camel_agent", "camel": "camel_agent",
    "camel-agent": "camel_agent", "camelagent": "camel_agent",
    "claude_code_cli": "claude_code_cli", "claude": "claude_code_cli",
    "claude-code": "claude_code_cli", "claude_code": "claude_code_cli",
    "claude-code-harness": "claude_code_cli",
}
_HARNESS_TARGETS = {
    "camel_agent": ("agentic_rl.harnesses.camel.agent", "CamelAgent"),
    "claude_code_cli": ("agentic_rl.harnesses.claude_code.agent", "ClaudeCodeAgent"),
}
# Canonical -> user-facing display name (logs, turn records, CLI help).
_HARNESS_DISPLAY_NAMES = {
    "camel_agent": "camel-agent",
    "claude_code_cli": "claude-code",
}


def normalize_harness_name(value: str | None) -> str:
    requested = value or "camel_agent"
    try:
        return _HARNESS_ALIASES[requested]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported harness option: {value!r}. Available: camel-agent, claude-code"
        ) from exc


def display_harness_name(value: str | None) -> str:
    """Resolve any accepted harness alias to its canonical display name."""
    canonical = normalize_harness_name(value)
    return _HARNESS_DISPLAY_NAMES.get(canonical, canonical.replace("_", "-"))


def create_harness(name: str, **kwargs: Any) -> Any:
    """Instantiate a supported harness while keeping optional imports lazy."""
    module_name, class_name = _HARNESS_TARGETS[normalize_harness_name(name)]
    factory = getattr(import_module(module_name), class_name)
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
