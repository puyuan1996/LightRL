from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class PluginSpec:
    group: str
    name: str
    target: str


class PluginRegistry:
    """Lazy registry for LightRL extension points."""

    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, PluginSpec]] = {}
        self._aliases: dict[str, dict[str, str]] = {}

    @staticmethod
    def _key(value: str) -> str:
        return str(value).strip().lower().replace("-", "_")

    def register(
        self,
        group: str,
        name: str,
        target: str,
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        group_key = self._key(group)
        name_key = self._key(name)
        self._plugins.setdefault(group_key, {})[name_key] = PluginSpec(
            group=group_key,
            name=name_key,
            target=target,
        )
        alias_map = self._aliases.setdefault(group_key, {})
        alias_map[name_key] = name_key
        for alias in aliases:
            alias_map[self._key(alias)] = name_key

    def canonical_name(self, group: str, name: str) -> str:
        group_key = self._key(group)
        name_key = self._key(name)
        try:
            return self._aliases[group_key][name_key]
        except KeyError as exc:
            choices = ", ".join(self.names(group_key))
            raise KeyError(
                f"Unknown {group_key} plugin {name!r}; available: {choices}"
            ) from exc

    def spec(self, group: str, name: str) -> PluginSpec:
        group_key = self._key(group)
        canonical = self.canonical_name(group_key, name)
        return self._plugins[group_key][canonical]

    def load(self, group: str, name: str) -> Any:
        spec = self.spec(group, name)
        module_name, separator, attribute = spec.target.partition(":")
        module = import_module(module_name)
        return getattr(module, attribute) if separator else module

    def names(self, group: str) -> tuple[str, ...]:
        return tuple(sorted(self._plugins.get(self._key(group), {})))


REGISTRY = PluginRegistry()

REGISTRY.register(
    "harnesses",
    "camel_agent",
    "agentic_rl.harnesses.camel.agent:CamelAgent",
    aliases=("camel", "camel-agent", "camelagent"),
)
REGISTRY.register(
    "harnesses",
    "claude_code_cli",
    "agentic_rl.harnesses.claude_code.agent:ClaudeCodeAgent",
    aliases=("claude", "claude-code", "claude_code", "claude-code-harness"),
)

REGISTRY.register("models", "qwen3_8b", "agentic_rl.models.profiles:QWEN3_8B")
REGISTRY.register(
    "models", "qwen3_30b_a3b", "agentic_rl.models.profiles:QWEN3_30B_A3B"
)
REGISTRY.register("models", "glm_5_1", "agentic_rl.models.profiles:GLM_5_1")

REGISTRY.register("algorithms", "grpo", "agentic_rl.algorithms.grpo")
REGISTRY.register("algorithms", "dapo", "agentic_rl.algorithms.dapo")
REGISTRY.register("algorithms", "dive_po", "agentic_rl.algorithms.dive_po")
REGISTRY.register("algorithms", "lwm", "agentic_rl.algorithms.lwm")
