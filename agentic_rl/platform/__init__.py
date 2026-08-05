"""Flat platform layer: configuration, runtime, backend, and services."""

from agentic_rl.platform.config_loader import compose_config, load_config
from agentic_rl.platform.registry import REGISTRY

__all__ = ["REGISTRY", "compose_config", "load_config"]
