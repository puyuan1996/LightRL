"""LightRL agentic RL package."""

from .config.loader import load_config
from .core.registry import REGISTRY

__all__ = ["REGISTRY", "load_config"]
