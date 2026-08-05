"""LightRL agentic RL package."""

from .platform.config_loader import load_config
from .platform.registry import REGISTRY

__all__ = ["REGISTRY", "load_config"]
