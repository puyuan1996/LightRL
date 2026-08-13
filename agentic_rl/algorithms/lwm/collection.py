"""Rollout collection API owned by AgenticRL.

The strict metadata implementation remains in ``slime.world_model`` so offline
training and rollout collection share one schema and validation path.
"""

from slime.world_model.metadata import attach_terminal_world_model_metadata, is_world_model_enabled

__all__ = ["attach_terminal_world_model_metadata", "is_world_model_enabled"]
