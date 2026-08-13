"""AgenticRL-facing interfaces for the latent world model."""

__all__ = [
    "SIGReg",
    "TerminalTransition",
    "TextLatentWorldModel",
    "TextLatentWorldModelConfig",
    "TrajectoryReplayBuffer",
    "attach_terminal_world_model_metadata",
    "is_world_model_enabled",
    "world_model_records_from_samples",
]


def __getattr__(name):
    if name in {"attach_terminal_world_model_metadata", "is_world_model_enabled"}:
        from .collection import attach_terminal_world_model_metadata, is_world_model_enabled

        return {
            "attach_terminal_world_model_metadata": attach_terminal_world_model_metadata,
            "is_world_model_enabled": is_world_model_enabled,
        }[name]
    if name in {"SIGReg", "TextLatentWorldModel", "TextLatentWorldModelConfig"}:
        from slime.world_model import SIGReg, TextLatentWorldModel, TextLatentWorldModelConfig

        return {
            "SIGReg": SIGReg,
            "TextLatentWorldModel": TextLatentWorldModel,
            "TextLatentWorldModelConfig": TextLatentWorldModelConfig,
        }[name]
    if name == "TerminalTransition":
        from slime.world_model import TerminalTransition

        return TerminalTransition
    if name in {"TrajectoryReplayBuffer", "world_model_records_from_samples"}:
        from .replay import TrajectoryReplayBuffer, world_model_records_from_samples

        return {
            "TrajectoryReplayBuffer": TrajectoryReplayBuffer,
            "world_model_records_from_samples": world_model_records_from_samples,
        }[name]
    raise AttributeError(name)
