from agentic_rl.algorithms import ALGORITHMS, AUXILIARY_ALGORITHMS
from agentic_rl.algorithms.lwm import (
    SIGReg,
    TerminalTransition,
    TextLatentWorldModel,
    TextLatentWorldModelConfig,
    TrajectoryReplayBuffer,
    attach_terminal_world_model_metadata,
    is_world_model_enabled,
    world_model_records_from_samples,
)
from slime.world_model import (
    SIGReg as SlimeSIGReg,
)
from slime.world_model import (
    TerminalTransition as SlimeTerminalTransition,
)
from slime.world_model import (
    TextLatentWorldModel as SlimeTextLatentWorldModel,
)
from slime.world_model import (
    TextLatentWorldModelConfig as SlimeTextLatentWorldModelConfig,
)
from slime.world_model.metadata import (
    attach_terminal_world_model_metadata as metadata_attach,
)
from slime.world_model.metadata import (
    is_world_model_enabled as metadata_is_enabled,
)
from slime.world_model.replay_buffer import TrajectoryReplayBuffer as ReplayBuffer
from slime.world_model.replay_buffer import world_model_records_from_samples as records_from_samples


def test_lwm_is_registered_as_auxiliary_algorithm():
    assert "lwm" in AUXILIARY_ALGORITHMS
    assert "lwm" not in ALGORITHMS


def test_lwm_public_api_reuses_hardened_world_model_implementation():
    assert attach_terminal_world_model_metadata is metadata_attach
    assert is_world_model_enabled is metadata_is_enabled
    assert TrajectoryReplayBuffer is ReplayBuffer
    assert world_model_records_from_samples is records_from_samples
    assert TextLatentWorldModel is SlimeTextLatentWorldModel
    assert TextLatentWorldModelConfig is SlimeTextLatentWorldModelConfig
    assert SIGReg is SlimeSIGReg
    assert TerminalTransition is SlimeTerminalTransition
