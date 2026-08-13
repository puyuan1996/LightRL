from types import SimpleNamespace

import pytest
import torch

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


def _sample():
    return SimpleNamespace(
        tokens=[1, 2, 3],
        response_length=1,
        reward={"score": 1.0},
        metadata={"turn_idx": 0, "num_turns": 1},
        train_metadata=None,
    )


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


def test_collection_is_default_off_and_redacts_enabled_records():
    sample = _sample()
    common = {
        "samples": [sample],
        "turn_records": [
            {
                "turn_idx": 0,
                "context_messages": [{"role": "user", "content": "token=secret-value"}],
                "assistant_output": "run",
                "tool_calls": [
                    {
                        "tool_name": "bash",
                        "args": {"command": "pwd", "api_key": "secret-value"},
                        "result": {"status": "ok", "password": "secret-value"},
                    }
                ],
            }
        ],
        "task_meta": {"task_id": "task-1"},
        "run_ctx": SimpleNamespace(uid="u1"),
        "status": SimpleNamespace(value="completed"),
    }
    attach_terminal_world_model_metadata(
        args=SimpleNamespace(world_model_enable=False),
        **common,
    )
    assert "world_model" not in sample.metadata

    attach_terminal_world_model_metadata(
        args=SimpleNamespace(world_model_enable=True, world_model_metadata_max_chars=4096),
        **common,
    )
    record = sample.metadata["world_model"]
    assert record["schema"] == "openclaw_text_jepa_world_model_v3"
    assert record["redaction_applied"] is True
    assert "secret-value" not in str(record)
    assert "[REDACTED]" in str(record)


def test_replay_roundtrip_verifies_digest_and_rejects_tampering(tmp_path):
    replay = TrajectoryReplayBuffer(buffer_size=2, seed=7)
    assert replay.push(
        [{"uid": "u1", "turn_idx": 0, "action_text": "pwd", "reward_score": 1.0}],
        current_step=3,
    ) == 1
    path = tmp_path / "replay.pt"
    replay.save(path)
    loaded = TrajectoryReplayBuffer.load(path)
    assert loaded.records()[0]["uid"] == "u1"
    assert loaded.provenance_verified is True

    payload = torch.load(path, weights_only=False)
    payload["records"][0]["action_text"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        TrajectoryReplayBuffer(buffer_size=2).load_state_dict(payload)


def test_shared_latent_jepa_forward_contract():
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=4,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            predictor_depth=1,
            predictor_num_heads=1,
            sigreg_num_proj=4,
            value_head=False,
            uncertainty_head=False,
        )
    )
    output = model(
        state_hidden=torch.randn(3, 2, 8),
        action_hidden=torch.randn(3, 2, 8),
        next_state_hidden=torch.randn(3, 2, 8),
    )
    assert output["pred_latent"].shape == (3, 4)
    assert output["next_state_target_latent"].shape == (3, 4)
    assert torch.isfinite(output["pred_latent"]).all()
