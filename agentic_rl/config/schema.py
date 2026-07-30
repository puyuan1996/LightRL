from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HarnessConfig:
    name: str = "camel_agent"
    max_turns: int = 16
    timeout_sec: int = 900


@dataclass
class ModelConfig:
    name: str = "qwen3_8b"
    checkpoint: str = ""
    tokenizer: str = ""


@dataclass
class DAPOConfig:
    name: str = "dapo"
    clip_ratio: float = 0.2
    dynamic_sampling: bool = False


@dataclass
class DivePOConfig:
    enabled: bool = True
    centered_gate: bool = True
    quality_blend: float = 1.0
    reward_processor: str = (
        "agentic_rl.algorithms.dive_po.rewards.centered_gate."
        "post_process_rewards"
    )


@dataclass
class LWMConfig:
    enabled: bool = False
    encoder: str = "hash"
    use_dapo_replay_buffer: bool = False
    loss_coef: float = 0.0


@dataclass
class AlgorithmConfig:
    name: str = "dive_po"
    base: DAPOConfig = field(default_factory=DAPOConfig)
    dive_po: DivePOConfig = field(default_factory=DivePOConfig)
    lwm: LWMConfig = field(default_factory=LWMConfig)


@dataclass
class EnvironmentConfig:
    name: str = "seta"
    dataset: str = ""
    max_turn: int = 10


@dataclass
class BackendConfig:
    name: str = "slime"
    train_entrypoint: str = "slime/train_async.py"


@dataclass
class ClusterConfig:
    name: str = "local"
    num_gpus: int = 8


@dataclass
class RuntimeConfig:
    launcher: str = ""
    run_id: str = ""
    run_name: str = ""
    wandb_group: str = ""
    env: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightRLConfig:
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
