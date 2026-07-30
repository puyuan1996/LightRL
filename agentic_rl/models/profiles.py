from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    family: str
    tool_call_parser: str


QWEN3_8B = ModelProfile("qwen3_8b", "qwen3", "qwen25")
QWEN3_30B_A3B = ModelProfile("qwen3_30b_a3b", "qwen3_moe", "qwen25")
GLM_5_1 = ModelProfile("glm_5_1", "glm", "qwen25")
