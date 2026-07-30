from __future__ import annotations


REGISTRY = {
    "harnesses": {
        "camel_agent": "agentic_rl.harnesses.camel_agent",
        "claude_code_cli": "agentic_rl.harnesses.claude_code_agent",
    },
    "models": {
        "qwen3_8b": "qwen3_8b",
        "qwen3_30b_a3b": "qwen3_30b_a3b",
        "glm_5_1": "glm_5_1",
    },
    "algorithms": {
        "grpo": "grpo",
        "dapo": "dapo",
        "dive_po": "agentic_rl.algorithms.dive_po",
        "lwm": "agentic_rl.algorithms.lwm",
    },
}
