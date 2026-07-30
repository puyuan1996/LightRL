from __future__ import annotations

import logging
import math
import os
from typing import Any


logger = logging.getLogger(__name__)


def env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def schedule_multiplier(schedule: str, train_step: Any, decay_steps: int) -> float:
    name = str(schedule or "constant").strip().lower()
    if name in {"", "constant", "none", "off"}:
        return 1.0
    if decay_steps <= 0 or train_step is None:
        return 1.0
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return 1.0
    progress = min(1.0, step / float(decay_steps))
    if name == "linear":
        return max(0.0, 1.0 - progress)
    if name == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    logger.warning("Unknown reward schedule=%r; using constant", name)
    return 1.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def advantage_gate_mode() -> str:
    return env_str("EXPLORE_ADVANTAGE_GATE_MODE", "legacy").lower()


def uses_outcome_status_gate(gate_mode: str | None = None) -> bool:
    mode = advantage_gate_mode() if gate_mode is None else str(gate_mode or "").lower()
    return mode in {
        "outcome",
        "outcome_status",
        "quality",
        "quality_gate",
        "status_quality",
    }


def outcome_candidate_keys() -> list[str]:
    configured = env_str("EXPLORE_ADVANTAGE_OUTCOME_KEY", "raw_score")
    keys = [configured] if configured else []
    keys.extend(
        [
            "raw_score",
            "accuracy",
            "success_score",
            "unit_test_pass_rate",
            "test_acc",
            "pass_rate",
            "base_score",
            "score",
        ]
    )
    return list(dict.fromkeys(key for key in keys if key))


def normalize_outcome_value(key: str, value: float) -> float:
    if key in {"score", "base_score", "task_reward", "raw_reward"} and value < 0.0:
        return clamp01(0.5 * (value + 1.0))
    return clamp01(value)


def normalize_values(values: list[float], use_std: bool) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    if not use_std:
        return centered
    if len(values) <= 1:
        return [0.0 for _ in values]
    variance = sum(value * value for value in centered) / max(1, len(values) - 1)
    std = math.sqrt(max(variance, 0.0))
    return [value / (std + 1e-6) for value in centered]
