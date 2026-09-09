"""Slime custom RM adapter using the exact offline verifier."""

from __future__ import annotations

import os
from typing import Any

from .scorer import ScoreConfig, score_sample
from .verifier import verifier_digest


def _config(args: Any) -> ScoreConfig:
    reward_type = str(
        getattr(args, "math_rlvr_reward_type", "")
        or os.environ.get("MATH_RLVR_REWARD_TYPE", "")
        or getattr(args, "rm_type", "math")
    )
    cap = int(getattr(args, "rollout_max_response_len", 0) or os.environ.get("MATH_RLVR_RESPONSE_CAP", "32768"))
    return ScoreConfig(reward_type=reward_type, response_cap=cap)


def _result(args: Any, sample: Any) -> dict[str, Any]:
    config = _config(args)
    record = score_sample(
        getattr(sample, "response", ""),
        getattr(sample, "label", ""),
        completion_tokens=getattr(sample, "completion_tokens", None),
        finish_reason=getattr(sample, "finish_reason", None),
        config=config,
    )
    record["verifier_sha256"] = verifier_digest()
    # Slime's default ``--reward-key score`` consumes this field.  Returning
    # diagnostics alongside it keeps structured logging possible without a
    # second reward implementation.
    scalar = record["reward"]
    if config.reward_type == "dapo":
        scalar = 1.0 if record["configured_correct"] else -1.0
    return {"score": scalar, "acc": record["configured_correct"], **record}


async def reward_func(args: Any, sample: Any) -> dict[str, Any]:
    return _result(args, sample)


async def batched_reward_func(args: Any, samples: list[Any]) -> list[dict[str, Any]]:
    return [_result(args, sample) for sample in samples]


__all__ = ["batched_reward_func", "reward_func"]
