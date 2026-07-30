from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from slime.utils.types import Sample

from agentic_rl.core.types import RunContext, TaskSpec

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


# ─── Trajectory export (parallels swe-rl/generate_with_swe_remote.py:78-137) ───
# Toggle via env var TERMINAL_SAVE_TRAJ_DIR (empty=disabled).
# Output layout (one dir per rollout sample):
#   {save_dir}/t{task}_r{rollout_id}_st{train_step}_g{group}_s{sample}_{uid}_{ts}/
#       meta.json       # task spec + sampling params + reward breakdown
#       traj.json       # per-turn dialogue + tool calls + ClawSentry decisions

def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def _get_terminal_save_dir() -> Path | None:
    save_dir = os.getenv("TERMINAL_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("TERMINAL_SAVE_TRAJ_DIR=%s mkdir failed: %s", save_dir, exc)
        return None
    return path


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_or_env_int(sample: Sample, key: str, env_name: str) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key)
    if value is None:
        value = os.getenv(env_name)
    return _optional_int(value)


def _trajectory_dataset_slug(data_source: str | None) -> str:
    raw = str(data_source or "").strip().lower()
    if raw in {"", "terminal_bench", "seta", "seta_env"}:
        return "seta"
    if raw in {"agent_safetybench", "agent-safety-bench", "safety", "asb"}:
        return "agent_safetybench"
    if raw in {"agentharm", "agent_harm", "ah"}:
        return "agentharm"
    return _sanitize_filename(raw) or "unknown"


def _interval_candidates_for_dataset(dataset_slug: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if dataset_slug == "seta":
        return (
            ("trajectory_save_interval_seta", "trajectory_save_interval_terminal_bench"),
            ("TRAJECTORY_SAVE_INTERVAL_SETA", "SAVE_INTERVAL_SETA"),
        )
    if dataset_slug == "agent_safetybench":
        return (
            (
                "trajectory_save_interval_agent_safetybench",
                "trajectory_save_interval_asb",
                "trajectory_save_interval_safety",
            ),
            (
                "TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH",
                "TRAJECTORY_SAVE_INTERVAL_ASB",
                "TRAJECTORY_SAVE_INTERVAL_SAFETY",
                "SAVE_INTERVAL_AGENT_SAFETYBENCH",
                "SAVE_INTERVAL_ASB",
                "SAVE_INTERVAL_SAFETY",
            ),
        )
    if dataset_slug == "agentharm":
        return (
            ("trajectory_save_interval_agentharm", "trajectory_save_interval_agent_harm"),
            (
                "TRAJECTORY_SAVE_INTERVAL_AGENTHARM",
                "TRAJECTORY_SAVE_INTERVAL_AGENT_HARM",
                "SAVE_INTERVAL_AGENTHARM",
                "SAVE_INTERVAL_AGENT_HARM",
            ),
        )
    return (
        (f"trajectory_save_interval_{dataset_slug}",),
        (f"TRAJECTORY_SAVE_INTERVAL_{dataset_slug.upper()}",),
    )


def _trajectory_save_interval(args, data_source: str | None = None) -> int:
    dataset_slug = _trajectory_dataset_slug(data_source)
    arg_names, env_names = _interval_candidates_for_dataset(dataset_slug)
    raw = None
    raw_source = None
    for name in arg_names:
        value = getattr(args, name, None)
        if value is not None and value != "":
            raw = value
            raw_source = name
            break
    if raw is None:
        for name in env_names:
            value = os.getenv(name)
            if value is not None and value != "":
                raw = value
                raw_source = name
                break
    if raw is None:
        raw = getattr(args, "trajectory_save_interval", None)
        raw_source = "trajectory_save_interval"
    if raw is None or raw == "":
        raw = os.getenv("TRAJECTORY_SAVE_INTERVAL", "1")
        raw_source = "TRAJECTORY_SAVE_INTERVAL"
    value = _optional_int(raw)
    if value is None:
        logger.warning(
            "Invalid trajectory save interval %s=%r for dataset=%s; falling back to 1",
            raw_source,
            raw,
            dataset_slug,
        )
        return 1
    return value


def _should_save_trajectory(run_ctx: RunContext, interval: int) -> bool:
    if interval <= 0:
        return False
    if interval == 1:
        return True
    step = run_ctx.train_step
    if step is None:
        step = run_ctx.rollout_id
    if step is None:
        # No rollout metadata is available, so preserve old save-all behavior.
        return True
    return int(step) % interval == 0


def _trajectory_save_policy() -> str:
    raw = os.getenv("TRAJECTORY_SAVE_POLICY", "step_interval").strip().lower()
    if raw in {"", "legacy", "interval"}:
        return "step_interval"
    if raw in {"task_timeseries", "task-time-series", "task_step", "task-step"}:
        return "task_timeseries"
    logger.warning("Unknown TRAJECTORY_SAVE_POLICY=%r; using step_interval", raw)
    return "step_interval"


def _trajectory_env_int(name: str, default: int) -> int:
    return _env_int(name, default)


def _trajectory_task_save_interval(default_interval: int) -> int:
    raw = os.getenv("TRAJECTORY_TASK_SAVE_INTERVAL", "").strip()
    if not raw:
        return default_interval
    value = _optional_int(raw)
    if value is None:
        logger.warning(
            "Invalid TRAJECTORY_TASK_SAVE_INTERVAL=%r; using %d",
            raw,
            default_interval,
        )
        return default_interval
    return value


def _trajectory_reward_strata() -> set[str]:
    raw = os.getenv("TRAJECTORY_SAVE_REWARD_STRATA", "best,worst")
    values = {
        part.strip().lower()
        for part in raw.split(",")
        if part.strip()
    }
    allowed = {"best", "worst", "latest"}
    unknown = values - allowed
    if unknown:
        logger.warning(
            "Ignoring unknown TRAJECTORY_SAVE_REWARD_STRATA entries: %s",
            sorted(unknown),
        )
    values &= allowed
    return values or {"best", "worst"}


def _trajectory_step_value(run_ctx: RunContext) -> int | None:
    for value in (run_ctx.train_step, run_ctx.rollout_step, run_ctx.rollout_id):
        step = _optional_int(value)
        if step is not None:
            return step
    return None


def _trajectory_task_id(task_spec: TaskSpec) -> str:
    name = str(task_spec.task_name or "unknown")
    path = str(task_spec.task_path or "")
    digest = hashlib.sha1(f"{name}\n{path}".encode("utf-8")).hexdigest()[:8]
    slug = _sanitize_filename(name)[:96].strip("._-") or "unknown"
    return f"{slug}-{digest}"


def _trajectory_reward_value(reward: Dict[str, Any]) -> float | None:
    for key in ("total_reward", "score", "raw_reward", "raw_score", "accuracy"):
        value = reward.get(key)
        try:
            if value is None or value == "":
                continue
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def _format_reward_for_filename(value: float | None) -> str:
    if value is None:
        return "na"
    text = f"{value:+.3f}"
    return (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )
