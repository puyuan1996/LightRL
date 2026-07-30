from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from agentic_rl.harnesses.prm.agent import TerminalPRMAgent
from agentic_rl.integrations.clawsentry.client import ClawSentryClient
from agentic_rl.core.types import (
    Interaction,
    RunContext,
    TaskSpec,
    TaskTimeouts,
    TurnContext,
    TurnResult,
)
from agentic_rl.inference.sglang import SGLangTurnClient
from agentic_rl.rollout.runner import create_agent_runner, normalize_harness_option
from agentic_rl.environments.client import TerminalEnvClient
from agentic_rl.algorithms.dive_po.exploration.agent57.memory import create_episodic_memory_backend
from agentic_rl.algorithms.dive_po.exploration.agent57.controller import (
    coarse_observation_fingerprint as _agent57_coarse_observation_fingerprint,
    coarse_observation_label as _agent57_coarse_observation_label,
    compute_ngu_lite_bonus as _agent57_compute_ngu_lite_bonus,
    compute_lifelong_bonus as _agent57_compute_lifelong_bonus,
    config_from_env as _agent57_config_from_env,
    exit_code_bucket as _agent57_exit_code_bucket,
    record_arm_event as _agent57_record_arm_event,
)
from agentic_rl.rewards.safety import (
    DEFAULT_ZERO_THRESHOLD as _SAFETY_ZERO_THRESHOLD,
    broadcast_to_turns as _safety_broadcast,
    per_turn_score as _safety_per_turn_score,
    trajectory_score as _safety_trajectory_score,
)

logger = logging.getLogger(__name__)

_DIRECT_SCORE_DATA_SOURCES = {"agent_safetybench", "agentharm", "tau2"}
_AGENT57_CONFIG = _agent57_config_from_env()


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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.4f", name, raw, default)
        return default


def _env_csv_set(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {part.strip() for part in raw.split(",") if part.strip()}


from agentic_rl.rollout.admission import (
    _acquire_remote_env_admission,
    _await_with_optional_timeout,
    _is_reset_fresh_lease_retryable,
    _release_remote_env_admission,
    _remote_env_close_semaphore,
    _task_circuit_open_reason,
    _task_circuit_record_failure,
    _task_circuit_record_success,
    _uses_remote_terminal_env,
)
from agentic_rl.algorithms.dive_po.exploration.rollout_bonus import (
    _AGENT57_CONFIG,
    _explore_cde_actor_metrics,
    _explore_debug_metrics,
    _explore_episode_signature_novelty,
    _explore_intrinsic_bonus,
    _explore_lprnd_bonus,
    _explore_safety_penalty,
    _explore_schedule_multiplier,
    _explore_score_bonus_from_components,
    _finite_float,
    _iter_explore_actions,
    _summarize_turn_uncertainty,
    _turn_uncertainty_metrics,
)
from agentic_rl.rollout.trajectory_store import (
    _jsonable,
    _optional_int,
    _sample_or_env_int,
    _save_rollout_artifacts,
    _trajectory_save_interval,
)


from agentic_rl.rollout.sample_builder import (
    _build_agent_safetybench_eval_payload,
    _build_samples,
    _dapo_overlong_cfg,
    _env_flag,
    _extract_task_meta,
    _last_eval_details,
    _make_task_spec,
    _mark_non_trainable_samples,
    _safety_split_from_meta,
    _sync_reward_aliases,
)


from agentic_rl.inference.factory import (
    _create_sglang_client,
    _normalize_tool_schemas,
)
from agentic_rl.rollout.environment_factory import (
    _create_env_client,
    _normalize_tau2_conversation_mode,
)


async def generate(
    args,
    sample: Sample,
    sampling_params: Dict[str, Any],
    evaluation: bool = False,
) -> List[Sample]:
    _ = evaluation
    state = GenerateState(args)

    task_meta = _extract_task_meta(sample)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    data_source = str(task_meta.get("data_source", ""))
    seta_safety_mode = os.getenv("SETA_SAFETY", "none")
    safety_bench_reward_mode = os.getenv("SAFETY_BENCH_REWARD", "rule")
    agentharm_reward_mode = os.getenv("AGENTHARM_REWARD", "rule")
    uid = (sample.metadata or {}).get("uid") or uuid.uuid4().hex[:8]
    group_index = int(sample.group_index) if sample.group_index is not None else -1
    sample_index = int(sample.index) if sample.index is not None else -1
    rollout_id = _sample_or_env_int(sample, "rollout_id", "_CURRENT_ROLLOUT_ID")
    train_step = _sample_or_env_int(sample, "train_step", "_CURRENT_TRAIN_STEP")
    rollout_step = _sample_or_env_int(sample, "rollout_step", "_CURRENT_ROLLOUT_STEP")
    task_spec = _make_task_spec(task_meta)
    run_ctx = RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=Path(getattr(args, "tbench_output_root", "build_outputs"))
        / "AgentRunner_Output",
        rollout_id=rollout_id,
        train_step=train_step,
        rollout_step=rollout_step,
    )
    run_ctx_payload = run_ctx.to_payload()

    def _timeout_arg(
        attr_name: str,
        env_name: str,
        default: float,
        *,
        minimum: float | None = None,
    ) -> float:
        raw = getattr(args, attr_name, None)
        if raw is None:
            raw = os.getenv(env_name)
        if raw is None or raw == "":
            value = default
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = default
        if value <= 0:
            value = default
        if minimum is not None and value < minimum:
            value = minimum
        return value

    timeouts = TaskTimeouts(
        ensure_image=_timeout_arg(
            "ensure_image_timeout",
            "ENSURE_IMAGE_TIMEOUT",
            1200.0,
            minimum=1200.0,
        ),
        reset_session=_timeout_arg(
            "reset_session_timeout",
            "RESET_SESSION_TIMEOUT",
            600.0,
            minimum=600.0,
        ),
        close_session=_timeout_arg(
            "close_session_timeout",
            "CLOSE_SESSION_TIMEOUT",
            60.0,
        ),
        eval=_timeout_arg("eval_timeout", "EVAL_TIMEOUT", 600.0),
    )
    timeouts_payload = timeouts.to_payload()

    env_client: Optional[TerminalEnvClient] = None
    lease_id: Optional[str] = None
    remote_env_admission_key: Optional[str] = None
    agent_runner = None
    heartbeat_task: asyncio.Task | None = None

    prm_enable = bool(getattr(args, "prm_enable", False)) and (not evaluation)
    prm_coef = float(getattr(args, "prm_turn_coef", 1.0))
    prm_agent: TerminalPRMAgent | None = None
    prm_pending: list[tuple[int, asyncio.Task]] = []
    prm_turn_scores: dict[int, float] = {}
    prm_turn_details: list[dict[str, Any]] = []

    def _env_truthy(name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    if data_source == "agent_safetybench":
        safety_enable = safety_bench_reward_mode == "clawsentry"
    elif data_source == "agentharm":
        safety_enable = agentharm_reward_mode == "clawsentry"
    else:
        safety_enable = seta_safety_mode == "clawsentry"
    safety_enable = safety_enable and (not evaluation)
    safety_coef = _env_float("SAFETY_REWARD_COEF", 0.0)
    traj_save_interval = _trajectory_save_interval(args, data_source=data_source)
    safety_summary_weight = _env_float("SAFETY_REWARD_SUMMARY_WEIGHT", 0.3)
    safety_zero_threshold = _env_float(
        "SAFETY_REWARD_ZERO_THRESHOLD", _SAFETY_ZERO_THRESHOLD
    )
    cs_client: ClawSentryClient | None = None
    cs_per_call: list[tuple[int, float]] = []
    cs_per_call_full: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []

    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    _log_tag = f"[task={task_spec.task_name} uid={run_ctx.uid} group_idx={run_ctx.group_index} sample_idx={run_ctx.sample_index}]"

    try:
        env_client, lease_id = await _create_env_client(
            task_spec, run_ctx, task_meta
        )
        reset_kwargs = {
            "lease_id": lease_id,
            "task_meta": task_meta,
            "run_ctx": run_ctx_payload,
            "task_timeouts": timeouts_payload,
        }
        if _uses_remote_terminal_env(task_meta):
            open_reason = _task_circuit_open_reason(task_key)
            if open_reason is not None:
                raise RuntimeError(
                    f"TASK_CIRCUIT_OPEN task_key={task_key}: {open_reason}"
                )
            remote_env_admission_key = await _acquire_remote_env_admission(
                task_key,
                log_tag=_log_tag,
            )
        default_reset_http_timeout = (
            float(timeouts.ensure_image) + float(timeouts.reset_session) + 300.0
        )
        reset_http_timeout = _env_float(
            "ENV_RESET_HTTP_TIMEOUT",
            default_reset_http_timeout,
        )
        max_reset_lease_attempts = max(1, _env_int("ENV_RESET_LEASE_MAX_ATTEMPTS", 1))
        reset_payload: dict[str, Any] | None = None
        last_reset_exc: BaseException | None = None

        # Reset can fail because the remote worker admitted the lease but could not
        # enter reset before its admission timeout.  In that state the server may
        # clean up the lease and subsequent reset attempts return 410.  Never retry
        # the same lease after those failures; close it best-effort and allocate a
        # fresh lease with a unique request_id instead.
        reset_fresh_lease_retries = (
            max(0, _env_int("ENV_RESET_FRESH_LEASE_RETRIES", 2))
            if _uses_remote_terminal_env(task_meta)
            else 0
        )
        reset_payload: dict[str, Any] | None = None
        for reset_attempt in range(reset_fresh_lease_retries + 1):
            reset_kwargs["lease_id"] = lease_id
            if remote_env_admission_key is not None:
                reset_kwargs["request_id"] = (
                    f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:"
                    f"{run_ctx.sample_index}:reset:{reset_attempt}"
                )
            reset_coro = env_client.reset(**reset_kwargs)
            try:
                reset_payload = await _await_with_optional_timeout(
                    reset_coro,
                    reset_http_timeout,
                    op_name=f"{_log_tag} env reset",
                )
                break
            except (TimeoutError, asyncio.TimeoutError) as reset_exc:
                should_retry_reset = reset_attempt < reset_fresh_lease_retries
                logger.error(
                    "%s Reset timed out after %.1fs on lease %s%s",
                    _log_tag,
                    reset_http_timeout,
                    lease_id,
                    "; allocating fresh lease" if should_retry_reset else "",
                )
                try:
                    await env_client.close(lease_id)
                except Exception as close_exc:
                    logger.debug(
                        "%s Best-effort close after reset timeout: %s",
                        _log_tag,
                        close_exc,
                    )
                if not should_retry_reset:
                    raise reset_exc
            except Exception as reset_exc:
                should_retry_reset = (
                    reset_attempt < reset_fresh_lease_retries
                    and _is_reset_fresh_lease_retryable(reset_exc)
                )
                if not should_retry_reset:
                    raise
                logger.warning(
                    "%s Reset failed on lease %s with retryable remote error; "
                    "allocating fresh lease (attempt %d/%d): %s",
                    _log_tag,
                    lease_id,
                    reset_attempt + 1,
                    reset_fresh_lease_retries,
                    reset_exc,
                )
                try:
                    await env_client.close(lease_id)
                except Exception as close_exc:
                    logger.debug(
                        "%s Best-effort close after reset failure: %s",
                        _log_tag,
                        close_exc,
                    )

            fresh_request_id = (
                f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:"
                f"{run_ctx.sample_index}:reset-fresh:{reset_attempt + 1}:"
                f"{uuid.uuid4().hex[:8]}"
            )
            allocate_timeout = _env_float("ENV_ALLOCATE_HTTP_TIMEOUT", 300.0)
            fresh_lease = await _await_with_optional_timeout(
                env_client.allocate(task_key=task_key, request_id=fresh_request_id),
                allocate_timeout,
                op_name=f"{_log_tag} terminal env re-allocate after reset failure",
            )
            lease_id = str(fresh_lease["lease_id"])
            logger.info(
                "%s Re-allocated remote terminal env lease=%s after reset failure",
                _log_tag,
                lease_id,
            )
        if reset_payload is None:
            raise RuntimeError(f"{_log_tag} env reset did not return a payload")

        heartbeat_interval = _env_float("ENV_HEARTBEAT_INTERVAL", 30.0)
        if _uses_remote_terminal_env(task_meta) and heartbeat_interval > 0:
            async def _remote_env_heartbeat_loop() -> None:
                assert env_client is not None and lease_id is not None
                while True:
                    await asyncio.sleep(heartbeat_interval)
                    try:
                        await env_client.heartbeat(lease_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception as heartbeat_exc:
                        logger.warning(
                            "%s Background heartbeat failed for lease %s: %s",
                            _log_tag,
                            lease_id,
                            heartbeat_exc,
                        )
                        if _is_reset_fresh_lease_retryable(heartbeat_exc):
                            return

            heartbeat_task = asyncio.create_task(_remote_env_heartbeat_loop())

        user_msg = str(reset_payload.get("user_msg", ""))
        raw_tools = list(reset_payload.get("tool_schemas", []))
        logger.info("%s Start terminal rollout", _log_tag)

        tool_schemas = _normalize_tool_schemas(raw_tools)
        tau2_conversation_mode = _normalize_tau2_conversation_mode(
            task_meta.get("tau2_mode") or os.getenv("TAU2_MODE", "solo")
        )
        agent_type = normalize_harness_option(
            getattr(args, "harness_option", None)
            or getattr(args, "terminal_agent_type", None)
            or "camel_agent"
        )
        model_type = str(getattr(args, "model_type", "slime-sglang"))
        non_think_mode = bool(getattr(args, "non_think_mode", True))
        non_think_mode_source = str(
            getattr(args, "non_think_mode_source", "prompt")
        ).lower()
        if non_think_mode_source not in {"prompt", "sglang", "both"}:
            non_think_mode_source = "prompt"
        enable_prompt_non_think = non_think_mode and non_think_mode_source in {
            "prompt",
            "both",
        }
        enable_sglang_non_think = non_think_mode and non_think_mode_source in {
            "sglang",
            "both",
        }

        terminal_max_iterations = max(1, int(getattr(args, "max_iteration", 10)))
        terminal_max_parse_errors = max(1, int(getattr(args, "max_parse_errors", 3)))
        max_total_tokens = int(getattr(args, "max_total_tokens", 32768))
        sglang_client = _create_sglang_client(
            args=args,
            tokenizer=state.tokenizer,
            sampling_params=sampling_params,
            max_total_tokens=max_total_tokens,
            enable_sglang_non_think=enable_sglang_non_think,
        )

        if prm_enable:
            prm_router_ip = getattr(args, "prm_router_ip", None)
            prm_router_port = getattr(args, "prm_router_port", None)
            if prm_router_ip and prm_router_port:
                prm_sglang_url = f"http://{prm_router_ip}:{prm_router_port}/generate"
            else:
                prm_sglang_url = getattr(args, "prm_sglang_url", None) or os.getenv(
                    "PRM_SGLANG_URL", ""
                )
            if not prm_sglang_url:
                raise RuntimeError(
                    "prm_enable=True but no PRM endpoint: set prm_router_ip/port, "
                    "prm_sglang_url, or PRM_SGLANG_URL env var."
                )
            prm_sampling_params = {
                "temperature": float(getattr(args, "prm_temperature", 0.0)),
                "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 4096)),
            }
            prm_max_total_tokens = int(getattr(args, "prm_max_total_tokens", 16384))
            prm_sglang_client = _create_sglang_client(
                args=args,
                tokenizer=state.tokenizer,
                sampling_params=prm_sampling_params,
                max_total_tokens=prm_max_total_tokens,
                enable_sglang_non_think=True,
                sglang_url=prm_sglang_url,
                max_retries=10,
            )
            prm_agent = TerminalPRMAgent(
                sglang_client=prm_sglang_client,
                task_instruction=task_spec.instruction,
                history_mode=str(getattr(args, "prm_history_mode", "head_tail")),
            )
            logger.info(
                "%s PRM enabled: url=%s coef=%.3f", _log_tag, prm_sglang_url, prm_coef
            )

        if safety_enable:
            cs_base = os.getenv("CS_HTTP_URL", "http://127.0.0.1:8090")
            cs_session_id = (
                f"openclaw-rl:{task_spec.task_name}:{run_ctx.uid}"
                f":g{run_ctx.group_index}:s{run_ctx.sample_index}"
            )
            cs_timeout = _env_float("SAFETY_REWARD_TIMEOUT", 2.0)
            cs_client = ClawSentryClient(
                base_url=cs_base,
                session_id=cs_session_id,
                agent_id="openclaw-rl-trainer",
                auth_token=os.getenv("CS_AUTH_TOKEN") or None,
                timeout=cs_timeout,
                enabled=True,
            )
            logger.info(
                "%s ClawSentry enabled: url=%s coef=%.3f sid=%s",
                _log_tag,
                cs_base,
                safety_coef,
                cs_session_id,
            )

        agent_runner = create_agent_runner(
            agent_type=agent_type,
            sglang_client=sglang_client,
            model_type=model_type,
            tool_schemas=tool_schemas,
            non_think_mode=enable_prompt_non_think,
            max_total_tokens=max_total_tokens,
            env_client=env_client,
            lease_id=lease_id,
            run_context=run_ctx,
            task_meta=task_meta,
        )
        agent_runner.reset(user_msg)
        agent_runner.set_max_parse_errors(terminal_max_parse_errors)
        agent_runner.set_max_iterations(terminal_max_iterations)

        # Loop
        interactions: List[Interaction] = []
        final_model_response = None
        final_response = None
        reached_iteration_limit = False
        reached_parse_error_limit = False
        previous_turn_uncertainty_score: float | None = None
        turn_uncertainty_records: list[dict[str, Any]] = []

        while True:
            context_result: TurnContext = await agent_runner.get_turn_context()
            if context_result.terminated_response is not None:
                logger.warning("%s Rollout pre-terminated before model turn.", _log_tag)
                final_response = context_result.terminated_response
                break
            if context_result.context_messages is None:
                logger.warning("%s Rollout context is empty; aborting loop.", _log_tag)
                break


            turn_state: TurnResult = await agent_runner.run_model_turn(
                context_result.context_messages
            )
            turn_interactions = (
                getattr(turn_state, "interactions", None) or [turn_state.interaction]
            )
            turn_uncertainties: list[dict[str, Any]] = []
            for it in turn_interactions:
                uncertainty = _turn_uncertainty_metrics(
                    it,
                    previous_turn_score=previous_turn_uncertainty_score,
                )
                if uncertainty:
                    turn_uncertainties.append(uncertainty)
                    score = _finite_float(uncertainty.get("turn_level_score"))
                    if score is not None:
                        previous_turn_uncertainty_score = score
            turn_uncertainty_records.extend(turn_uncertainties)
            interaction = turn_interactions[-1]
            turn_idx = int(interaction.turn_idx)
            interactions.extend(turn_interactions)
            sdk_tool_calls = getattr(turn_state.model_response, "tool_calls", None)
            sdk_tool_calls_count = getattr(
                turn_state.model_response,
                "tool_calls_count",
                len(sdk_tool_calls or []),
            )

            current_turn_record: dict[str, Any] = {
                "turn_idx": turn_idx,
                "harness_option": agent_type,
                "context_messages": context_result.context_messages,
                "assistant_output": interaction.output_text or "",
                "finish_reason": interaction.finish_reason,
                "latency_ms": float(interaction.latency_ms),
                "n_input_tokens": len(interaction.input_ids or []),
                "n_output_tokens": len(interaction.output_token_ids or []),
                "parse_error_recorded": bool(turn_state.parse_error_recorded),
                "sdk_model_turns": [
                    {
                        "turn_idx": int(it.turn_idx),
                        "assistant_output": it.output_text or "",
                        "finish_reason": it.finish_reason,
                        "latency_ms": float(it.latency_ms),
                        "n_input_tokens": len(it.input_ids or []),
                        "n_output_tokens": len(it.output_token_ids or []),
                        **(
                            {"uncertainty": uncertainty}
                            if uncertainty
                            else {}
                        ),
                    }
                    for it, uncertainty in zip(
                        turn_interactions,
                        turn_uncertainties or [{} for _ in turn_interactions],
                    )
                ],
                "sdk_tool_calls": _jsonable(sdk_tool_calls) if sdk_tool_calls else [],
                "sdk_tool_calls_count": int(sdk_tool_calls_count or 0),
                "tool_calls": [],
            }
            if turn_uncertainties:
                current_turn_record["uncertainty"] = turn_uncertainties[-1]
            if sdk_tool_calls:
                for call in _jsonable(sdk_tool_calls):
                    if isinstance(call, dict):
                        normalized_call = dict(call)
                        normalized_call.setdefault("source", "harness-sdk")
                        current_turn_record["tool_calls"].append(normalized_call)
            turn_records.append(current_turn_record)

            if prm_agent is not None:
                tool_calls_for_prm = [
                    {"tool_name": tc.tool_name, "args": tc.args}
                    for tc in (turn_state.tool_call_requests or [])
                ]
                if not tool_calls_for_prm and sdk_tool_calls:
                    tool_calls_for_prm = [
                        call for call in _jsonable(sdk_tool_calls)
                        if isinstance(call, dict)
                    ]
                prm_agent.record_model_turn(
                    turn_idx,
                    assistant_text=interaction.output_text or "",
                    tool_calls=tool_calls_for_prm or None,
                    parse_error_recorded=turn_state.parse_error_recorded,
                    finish_reason=interaction.finish_reason,
                )

            if turn_state.terminated_response is not None:
                logger.warning(
                    "%s Rollout terminated during model turn %d.", _log_tag, turn_idx
                )
                final_response = turn_state.terminated_response
                break
            if turn_state.model_response is None:
                logger.warning(
                    "%s Model turn %d returned empty model_response.",
                    _log_tag,
                    turn_idx,
                )
                break

            should_continue_loop = False
            if tool_call_requests := turn_state.tool_call_requests:
                logger.info(
                    "%s Turn %d: executing %d tool call(s).",
                    _log_tag,
                    turn_idx,
                    len(tool_call_requests),
                )
                for tool_call_request in tool_call_requests:
                    assert env_client is not None and lease_id is not None
                    await env_client.heartbeat(lease_id)
                    cs_dec_dict: dict[str, Any] | None = None
                    if cs_client is not None:
                        cs_dec = await cs_client.pre_action(
                            tool_call_request.tool_name,
                            tool_call_request.args,
                        )
                        cs_score = _safety_per_turn_score(
                            cs_dec, zero_threshold=safety_zero_threshold
                        )
                        cs_per_call.append((turn_idx, cs_score))
                        if cs_dec is not None:
                            cs_dec_dict = {
                                "decision": cs_dec.decision,
                                "risk_level": cs_dec.risk_level,
                                "composite_score": cs_dec.composite_score,
                                "reason": cs_dec.reason,
                                "safety_score": cs_score,
                            }
                            cs_per_call_full.append(cs_dec_dict)
                    raw_result = await env_client.exec_tool(
                        lease_id,
                        tool_call_request.tool_name,
                        tool_call_request.args,
                    )
                    agent_runner.record_tool_result(tool_call_request, raw_result)
                    if prm_agent is not None:
                        prm_agent.record_tool_result(
                            turn_idx, tool_call_request, raw_result
                        )
                    current_turn_record["tool_calls"].append({
                        "tool_call_id": getattr(tool_call_request, "tool_call_id", None),
                        "tool_name": tool_call_request.tool_name,
                        "args": tool_call_request.args,
                        "result": raw_result[:4096] if isinstance(raw_result, str) else str(raw_result)[:4096],
                        "clawsentry": cs_dec_dict,
                    })
                should_continue_loop = True

            if turn_state.parse_error_recorded:
                logger.warning(
                    "%s Turn %d: tool-call parse error.",
                    _log_tag,
                    turn_idx,
                )
                should_continue_loop = True

            if prm_agent is not None:
                task = asyncio.create_task(prm_agent.judge_turn(turn_idx))
                prm_pending.append((turn_idx, task))

            if should_continue_loop:
                if (
                    turn_state.parse_error_recorded
                    and agent_runner.reached_parse_error_limit()
                ):
                    logger.error(
                        "%s Max parse errors (%d) reached at turn %d.",
                        _log_tag,
                        agent_runner.max_parse_errors,
                        turn_idx,
                    )
                    reached_parse_error_limit = True
                    final_model_response = turn_state.model_response
                    break
                if agent_runner.reached_iteration_limit():
                    logger.warning(
                        "%s Max iterations (%d) reached.",
                        _log_tag,
                        agent_runner.max_iterations,
                    )
                    reached_iteration_limit = True
                    final_model_response = turn_state.model_response
                    break
                continue

            if (
                task_meta.get("data_source") == "tau2"
                and tau2_conversation_mode == "non_solo"
                and env_client is not None
                and lease_id is not None
            ):
                follow_up = await env_client.agent_reply(
                    lease_id,
                    interaction.output_text or "",
                )
                follow_up_message = str(follow_up.get("user_message", "") or "").strip()
                if follow_up.get("continue") and follow_up_message:
                    agent_runner.record_user_message(follow_up_message)
                    current_turn_record["env_user_message"] = follow_up_message
                    if agent_runner.reached_iteration_limit():
                        logger.warning(
                            "%s Max iterations (%d) reached after non-solo follow-up.",
                            _log_tag,
                            agent_runner.max_iterations,
                        )
                        reached_iteration_limit = True
                        final_model_response = turn_state.model_response
                        break
                    continue

            final_model_response = turn_state.model_response
            break

        if final_response is None and final_model_response is not None:
            final_response = agent_runner.finalize_response(final_model_response)

        if final_response is None:
            logger.error(
                "%s No final response produced; mark sample aborted.", _log_tag
            )
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            return [sample]

        finish_reasons = final_response.info.get("termination_reasons", [])
        is_aborted = not final_response.msg

        if final_response.terminated and "max_tokens_exceeded" in finish_reasons:
            status = Sample.Status.TRUNCATED
        elif reached_iteration_limit:
            status = Sample.Status.TRUNCATED
        elif reached_parse_error_limit:
            status = Sample.Status.FAILED
        elif is_aborted:
            status = Sample.Status.ABORTED
        else:
            status = Sample.Status.COMPLETED
        logger.info(
            "%s Rollout finished: status=%s turns=%d parse_errors=%d",
            _log_tag,
            status,
            agent_runner.model_turn_count,
            agent_runner.parse_error_count,
        )

        # Evaluation & Reward
        reward = 0.0
        eval_error: str | None = None
        eval_details: dict[str, Any] | None = None
        deferred_sweverified = (
            data_source == "sweverified"
            and _env_bool("SWEBENCH_DEFER_GRADING", False)
        )

        if (not is_aborted) and (
            status != Sample.Status.FAILED or deferred_sweverified
        ):
            try:
                assert env_client is not None and lease_id is not None
                await env_client.heartbeat(lease_id)
                eval_payload = None
                if data_source in _DIRECT_SCORE_DATA_SOURCES:
                    eval_payload = _build_agent_safetybench_eval_payload(
                        task_meta=task_meta,
                        turn_records=turn_records,
                        final_response=final_response,
                        interactions=interactions,
                        status=status,
                        parse_error_count=agent_runner.parse_error_count,
                    )
                elif deferred_sweverified:
                    eval_payload = {"swebench_defer_grading": True}
                raw_score = await env_client.evaluate(lease_id, trajectory=eval_payload)
                reward = float(raw_score)
                eval_details = _last_eval_details(env_client)
                logger.info("%s Evaluation reward=%.4f", _log_tag, reward)
                if eval_details:
                    reason = eval_details.get("reason")
                    base = eval_details.get("base")
                    split = _safety_split_from_meta(task_meta)
                    logger.info(
                        "%s Reward details: source=%s split=%s mode=%s reason=%s base=%s "
                        "refused=%s verbal_refused=%s tools=%s turns=%s parse_errors=%s",
                        _log_tag,
                        data_source or "seta",
                        split,
                        eval_details.get("mode"),
                        reason,
                        base,
                        eval_details.get("refused"),
                        eval_details.get("verbal_refused", eval_details.get("text_refused")),
                        eval_details.get("n_tool_calls"),
                        eval_details.get("n_turns"),
                        eval_details.get("parse_errors"),
                    )
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"
                status = Sample.Status.FAILED
                reward = 0.0
                logger.error(
                    "%s Evaluation failed, marking FAILED: %s",
                    _log_tag,
                    eval_error,
                )

        if not interactions:
            logger.warning("%s No interactions recorded; remove sample.", _log_tag)
            sample.status = status
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            return [sample]

        trajectory_uncertainty = _summarize_turn_uncertainty(
            turn_uncertainty_records,
            run_ctx=run_ctx,
        )
        if trajectory_uncertainty:
            sample.metadata["trajectory_uncertainty"] = trajectory_uncertainty
            mean_uncertainty = _finite_float(
                trajectory_uncertainty.get("mean_turn_level_uncertainty")
            )
            mean_delta = _finite_float(trajectory_uncertainty.get("mean_abs_score_delta"))
            logger.info(
                "%s Turn uncertainty: available=%s/%s mean_nll=%s "
                "mean_abs_delta=%s low_progress=%s/%s",
                _log_tag,
                trajectory_uncertainty.get("available_turn_count"),
                trajectory_uncertainty.get("turn_count"),
                f"{mean_uncertainty:.4f}" if mean_uncertainty is not None else "n/a",
                f"{mean_delta:.4f}" if mean_delta is not None else "n/a",
                trajectory_uncertainty.get("low_progress_turn_count"),
                trajectory_uncertainty.get("available_turn_count"),
            )

        if prm_agent is not None and prm_pending:
            for turn_idx, prm_task in prm_pending:
                try:
                    output_text, score = await prm_task
                    prm_turn_scores[turn_idx] = float(score)
                    prm_turn_details.append(
                        {
                            "turn_idx": turn_idx,
                            "score": float(score),
                            "output_text": output_text,
                        }
                    )
                    logger.info(
                        "%s PRM judge turn %d score=%.4f, output_text=%s",
                        _log_tag,
                        turn_idx,
                        float(score),
                        output_text.replace("\n", ""),
                    )
                except Exception as exc:
                    logger.warning(
                        "%s PRM judge failed for turn %d (ignored): %s",
                        _log_tag,
                        turn_idx,
                        exc,
                    )
                    prm_turn_scores[turn_idx] = 0.0
                    prm_turn_details.append(
                        {"turn_idx": turn_idx, "score": 0.0, "error": str(exc)}
                    )

        if prm_agent is not None:
            sample.metadata["prm"] = {
                "enabled": True,
                "coef": prm_coef,
                "turn_scores": prm_turn_scores,
                "turn_details": prm_turn_details,
            }

        safety_turn_scores: dict[int, float] | None = None
        if cs_client is not None:
            cs_summary = await cs_client.fetch_summary()
            per_call_scores = [score for (_idx, score) in cs_per_call]
            safety_traj = _safety_trajectory_score(
                per_call_scores,
                cs_summary,
                summary_weight=safety_summary_weight,
                zero_threshold=safety_zero_threshold,
            )
            turn_indices = [it.turn_idx for it in interactions]
            safety_turn_scores = _safety_broadcast(safety_traj, turn_indices)
            cs_stats = cs_client.stats()
            sample.metadata["safety"] = {
                "enabled": True,
                "coef": safety_coef,
                "summary_weight": safety_summary_weight,
                "zero_threshold": safety_zero_threshold,
                "trajectory_score": safety_traj,
                "per_call_scores": cs_per_call,
                "summary_composite_score": (
                    cs_summary.composite_score if cs_summary is not None else None
                ),
                "summary_dimensions": (
                    cs_summary.dimensions if cs_summary is not None else None
                ),
                "n_calls": cs_stats["calls"],
                "n_errors": cs_stats["errors"],
                "decisions": cs_stats["decisions"],
            }
            logger.info(
                "%s ClawSentry trajectory_score=%.4f calls=%d errors=%d",
                _log_tag,
                safety_traj,
                cs_stats["calls"],
                cs_stats["errors"],
            )

        # Build training samples
        dapo_overlong_cfg = _dapo_overlong_cfg(args)
        samples = _build_samples(
            interactions=interactions,
            base_sample=sample,
            outcome=reward,
            status=status,
            prm_turn_scores=(prm_turn_scores if prm_agent is not None else None),
            prm_coef=prm_coef,
            safety_turn_scores=safety_turn_scores,
            safety_coef=safety_coef,
            discount=1.0,
            encourage=False,
            outcome_is_score=(data_source in _DIRECT_SCORE_DATA_SOURCES),
            penalize_short_response=(data_source not in _DIRECT_SCORE_DATA_SOURCES),
            dapo_overlong_cfg=dapo_overlong_cfg,
        )
        if dapo_overlong_cfg is not None:
            logger.info(
                "%s DAPO overlong cfg: max_resp_len=%s buffer_len=%s expected_len=%s penalty_factor=%s",
                _log_tag,
                dapo_overlong_cfg["max_resp_len"],
                dapo_overlong_cfg["buffer_len"],
                dapo_overlong_cfg["expected_len"],
                dapo_overlong_cfg["penalty_factor"],
            )

        # ── Exploration: add intrinsic + safety + LP-RND + CDE actor bonuses (no-op when disabled) ────
        if (
            _EXPLORE_INTRINSIC_ENABLED
            or _EXPLORE_SAFETY_FILTER_ENABLED
            or _EXPLORE_LPRND_ENABLED
            or _EXPLORE_CDE_ACTOR_ENABLED
            or _AGENT57_CONFIG.active
        ):
            _intr_bonus = _explore_intrinsic_bonus(turn_records)
            _intr_schedule_multiplier = _explore_schedule_multiplier(
                _EXPLORE_INTRINSIC_SCHEDULE,
                run_ctx.train_step,
                _EXPLORE_INTRINSIC_DECAY_STEPS,
            )
            _intr_effective_coef = _EXPLORE_INTRINSIC_COEF * _intr_schedule_multiplier
            _intr_scaled = _intr_bonus * _intr_effective_coef
            _safe_penalty = _explore_safety_penalty(turn_records)
            _lprnd_raw = _explore_lprnd_bonus(interactions)
            _lprnd_schedule_multiplier = _explore_schedule_multiplier(
                _EXPLORE_LPRND_SCHEDULE,
                run_ctx.train_step,
                _EXPLORE_LPRND_DECAY_STEPS,
            )
            _lprnd_effective_coef = _EXPLORE_LPRND_COEF * _lprnd_schedule_multiplier
            _lprnd_bonus = _lprnd_raw * _lprnd_effective_coef
            try:
                _agent57_arm_id = int(
                    (sample.metadata or {}).get(
                        "agent57_arm_id",
                        int(sample.index or 0) % max(1, _AGENT57_CONFIG.k),
                    )
                )
            except (TypeError, ValueError):
                _agent57_arm_id = 0
            _agent57_lifelong_metadata = dict(sample.metadata or {})
            _agent57_lifelong_metadata.setdefault("data_source", data_source)
            _agent57_metrics = _agent57_compute_lifelong_bonus(
                config=_AGENT57_CONFIG,
                arm_id=_agent57_arm_id,
                actions=_iter_explore_actions(turn_records),
                turn_records=turn_records,
                status=status,
                parse_error_count=agent_runner.parse_error_count,
                metadata=_agent57_lifelong_metadata,
            )
            _agent57_bonus = float(
                _agent57_metrics.get("explore_agent57_lifelong_bonus", 0.0) or 0.0
            )
            if (
                _AGENT57_CONFIG.active
                and _AGENT57_CONFIG.combine_mode == "ngu_lite"
            ):
                _agent57_episodic = (
                    _intr_bonus
                    if _AGENT57_CONFIG.ngu_episodic_source == "intrinsic"
                    else _explore_episode_signature_novelty(
                        turn_records,
                        reducer=_AGENT57_CONFIG.ngu_episodic_reducer,
                    )
                )
                _agent57_ngu_metrics = _agent57_compute_ngu_lite_bonus(
                    config=_AGENT57_CONFIG,
                    arm_id=_agent57_arm_id,
                    episodic_novelty=_agent57_episodic,
                    lifelong_raw=float(
                        _agent57_metrics.get(
                            "explore_agent57_lifelong_raw", 0.0
                        )
                        or 0.0
                    ),
                    lifelong_eligible=bool(
                        _agent57_metrics.get(
                            "explore_agent57_lifelong_eligible", 0.0
                        )
                    ),
                    trust_gate=float(
                        _agent57_metrics.get("explore_agent57_trust", 1.0) or 0.0
                    ),
                    life_mod_override=_agent57_metrics.get(
                        "explore_agent57_ngu_life_mod"
                    ),
                )
                _agent57_metrics.update(_agent57_ngu_metrics)
                _agent57_metrics.update(_AGENT57_LAST_EPISODIC_STATS)
                _agent57_bonus = float(
                    _agent57_ngu_metrics.get("explore_agent57_ngu_bonus", 0.0)
                    or 0.0
                )
            _base_score_values = []
            for _sample in samples:
                if isinstance(_sample.reward, dict) and "score" in _sample.reward:
                    try:
                        _base_score_values.append(float(_sample.reward["score"]))
                    except (TypeError, ValueError):
                        pass
            _base_score_mean = (
                sum(_base_score_values) / len(_base_score_values)
                if _base_score_values
                else 0.0
            )
            _agent57_dataset_name = str(data_source or "").strip().lower()
            _agent57_normalized_score_values = []
            if _agent57_dataset_name == "seta":
                for _sample in samples:
                    if not isinstance(_sample.reward, dict):
                        continue
                    _raw_score = _sample.reward.get(
                        "raw_score",
                        _sample.reward.get("accuracy"),
                    )
                    try:
                        _agent57_normalized_score_values.append(float(_raw_score))
                    except (TypeError, ValueError):
                        pass
            _agent57_normalized_score_mean = (
                sum(_agent57_normalized_score_values)
                / len(_agent57_normalized_score_values)
                if _agent57_normalized_score_values
                else None
            )
            _cde_actor = _explore_cde_actor_metrics(
                interactions,
                _base_score_mean,
                run_ctx.train_step,
            )
            _cde_actor_bonus = _cde_actor["bonus"]
            _intr_for_total = (
                0.0
                if (
                    _AGENT57_CONFIG.active
                    and _AGENT57_CONFIG.combine_mode == "ngu_lite"
                )
                else _intr_scaled
            )
            _explore_total = (
                _intr_for_total
                + _safe_penalty
                + _lprnd_bonus
                + _agent57_bonus
                + _cde_actor_bonus
            )
            _explore_score_bonus = _explore_score_bonus_from_components(
                _EXPLORE_SCORE_BONUS_COMPONENTS,
                intrinsic=_intr_for_total,
                safety=_safe_penalty,
                lprnd=_lprnd_bonus,
                agent57=_agent57_bonus,
                cde_actor=_cde_actor_bonus,
            )
            _explore_debug = _explore_debug_metrics(
                status=status,
                base_score_mean=_base_score_mean,
                total_bonus=_explore_total,
                intrinsic_scaled=_intr_for_total,
                safety_penalty=_safe_penalty,
                lprnd_bonus=_lprnd_bonus,
                agent57_bonus=_agent57_bonus,
                cde_actor=_cde_actor,
                turn_records=turn_records,
                parse_error_count=agent_runner.parse_error_count,
            )
            _agent57_record_arm_event(
                config=_AGENT57_CONFIG,
                arm_id=_agent57_arm_id,
                base_score=_base_score_mean,
                final_score=_base_score_mean + _explore_score_bonus,
                status=status,
                parse_error_count=agent_runner.parse_error_count,
                bonus=_agent57_bonus,
                dataset=data_source,
                normalized_base_score=_agent57_normalized_score_mean,
                success_score=_agent57_normalized_score_mean,
                infra_failure=eval_error is not None,
            )
            for s in samples:
                if isinstance(s.reward, dict) and "score" in s.reward:
                    s.reward["score"] += _explore_score_bonus
                    s.reward["explore_intrinsic"] = _intr_bonus
                    s.reward["explore_intrinsic_scaled"] = _intr_scaled
                    s.reward["explore_intrinsic_in_total"] = _intr_for_total
                    s.reward["explore_intrinsic_coef"] = _EXPLORE_INTRINSIC_COEF
                    s.reward["explore_intrinsic_effective_coef"] = _intr_effective_coef
                    s.reward["explore_intrinsic_schedule"] = _EXPLORE_INTRINSIC_SCHEDULE
                    s.reward["explore_intrinsic_decay_steps"] = _EXPLORE_INTRINSIC_DECAY_STEPS
                    s.reward["explore_intrinsic_reducer"] = _EXPLORE_INTRINSIC_REDUCER
                    s.reward["explore_intrinsic_schedule_multiplier"] = _intr_schedule_multiplier
                    s.reward["explore_intrinsic_granularity"] = _EXPLORE_INTRINSIC_GRANULARITY
                    s.reward["explore_intrinsic_scope"] = _EXPLORE_INTRINSIC_SCOPE
                    s.reward["explore_safety_penalty"] = _safe_penalty
                    s.reward["explore_lprnd"] = _lprnd_bonus
                    s.reward["explore_lprnd_raw"] = _lprnd_raw
                    s.reward["explore_lprnd_coef"] = _EXPLORE_LPRND_COEF
                    s.reward["explore_lprnd_effective_coef"] = _lprnd_effective_coef
                    s.reward["explore_lprnd_schedule"] = _EXPLORE_LPRND_SCHEDULE
                    s.reward["explore_lprnd_decay_steps"] = _EXPLORE_LPRND_DECAY_STEPS
                    s.reward["explore_lprnd_schedule_multiplier"] = _lprnd_schedule_multiplier
                    if _AGENT57_CONFIG.active:
                        if not isinstance(s.metadata, dict):
                            s.metadata = {}
                        s.reward.update(_agent57_metrics)
                        try:
                            _turn_idx = int(s.metadata.get("turn_idx", -1))
                        except (TypeError, ValueError):
                            _turn_idx = -1
                        _turn_episodic = float(
                            _AGENT57_LAST_EPISODIC_BY_TURN.get(_turn_idx, 0.0)
                        )
                        _turn_life_mod = float(
                            _agent57_metrics.get("explore_agent57_ngu_life_mod", 1.0)
                            or 1.0
                        )
                        s.reward["explore_agent57_turn_episodic"] = _turn_episodic
                        s.reward["explore_agent57_turn_intrinsic_signal"] = (
                            _turn_episodic * _turn_life_mod
                        )
                        s.metadata["agent57"] = {
                            "enabled": bool(_AGENT57_CONFIG.enabled),
                            "arm_id": _agent57_arm_id,
                            "k": _AGENT57_CONFIG.k,
                            "beta": _AGENT57_CONFIG.beta_for_arm(_agent57_arm_id),
                            "controller": _AGENT57_CONFIG.controller,
                            "combine_mode": _AGENT57_CONFIG.combine_mode,
                            "lifelong_enabled": bool(_AGENT57_CONFIG.lifelong_enabled),
                            "lifelong_backend": _AGENT57_CONFIG.lifelong_backend,
                            "lifelong_key_version": _AGENT57_CONFIG.lifelong_key_version,
                            "bonus": _agent57_bonus,
                            "lifelong_bonus": _agent57_metrics.get(
                                "explore_agent57_lifelong_bonus", 0.0
                            ),
                            "ngu_bonus": _agent57_metrics.get(
                                "explore_agent57_ngu_bonus", 0.0
                            ),
                            "lifelong_raw": _agent57_metrics.get(
                                "explore_agent57_lifelong_raw", 0.0
                            ),
                            "lifelong_eligible": _agent57_metrics.get(
                                "explore_agent57_lifelong_eligible", 0.0
                            ),
                        }
                    if _EXPLORE_CDE_ACTOR_ENABLED:
                        s.reward["explore_cde_actor_bonus"] = _cde_actor_bonus
                        s.reward["explore_cde_actor_log_ppl"] = _cde_actor["log_ppl"]
                        s.reward["explore_cde_actor_omega"] = _cde_actor["omega"]
                        s.reward["explore_cde_actor_alpha"] = _cde_actor["alpha"]
                        s.reward["explore_cde_actor_kappa"] = _cde_actor["kappa"]
                        s.reward["explore_cde_actor_reward_gate"] = _EXPLORE_CDE_ACTOR_REWARD_GATE
                        s.reward["explore_cde_actor_eligible"] = _cde_actor["eligible"]
                        s.reward["explore_cde_actor_decay_steps"] = _cde_actor["decay_steps"]
                        s.reward["explore_cde_actor_base_mean"] = _cde_actor["base_score_mean"]
                        s.reward["explore_cde_actor_base_magnitude"] = _cde_actor["base_score_magnitude"]
                        s.reward["explore_cde_actor_cap"] = _cde_actor["cap"]
                        s.reward["explore_cde_actor_scaled"] = _cde_actor["scaled"]
                        s.reward["explore_cde_actor_clipped"] = _cde_actor["clipped"]
                    s.reward["explore_total_bonus"] = _explore_score_bonus
                    s.reward["explore_score_bonus"] = _explore_score_bonus
                    s.reward["explore_all_bonus"] = _explore_total
                    s.reward["explore_score_bonus_components"] = _EXPLORE_SCORE_BONUS_COMPONENTS
                    s.reward.update(_explore_debug)
                    _sync_reward_aliases(s.reward)

        turn_uncertainty_by_idx = {
            int(r["turn_idx"]): r
            for r in turn_uncertainty_records
            if isinstance(r, dict) and r.get("turn_idx") is not None
        }
        for s in samples:
            s.metadata["train_step"] = run_ctx.train_step
            s.metadata["rollout_step"] = run_ctx.rollout_step
            s.metadata["rollout_id"] = run_ctx.rollout_id
            s.metadata["uid"] = run_ctx.uid
            s.metadata["model_turn_count"] = agent_runner.model_turn_count
            s.metadata["parse_error_count"] = agent_runner.parse_error_count
            s.metadata["data_source"] = data_source or s.metadata.get("data_source")
            s.metadata["safety_split"] = _safety_split_from_meta(task_meta)
            claude_backend = str(os.getenv("CLAUDE_CODE_LLM_BACKEND", "sglang")).strip().lower()
            claude_sglang_backend = claude_backend.replace("_", "-") in {
                "sglang",
                "qwen",
                "qwen-sglang",
                "local",
                "local-sglang",
            }
            if agent_type == "claude-code" and _env_flag(
                "CLAUDE_CODE_MARK_NON_TRAINABLE",
                not claude_sglang_backend,
            ):
                s.remove_sample = True
                s.metadata["non_trainable"] = True
                s.metadata["non_trainable_reason"] = (
                    "claude-code CLI uses an external model path without "
                    "terminal-rl policy logprobs"
                )
            if trajectory_uncertainty:
                s.metadata["trajectory_uncertainty"] = trajectory_uncertainty
            turn_uncertainty = turn_uncertainty_by_idx.get(
                int(s.metadata.get("turn_idx", -1))
            )
            if turn_uncertainty:
                s.metadata["turn_uncertainty"] = turn_uncertainty
            if eval_details is not None:
                s.metadata["reward_details"] = _jsonable(eval_details)
            if eval_error is not None:
                s.metadata["evaluation_failed"] = True
                s.metadata["evaluation_error"] = eval_error
        _mark_non_trainable_samples(samples)

        _save_rollout_artifacts(
            task_spec=task_spec,
            run_ctx=run_ctx,
            sampling_params=sampling_params,
            sample=sample,
            samples=samples,
            status=status,
            raw_score=reward,
            eval_error=eval_error,
            turn_records=turn_records,
            safety_meta=sample.metadata.get("safety") if sample.metadata else None,
            prm_meta=sample.metadata.get("prm") if sample.metadata else None,
            safety_coef=safety_coef,
            prm_coef=prm_coef,
            trajectory_save_interval=traj_save_interval,
        )

        if remote_env_admission_key is not None:
            _task_circuit_record_success(task_key)
        return samples

    except Exception as exc:
        if _uses_remote_terminal_env(task_meta):
            _task_circuit_record_failure(task_key, exc)
        log_traceback = _env_bool("TERMINAL_RL_GENERATE_FAILURE_TRACEBACK", False)
        logger.error(
            "%s Generate failed (%s): %s%s",
            _log_tag,
            type(exc).__name__,
            exc,
            "" if log_traceback else " (set TERMINAL_RL_GENERATE_FAILURE_TRACEBACK=1 for traceback)",
            exc_info=log_traceback,
        )
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = dict(metadata)
        sample.metadata["generate_failed"] = True
        sample.metadata["generate_error_type"] = type(exc).__name__
        sample.metadata["generate_error"] = str(exc)
        sample.status = Sample.Status.FAILED
        sample.remove_sample = True
        sample.reward = {"score": 0.0}
        _sync_reward_aliases(sample.reward)

        eos = state.tokenizer.eos_token_id
        if eos is None:
            sample.tokens = []
            sample.response_length = 0
            sample.rollout_log_probs = []
            sample.loss_mask = []
        else:
            sample.tokens = [eos, eos]
            sample.response_length = 1
            sample.rollout_log_probs = [0.0]
            sample.loss_mask = [0]

        failed_turn_records = list(turn_records)
        if not failed_turn_records and _env_bool("TRAJECTORY_SAVE_FAILED_SHORT_ROLLOUTS", False):
            agent_artifacts: dict[str, Any] = {}
            rollout_agent = getattr(agent_runner, "_rollout_agent", None) if agent_runner is not None else None
            for attr, key in (
                ("_local_run_dir", "local_run_dir"),
                ("_workspace", "workspace"),
                ("_stdout_path", "stdout_path"),
                ("_stderr_path", "stderr_path"),
                ("_tool_log_path", "tool_calls_path"),
                ("_qwen_records_path", "qwen_gateway_records_path"),
            ):
                value = getattr(rollout_agent, attr, None) if rollout_agent is not None else None
                if value is not None:
                    agent_artifacts[key] = str(value)
            failed_turn_records = [
                {
                    "turn_idx": 0,
                    "harness_option": locals().get("agent_type", None),
                    "context_messages": [{"role": "user", "content": user_msg}] if "user_msg" in locals() else [],
                    "assistant_output": "",
                    "finish_reason": "generate_failed",
                    "latency_ms": 0.0,
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "parse_error_recorded": False,
                    "tool_calls": [],
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "agent_artifacts": agent_artifacts,
                }
            ]
        if failed_turn_records:
            _save_rollout_artifacts(
                task_spec=task_spec,
                run_ctx=run_ctx,
                sampling_params=sampling_params,
                sample=sample,
                samples=[sample],
                status=sample.status,
                raw_score=0.0,
                eval_error=f"{type(exc).__name__}: {exc}",
                turn_records=failed_turn_records,
                safety_meta=sample.metadata.get("safety") if sample.metadata else None,
                prm_meta=sample.metadata.get("prm") if sample.metadata else None,
                safety_coef=safety_coef,
                prm_coef=prm_coef,
                trajectory_save_interval=traj_save_interval,
            )
        return [sample]

    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception as heartbeat_exc:
                logger.debug(
                    "%s Background heartbeat task ended with error: %s",
                    _log_tag,
                    heartbeat_exc,
                )

        for _turn_idx, t in prm_pending:
            if not t.done():
                t.cancel()

        if cs_client is not None:
            try:
                await cs_client.aclose()
            except Exception as exc:
                logger.debug("ClawSentry aclose ignored: %s", exc)

        if agent_runner is not None:
            try:
                await agent_runner.close()
            except Exception as exc:
                logger.debug("%s Agent runner close ignored: %s", _log_tag, exc)

        try:
            if env_client is not None and lease_id is not None:
                try:
                    close_timeout = _env_float(
                        "ENV_CLOSE_HTTP_TIMEOUT",
                        float(timeouts.close_session) + 30.0,
                    )
                    close_sem = (
                        _remote_env_close_semaphore()
                        if remote_env_admission_key is not None
                        else None
                    )
                    if close_sem is None:
                        await _await_with_optional_timeout(
                            env_client.close(lease_id),
                            close_timeout,
                            op_name=f"{_log_tag} env close",
                        )
                    else:
                        async with close_sem:
                            await _await_with_optional_timeout(
                                env_client.close(lease_id),
                                close_timeout,
                                op_name=f"{_log_tag} env close",
                            )
                except Exception as exc:
                    logger.debug(
                        "%s Best-effort remote close failed lease=%s: %s",
                        _log_tag,
                        lease_id,
                        exc,
                    )
        finally:
            if remote_env_admission_key is not None:
                await _release_remote_env_admission(remote_env_admission_key)
