from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

from agentic_rl.core.types import RunContext, TaskSpec
from agentic_rl.environments.client import TerminalEnvClient
from agentic_rl.rollout.sample_builder import _make_task_spec

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


async def _await_with_optional_timeout(awaitable, timeout: float, *, op_name: str):
    if timeout <= 0:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{op_name} timed out after {timeout:.1f}s") from exc


def _uses_local_agent_safetybench_env(task_meta: Dict[str, Any] | None) -> bool:
    return isinstance(task_meta, dict) and task_meta.get("data_source") == "agent_safetybench" and os.getenv("AGENT_SAFETYBENCH_REMOTE_ENV", "0") != "1"


def _uses_local_agentharm_env(task_meta: Dict[str, Any] | None) -> bool:
    return isinstance(task_meta, dict) and task_meta.get("data_source") == "agentharm" and os.getenv("AGENTHARM_REMOTE_ENV", "0") != "1"


def _uses_local_tau2_env(task_meta: Dict[str, Any] | None) -> bool:
    return isinstance(task_meta, dict) and task_meta.get("data_source") == "tau2" and os.getenv("TAU2_REMOTE_ENV", "0") != "1"


def _normalize_tau2_conversation_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "solo").strip().lower()
    if mode in {"non_solo", "nonsolo", "non-solo"}:
        return "non_solo"
    return "solo"


class _LocalAgentSafetyBenchClient:
    def __init__(self) -> None:
        from agentic_rl.environments.agent_safetybench.runtime import AgentSafetyBenchEnv

        self._env = AgentSafetyBenchEnv()
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(str(run_ctx.get("log_dir", "build_outputs"))),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        return {"user_msg": user_msg, "tool_schemas": tool_schemas}

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


class _LocalAgentHarmClient:
    def __init__(self) -> None:
        from agentic_rl.environments.agentharm.runtime import AgentHarmEnv

        self._env = AgentHarmEnv()
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(str(run_ctx.get("log_dir", "build_outputs"))),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        return {"user_msg": user_msg, "tool_schemas": tool_schemas}

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


class _LocalTau2Client:
    def __init__(self) -> None:
        from agentic_rl.environments.tau2.runtime import Tau2Env

        self._env = Tau2Env()
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(str(run_ctx.get("log_dir", "build_outputs"))),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        return {
            "user_msg": user_msg,
            "tool_schemas": tool_schemas,
            "conversation_mode": _normalize_tau2_conversation_mode(
                task_meta.get("tau2_mode")
            ),
        }

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def agent_reply(self, lease_id: str, assistant_text: str) -> dict[str, Any]:
        _ = lease_id
        return await self._env.handle_agent_reply(assistant_text)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


async def _create_env_client(
    task_spec: TaskSpec,
    run_ctx: RunContext,
    task_meta: Dict[str, Any] | None = None,
) -> tuple[Any, str]:
    if _uses_local_agent_safetybench_env(task_meta):
        logger.info(
            "Using local Agent-SafetyBench env backend for task=%s path=%s",
            task_spec.task_name,
            task_spec.task_path,
        )
        return _LocalAgentSafetyBenchClient(), "local-agent-safetybench"

    if _uses_local_agentharm_env(task_meta):
        logger.info(
            "Using local AgentHarm env backend for task=%s path=%s",
            task_spec.task_name,
            task_spec.task_path,
        )
        return _LocalAgentHarmClient(), "local-agentharm"

    if _uses_local_tau2_env(task_meta):
        logger.info(
            "Using local tau2 env backend for task=%s path=%s",
            task_spec.task_name,
            task_spec.task_path,
        )
        return _LocalTau2Client(), "local-tau2"

    env_server_url = os.getenv("ENV_SERVER_URL", "")
    if not env_server_url:
        raise RuntimeError("ENV_SERVER_URL is empty.")

    env_client = TerminalEnvClient(env_server_url)
    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    request_id = (
        f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:{run_ctx.sample_index}"
    )
    allocate_timeout = _env_float("ENV_ALLOCATE_HTTP_TIMEOUT", 300.0)
    lease = await _await_with_optional_timeout(
        env_client.allocate(task_key=task_key, request_id=request_id),
        allocate_timeout,
        op_name="terminal env allocate",
    )
    lease_id = str(lease["lease_id"])
    logger.info(
        "Using remote terminal env backend lease=%s server=%s", lease_id, env_server_url
    )
    return env_client, lease_id
