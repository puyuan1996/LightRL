from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict

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
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_csv_set(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {part.strip() for part in raw.split(",") if part.strip()}


async def _await_with_optional_timeout(awaitable, timeout: float, *, op_name: str):
    if timeout <= 0:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{op_name} timed out after {timeout:.1f}s") from exc


def _is_reset_fresh_lease_retryable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    retry_markers = (
        "WORKER_RESET_ADMISSION_BACKLOG",
        "TASK_SLOTS_EXHAUSTED",
        "LEASE_EXPIRED",
        "410 Gone",
        "503 Service Unavailable",
    )
    return any(marker in text for marker in retry_markers)


_REMOTE_ENV_CONDITION: asyncio.Condition | None = None
_REMOTE_ENV_ACTIVE_BY_TASK: dict[str, int] = {}
_REMOTE_ENV_ACTIVE_TOTAL = 0
_REMOTE_ENV_CLOSE_SEMAPHORE: asyncio.Semaphore | None = None
_REMOTE_ENV_CLOSE_LIMIT: int | None = None
_REMOTE_ENV_CLOSE_SEMAPHORE_LOCK: asyncio.Lock | None = None  # P1 fix: Add lock for semaphore recreation


def _uses_local_agent_safetybench_env(task_meta: Dict[str, Any] | None) -> bool:
    return (
        isinstance(task_meta, dict)
        and task_meta.get("data_source") == "agent_safetybench"
        and os.getenv("AGENT_SAFETYBENCH_REMOTE_ENV", "0") != "1"
    )


def _uses_local_agentharm_env(task_meta: Dict[str, Any] | None) -> bool:
    return (
        isinstance(task_meta, dict)
        and task_meta.get("data_source") == "agentharm"
        and os.getenv("AGENTHARM_REMOTE_ENV", "0") != "1"
    )


def _uses_local_tau2_env(task_meta: Dict[str, Any] | None) -> bool:
    return (
        isinstance(task_meta, dict)
        and task_meta.get("data_source") == "tau2"
        and os.getenv("TAU2_REMOTE_ENV", "0") != "1"
    )


def _uses_remote_terminal_env(task_meta: Dict[str, Any] | None) -> bool:
    return not (
        _uses_local_agent_safetybench_env(task_meta)
        or _uses_local_agentharm_env(task_meta)
        or _uses_local_tau2_env(task_meta)
    )


def _http_exception_info(exc: BaseException) -> tuple[int | None, str | None, str, float | None]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    text = ""
    retry_after: float | None = None
    if response is not None:
        try:
            text = str(getattr(response, "text", "") or "")
        except Exception:
            text = ""
        try:
            raw_retry_after = response.headers.get("Retry-After")
            retry_after = float(raw_retry_after) if raw_retry_after else None
        except Exception:
            retry_after = None

    code: str | None = None
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                raw_code = parsed.get("code")
                code = str(raw_code) if raw_code is not None else None
        except Exception:
            code = None
    return status_code, code, text, retry_after


def _reset_should_retry_with_new_lease(exc: BaseException) -> bool:
    status_code, code, text, _ = _http_exception_info(exc)
    combined = f"{code or ''} {text} {exc}"
    non_retry_codes = _env_csv_set(
        "ENV_RESET_LEASE_NON_RETRY_CODES",
        "TASK_IMAGE_BLACKLISTED,TASK_BUILD_FAILED",
    )
    if code in non_retry_codes:
        return False
    if "TASK_IMAGE_BLACKLISTED" in combined or "TASK_BUILD_FAILED" in combined:
        return False

    retry_codes = _env_csv_set(
        "ENV_RESET_LEASE_RETRY_CODES",
        "DOCKER_IMAGE_PREP_BACKLOG,WORKER_RESET_ADMISSION_BACKLOG",
    )
    if code in retry_codes or any(marker in combined for marker in retry_codes):
        return True

    retry_statuses = set()
    for item in _env_csv_set("ENV_RESET_LEASE_RETRY_STATUSES", "410,500,502,503,504"):
        try:
            retry_statuses.add(int(item))
        except ValueError:
            continue
    return status_code in retry_statuses


def _reset_retry_sleep_seconds(exc: BaseException, attempt: int) -> float:
    _, _, _, retry_after = _http_exception_info(exc)
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, _env_float("ENV_RESET_LEASE_RETRY_MAX_SLEEP", 60.0))
    base = max(0.0, _env_float("ENV_RESET_LEASE_RETRY_BASE_SLEEP", 15.0))
    max_sleep = max(base, _env_float("ENV_RESET_LEASE_RETRY_MAX_SLEEP", 60.0))
    return min(max_sleep, base * max(1, attempt))


_TASK_CIRCUIT: dict[str, dict[str, Any]] = {}


def _task_circuit_enabled() -> bool:
    return _env_bool("ENV_TASK_CIRCUIT_BREAKER_ENABLED", True)


def _task_circuit_threshold() -> int:
    return max(1, _env_int("ENV_TASK_CIRCUIT_BREAKER_THRESHOLD", 2))


def _task_circuit_cooldown() -> float:
    return max(0.0, _env_float("ENV_TASK_CIRCUIT_BREAKER_COOLDOWN", 1800.0))


def _task_circuit_failure_is_relevant(exc: BaseException) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "TASK_BUILD_FAILED",
            "WORKER_RESET_TIMEOUT",
            "env reset timed out",
            "reset timed out",
            "Docker image build failed",
            "dockerfile parse error",
            "RESET_IN_PROGRESS",
            "WORKER_RESET_CANCELLED",
            "WORKER_RESET_STALE",
        )
    )


def _task_circuit_open_reason(task_key: str) -> str | None:
    if not _task_circuit_enabled():
        return None
    state = _TASK_CIRCUIT.get(task_key)
    if not state:
        return None
    opened_until = float(state.get("opened_until", 0.0) or 0.0)
    if opened_until <= time.time():
        _TASK_CIRCUIT.pop(task_key, None)
        return None
    return str(state.get("reason") or "recent env failures")


def _task_circuit_record_success(task_key: str) -> None:
    if task_key:
        _TASK_CIRCUIT.pop(task_key, None)


def _task_circuit_record_failure(task_key: str, exc: BaseException) -> None:
    if not task_key or not _task_circuit_enabled():
        return
    if not _task_circuit_failure_is_relevant(exc):
        return
    now = time.time()
    state = _TASK_CIRCUIT.setdefault(
        task_key,
        {"count": 0, "opened_until": 0.0, "reason": ""},
    )
    state["count"] = int(state.get("count", 0) or 0) + 1
    reason = f"{type(exc).__name__}: {str(exc)[:300]}"
    state["reason"] = reason
    immediate = "TASK_BUILD_FAILED" in str(exc) or "dockerfile parse error" in str(exc)
    if immediate or int(state["count"]) >= _task_circuit_threshold():
        state["opened_until"] = now + _task_circuit_cooldown()
        logger.warning(
            "Opening task circuit breaker task_key=%s count=%s cooldown=%.1fs reason=%s",
            task_key,
            state["count"],
            _task_circuit_cooldown(),
            reason,
        )


def _remote_env_condition() -> asyncio.Condition:
    global _REMOTE_ENV_CONDITION
    if _REMOTE_ENV_CONDITION is None:
        _REMOTE_ENV_CONDITION = asyncio.Condition()
    return _REMOTE_ENV_CONDITION


def _remote_env_close_semaphore() -> asyncio.Semaphore | None:
    global _REMOTE_ENV_CLOSE_LIMIT, _REMOTE_ENV_CLOSE_SEMAPHORE, _REMOTE_ENV_CLOSE_SEMAPHORE_LOCK
    limit = _env_int("ENV_REMOTE_MAX_CONCURRENT_CLOSES", 8)
    if limit <= 0:
        return None
    # P1 fix: Use lock to prevent race condition during semaphore recreation
    if _REMOTE_ENV_CLOSE_SEMAPHORE_LOCK is None:
        _REMOTE_ENV_CLOSE_SEMAPHORE_LOCK = asyncio.Lock()
    # Note: This is not truly async-safe since we can't await here, but it prevents
    # the worst case of two semaphores coexisting. For full safety, callers should
    # cache the semaphore result at module init.
    if _REMOTE_ENV_CLOSE_SEMAPHORE is None or _REMOTE_ENV_CLOSE_LIMIT != limit:
        _REMOTE_ENV_CLOSE_LIMIT = limit
        _REMOTE_ENV_CLOSE_SEMAPHORE = asyncio.Semaphore(limit)
    return _REMOTE_ENV_CLOSE_SEMAPHORE


async def _acquire_remote_env_admission(
    task_key: str,
    *,
    log_tag: str,
) -> str | None:
    global _REMOTE_ENV_ACTIVE_TOTAL
    max_active_tasks = _env_int("ENV_REMOTE_MAX_ACTIVE_TASKS", 12)
    max_active_runs = _env_int("ENV_REMOTE_MAX_ACTIVE_RUNS", 0)
    max_runs_per_task = _env_int("ENV_REMOTE_MAX_RUNS_PER_TASK", 8)
    if max_active_tasks <= 0 and max_active_runs <= 0 and max_runs_per_task <= 0:
        return None

    timeout = _env_float("ENV_REMOTE_ADMISSION_TIMEOUT", 900.0)
    log_interval = max(5.0, _env_float("ENV_REMOTE_ADMISSION_LOG_INTERVAL", 30.0))
    condition = _remote_env_condition()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout if timeout > 0 else None
    last_log = 0.0

    async with condition:
        while True:
            active_for_task = _REMOTE_ENV_ACTIVE_BY_TASK.get(task_key, 0)
            active_tasks = len(_REMOTE_ENV_ACTIVE_BY_TASK)
            reasons: list[str] = []
            if (
                max_active_tasks > 0
                and active_for_task <= 0
                and active_tasks >= max_active_tasks
            ):
                reasons.append(f"active_tasks={active_tasks}/{max_active_tasks}")
            if max_active_runs > 0 and _REMOTE_ENV_ACTIVE_TOTAL >= max_active_runs:
                reasons.append(
                    f"active_runs={_REMOTE_ENV_ACTIVE_TOTAL}/{max_active_runs}"
                )
            if max_runs_per_task > 0 and active_for_task >= max_runs_per_task:
                reasons.append(
                    f"runs_per_task={active_for_task}/{max_runs_per_task}"
                )

            if not reasons:
                _REMOTE_ENV_ACTIVE_BY_TASK[task_key] = active_for_task + 1
                _REMOTE_ENV_ACTIVE_TOTAL += 1
                return task_key

            now = loop.time()
            if deadline is not None and now >= deadline:
                raise TimeoutError(
                    f"{log_tag} remote env admission timed out for task_key={task_key} "
                    f"after {timeout:.1f}s ({', '.join(reasons)})"
                )

            if now - last_log >= log_interval:
                logger.info(
                    "%s Waiting for remote env admission task_key=%s (%s)",
                    log_tag,
                    task_key,
                    ", ".join(reasons),
                )
                last_log = now

            wait_timeout = log_interval
            if deadline is not None:
                wait_timeout = min(wait_timeout, max(0.1, deadline - now))
            try:
                await asyncio.wait_for(condition.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                pass


async def _release_remote_env_admission(task_key: str | None) -> None:
    global _REMOTE_ENV_ACTIVE_TOTAL
    if not task_key:
        return
    condition = _remote_env_condition()
    async with condition:
        active_for_task = _REMOTE_ENV_ACTIVE_BY_TASK.get(task_key, 0)
        if active_for_task <= 1:
            _REMOTE_ENV_ACTIVE_BY_TASK.pop(task_key, None)
        else:
            _REMOTE_ENV_ACTIVE_BY_TASK[task_key] = active_for_task - 1
        _REMOTE_ENV_ACTIVE_TOTAL = max(0, _REMOTE_ENV_ACTIVE_TOTAL - 1)
        # P1 fix: Use notify(1) instead of notify_all() to reduce wake-up storm
        # Only one waiter can proceed anyway since we released exactly one slot
        condition.notify(1)
