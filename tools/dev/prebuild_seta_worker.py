#!/usr/bin/env python3
"""Prebuild SETA task images through the worker's public lifecycle API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_PRINT_LOCK = threading.Lock()
_INVALID_EVAL_REASONS = {"eval_timeout", "eval_parse_failed", "eval_no_results"}


def _request_json(
    base_url: str,
    path: str,
    payload: dict | None,
    timeout: float,
) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, body


def _load_tasks(path: Path, *, preserve_order: bool = False) -> list[dict]:
    tasks: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metadata = json.loads(line).get("metadata") or {}
        task_name = str(metadata.get("task_name") or "").strip()
        task_path = str(metadata.get("task_path") or "").strip()
        instruction = str(metadata.get("instruction") or "").strip()
        if not task_name or not task_path or not instruction:
            raise ValueError(f"invalid SETA metadata for task {task_name!r}")
        tasks[task_name] = metadata
    if preserve_order:
        return list(tasks.values())
    return [tasks[name] for name in sorted(tasks, key=lambda value: int(value))]


def _schedule_tasks(
    tasks: list[dict],
    *,
    shuffle_seed: int | None,
    skip_first: int,
    limit: int | None,
) -> list[dict]:
    """Match slime's epoch-0 prompt order for deterministic image warmup."""
    scheduled = list(tasks)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(scheduled)
    scheduled = scheduled[skip_first:]
    if limit is not None:
        scheduled = scheduled[:limit]
    return scheduled


def _wait_closed(base_url: str, lease_id: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, body = _request_json(base_url, "/status", None, timeout=10)
        tasks = ((body.get("pool") or {}).get("tasks") or {}) if code < 400 else {}
        present = any(
            lease_id in ((task or {}).get("runs") or {}) for task in tasks.values()
        )
        if code < 400 and not present:
            return True
        time.sleep(1)
    return False


def _allocate_with_capacity_retry(
    base_url: str,
    payload: dict,
    timeout: float,
) -> tuple[int, dict, int]:
    """Mirror the production client's retry semantics for worker backpressure.

    A retiring lease still owns its Docker network after it disappears from the
    active task map. Receiving 429 while that cleanup drains is admission
    backpressure, not a reset/grader attempt, so do not charge it to the task's
    ``max_attempts`` budget.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    retry_count = 0
    while True:
        code, body = _request_json(base_url, "/allocate", payload, timeout=60)
        capacity_code = str(body.get("code") or "") if isinstance(body, dict) else ""
        if code != 429 or capacity_code not in {
            "TOTAL_RUN_SLOTS_EXHAUSTED",
            "TASK_SLOTS_EXHAUSTED",
            "RUN_SLOTS_EXHAUSTED",
        }:
            return code, body, retry_count
        if time.monotonic() >= deadline:
            return code, body, retry_count
        retry_count += 1
        time.sleep(min(30.0, float(2 ** min(retry_count, 5))))


def _evaluation_failure_reason(code: int, body: dict) -> str:
    if code >= 400:
        return f"http_{code}"
    if not body.get("ok"):
        return "response_not_ok"
    score = body.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        return "invalid_score"
    details = body.get("details")
    reason = str(details.get("reason") or "") if isinstance(details, dict) else ""
    return reason if reason in _INVALID_EVAL_REASONS else ""


def _prebuild_one(
    base_url: str,
    metadata: dict,
    max_attempts: int,
    reset_timeout: float,
    evaluate: bool,
    eval_timeout: float,
    close_timeout: float,
) -> dict:
    task_name = str(metadata["task_name"])
    last: dict = {}
    for attempt in range(1, max_attempts + 1):
        request_id = f"seta-prebuild-{task_name}-{uuid.uuid4().hex[:10]}"
        lease_id = ""
        try:
            code, body, admission_retries = _allocate_with_capacity_retry(
                base_url,
                {"task_key": f"seta-prebuild:{task_name}", "request_id": request_id},
                timeout=reset_timeout,
            )
            if code >= 400 or not body.get("ok"):
                last = {
                    "phase": "allocate",
                    "http": code,
                    "admission_retries": admission_retries,
                    "body": body,
                }
                time.sleep(min(30, 2**attempt))
                continue
            lease_id = str(body["lease_id"])
            code, body = _request_json(
                base_url,
                "/reset",
                {
                    "lease_id": lease_id,
                    "task_meta": metadata,
                    "run_ctx": {
                        "uid": request_id,
                        "group_index": -1,
                        "sample_index": -1,
                    },
                    "task_timeouts": {
                        "ensure_image": reset_timeout,
                        "reset_session": 900,
                        "eval": eval_timeout,
                        "close_session": 90,
                    },
                    "request_id": request_id,
                },
                timeout=reset_timeout + 1200,
            )
            last = {"phase": "reset", "http": code, "body": body}
            if code < 400 and body.get("ok"):
                if not evaluate:
                    return {
                        "task_name": task_name,
                        "ok": True,
                        "attempt": attempt,
                        "admission_retries": admission_retries,
                    }
                code, body = _request_json(
                    base_url,
                    "/evaluate",
                    {"lease_id": lease_id, "trajectory": {}},
                    timeout=eval_timeout + 60,
                )
                failure_reason = _evaluation_failure_reason(code, body)
                last = {
                    "phase": "evaluate",
                    "http": code,
                    "reason": failure_reason,
                    "body": body,
                }
                if not failure_reason:
                    return {
                        "task_name": task_name,
                        "ok": True,
                        "attempt": attempt,
                        "admission_retries": admission_retries,
                        "score": float(body["score"]),
                    }
        except Exception as exc:  # retain enough context for an actionable report
            last = {"phase": "exception", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if lease_id:
                try:
                    _request_json(
                        base_url, "/close", {"lease_id": lease_id}, timeout=180
                    )
                    _wait_closed(base_url, lease_id, close_timeout)
                except Exception:
                    pass
        time.sleep(min(30, 2**attempt))
    return {"task_name": task_name, "ok": False, "attempt": max_attempts, **last}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--reset-timeout", type=float, default=1800)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-timeout", type=float, default=1200)
    parser.add_argument("--close-timeout", type=float, default=300)
    parser.add_argument(
        "--task-name",
        action="append",
        help="Limit the preflight to this task name (repeatable).",
    )
    parser.add_argument(
        "--repeat-per-task",
        type=int,
        default=1,
        help="Run concurrent duplicate leases per selected task (default: 1).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help=(
            "Shuffle tasks exactly like slime's first rollout epoch. Use the "
            "training --rollout-seed to warm images in consumption order."
        ),
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip this many tasks after optional shuffling.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Warm at most this many tasks after optional shuffling/skipping.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL artifact; overwritten at start and flushed per row.",
    )
    args = parser.parse_args()

    output_handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")

    def emit(payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with _PRINT_LOCK:
            print(line, flush=True)
            if output_handle is not None:
                output_handle.write(line + "\n")
                output_handle.flush()

    tasks = _load_tasks(
        args.dataset,
        preserve_order=args.shuffle_seed is not None,
    )
    if args.task_name:
        selected = set(args.task_name)
        tasks = [task for task in tasks if str(task["task_name"]) in selected]
        missing = selected - {str(task["task_name"]) for task in tasks}
        if missing:
            parser.error(f"task names not found in dataset: {sorted(missing)}")
    if args.skip_first < 0:
        parser.error("--skip-first must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    tasks = _schedule_tasks(
        tasks,
        shuffle_seed=args.shuffle_seed,
        skip_first=args.skip_first,
        limit=args.limit,
    )
    if args.repeat_per_task < 1:
        parser.error("--repeat-per-task must be at least 1")
    tasks = [task for task in tasks for _ in range(args.repeat_per_task)]
    if args.dry_run:
        print(json.dumps({"task_count": len(tasks), "task_names": [task["task_name"] for task in tasks]}))
        return 0

    health_code, health = _request_json(args.worker_url, "/healthz", None, 10)
    if health_code >= 400 or not health.get("ok"):
        print(json.dumps({"ok": False, "phase": "healthz", "http": health_code, "body": health}))
        return 2

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _prebuild_one,
                args.worker_url,
                task,
                args.max_attempts,
                args.reset_timeout,
                args.evaluate,
                args.eval_timeout,
                args.close_timeout,
            )
            for task in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            emit(result)

    failures = sorted(
        (result for result in results if not result["ok"]),
        key=lambda result: int(result["task_name"]),
    )
    emit(
        {
            "summary": True,
            "ok": not failures,
            "task_count": len(results),
            "success_count": len(results) - len(failures),
            "failure_count": len(failures),
            "failures": failures,
        }
    )
    if output_handle is not None:
        output_handle.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
