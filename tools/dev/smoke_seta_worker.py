#!/usr/bin/env python3
"""Exercise reset, evaluation, and cleanup for one SETA worker task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from prebuild_seta_worker import (
    _evaluation_failure_reason,
    _load_tasks,
    _request_json,
    _wait_closed,
)


def _verify_runtime_proxy(
    worker_url: str, lease_id: str, expected_url: str
) -> dict[str, bool]:
    """Verify proxy injection without returning or logging credential values."""
    code, body = _request_json(worker_url, "/status", None, timeout=10)
    if code >= 400 or not body.get("ok"):
        raise RuntimeError(f"worker status unavailable during proxy check: HTTP {code}")
    tasks = body.get("pool", {}).get("tasks", {})
    matches = [
        run
        for task in tasks.values()
        for run_id, run in task.get("runs", {}).items()
        if run_id == lease_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one status entry for lease, found {len(matches)}")
    container_name = matches[0].get("container", {}).get("name")
    if not container_name:
        raise RuntimeError("worker status did not report a container name")
    payload = json.loads(
        subprocess.check_output(
            ["docker", "inspect", str(container_name)], text=True
        )
    )[0]
    container_env = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in payload.get("Config", {}).get("Env", [])
        if "=" in item
    }
    results = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        value = container_env.get(key, "")
        results[key] = value == expected_url and "@" not in value
    if not all(results.values()):
        failed = ",".join(key for key, ok in results.items() if not ok)
        raise RuntimeError(f"unsafe or missing runtime proxy variables: {failed}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--reset-timeout", type=float, default=1800.0)
    parser.add_argument("--eval-timeout", type=float, default=1200.0)
    parser.add_argument("--close-timeout", type=float, default=300.0)
    parser.add_argument(
        "--before-eval-command",
        default="",
        help="Optional disposable-container command to run before evaluation.",
    )
    parser.add_argument(
        "--verify-runtime-proxy-url",
        default="",
        help=(
            "When set, inspect the reset container locally and fail unless all "
            "upper/lower HTTP(S) proxy variables exactly match this URL and "
            "contain no credentials. Intended for execution on the worker host."
        ),
    )
    args = parser.parse_args()

    matches = [
        task
        for task in _load_tasks(args.dataset)
        if str(task["task_name"]) == args.task_name
    ]
    if len(matches) != 1:
        parser.error(f"expected one task named {args.task_name!r}, found {len(matches)}")
    metadata = matches[0]

    code, body = _request_json(args.worker_url, "/healthz", None, timeout=10)
    if code >= 400 or not body.get("ok"):
        print(json.dumps({"ok": False, "phase": "healthz", "http": code, "body": body}))
        return 2

    request_id = f"seta-smoke-{args.task_name}-{uuid.uuid4().hex[:10]}"
    lease_id = ""
    result_code = 0
    started = time.monotonic()
    try:
        code, body = _request_json(
            args.worker_url,
            "/allocate",
            {
                "task_key": f"seta-smoke:{args.task_name}",
                "request_id": request_id,
            },
            timeout=60,
        )
        if code >= 400 or not body.get("ok"):
            print(json.dumps({"ok": False, "phase": "allocate", "http": code, "body": body}))
            return 3
        lease_id = str(body["lease_id"])

        code, body = _request_json(
            args.worker_url,
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
                    "ensure_image": args.reset_timeout,
                    "reset_session": 900,
                    "eval": args.eval_timeout,
                    "close_session": 90,
                },
                "request_id": request_id,
            },
            timeout=args.reset_timeout + 1200,
        )
        if code >= 400 or not body.get("ok"):
            print(json.dumps({"ok": False, "phase": "reset", "http": code, "body": body}))
            return 4

        proxy_verified = False
        if args.verify_runtime_proxy_url:
            try:
                _verify_runtime_proxy(
                    args.worker_url, lease_id, args.verify_runtime_proxy_url
                )
                proxy_verified = True
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "phase": "runtime_proxy",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                )
                return 7

        if args.before_eval_command:
            code, body = _request_json(
                args.worker_url,
                "/exec_tool",
                {
                    "lease_id": lease_id,
                    "tool_call": {
                        "name": "shell_exec",
                        "arguments": {
                            "id": "seta-smoke-before-eval",
                            "command": args.before_eval_command,
                            "block": True,
                            "timeout": 30,
                        },
                    },
                },
                timeout=60,
            )
            if code >= 400 or not body.get("ok"):
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "phase": "before_eval_command",
                            "http": code,
                            "body": body,
                        }
                    )
                )
                return 8

        code, body = _request_json(
            args.worker_url,
            "/evaluate",
            {"lease_id": lease_id, "trajectory": {}},
            timeout=args.eval_timeout + 60,
        )
        score = body.get("score") if isinstance(body, dict) else None
        details = body.get("details") if isinstance(body, dict) else None
        if _evaluation_failure_reason(code, body):
            print(json.dumps({"ok": False, "phase": "evaluate", "http": code, "body": body}))
            return 5
        print(
            json.dumps(
                {
                    "ok": True,
                    "phase": "evaluate",
                    "task_name": args.task_name,
                    "score": float(score),
                    "details": details,
                    "runtime_proxy_verified": proxy_verified,
                    "elapsed_sec": round(time.monotonic() - started, 3),
                },
                ensure_ascii=False,
            )
        )
    finally:
        if lease_id:
            code, body = _request_json(
                args.worker_url, "/close", {"lease_id": lease_id}, timeout=180
            )
            if code >= 400 or not body.get("ok") or body.get("found") is not True:
                print(json.dumps({"ok": False, "phase": "close", "http": code, "body": body}))
                result_code = 6
            elif not _wait_closed(args.worker_url, lease_id, args.close_timeout):
                print(json.dumps({"ok": False, "phase": "close_wait", "lease_id": lease_id}))
                result_code = 6
    return result_code


if __name__ == "__main__":
    sys.exit(main())
