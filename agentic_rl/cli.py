from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agentic_rl.config.loader import compose_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "compose"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", default="configs/config.yaml")
        command_parser.add_argument("overrides", nargs="*")
        if command == "train":
            command_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Resolve and print the launch without executing it.",
            )
    return parser


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return ",".join(_env_value(item) for item in value)
    return str(value)


def _runtime_env(config: dict[str, Any]) -> dict[str, str]:
    runtime = config.get("runtime", {})
    configured = runtime.get("env", {})
    if not isinstance(configured, dict):
        raise ValueError("runtime.env must be a mapping")

    env = {str(key): _env_value(value) for key, value in configured.items()}
    cluster = config.get("cluster", {})
    harness = config.get("harness", {})
    model = config.get("model", {})
    algorithm = config.get("algorithm", {})
    environment = config.get("environment", {})

    if cluster.get("num_gpus") is not None:
        env.setdefault("NUM_GPUS", _env_value(cluster["num_gpus"]))
    if environment.get("max_turn") is not None:
        env.setdefault("MAX_TURN", _env_value(environment["max_turn"]))
    if environment.get("name"):
        env.setdefault("DATASET", _env_value(environment["name"]))
    if model.get("checkpoint"):
        env.setdefault("HF_CKPT", _env_value(model["checkpoint"]))

    harness_names = {
        "camel_agent": "camel-agent",
        "claude_code_cli": "claude-code",
    }
    if harness.get("name"):
        env.setdefault(
            "HARNESS_OPTION",
            harness_names.get(harness["name"], _env_value(harness["name"])),
        )

    base_algorithm = algorithm.get("base", {})
    if isinstance(base_algorithm, dict) and base_algorithm.get("name"):
        env.setdefault("ALGO", _env_value(base_algorithm["name"]))
    elif algorithm.get("name"):
        env.setdefault("ALGO", _env_value(algorithm["name"]))
    return env


def _launch(config: dict[str, Any], dry_run: bool) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    launcher_value = config.get("runtime", {}).get("launcher")
    if not launcher_value:
        raise ValueError("runtime.launcher is required for training")

    launcher = Path(launcher_value)
    if not launcher.is_absolute():
        launcher = repo_root / launcher
    launcher = launcher.resolve()
    if not launcher.is_file():
        raise FileNotFoundError(f"Training launcher not found: {launcher}")

    env = os.environ.copy()
    env.update(_runtime_env(config))
    command = ["bash", str(launcher)]
    if dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "environment": _runtime_env(config),
                    "config": config,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    os.execvpe(command[0], command, env)
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    config = compose_config(Path(args.config), args.overrides)
    if args.command == "compose":
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    return _launch(config, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
