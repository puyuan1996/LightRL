"""Evaluation harness adapters (distinct from the training rollout harnesses).

Lazy registry mirroring :mod:`agentic_rl.harnesses.factory`; importing this
package never pulls heavy optional dependencies.
"""

from __future__ import annotations

from importlib import import_module

_EVAL_HARNESS_ALIASES = {
    "terminus2": "terminus2", "terminus-2": "terminus2", "terminus_2": "terminus2",
    "claude_code_cli": "claude_code_cli", "claude-code": "claude_code_cli",
    "claude-code-cli": "claude_code_cli", "claude_code": "claude_code_cli",
    "claude": "claude_code_cli",
    "camel_agent": "camel_agent", "camel": "camel_agent",
    "camel-agent": "camel_agent", "camelagent": "camel_agent",
}
_EVAL_HARNESS_TARGETS = {
    "terminus2": ("agentic_rl.harnesses.eval.terminus2", "Terminus2EvalHarness"),
    "claude_code_cli": ("agentic_rl.harnesses.eval.claude_code_cli", "ClaudeCodeCliEvalHarness"),
    "camel_agent": ("agentic_rl.harnesses.eval.camel_agent", "CamelAgentEvalHarness"),
}


def normalize_eval_harness_name(value: str | None) -> str:
    requested = value or "terminus2"
    try:
        return _EVAL_HARNESS_ALIASES[requested]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported eval harness option: {value!r}. "
            "Available: terminus-2, claude-code, camel-agent"
        ) from exc


def create_eval_harness(name: str | None):
    """Instantiate an eval harness adapter, importing its module lazily."""
    module_name, class_name = _EVAL_HARNESS_TARGETS[normalize_eval_harness_name(name)]
    return getattr(import_module(module_name), class_name)()


__all__ = ["create_eval_harness", "normalize_eval_harness_name"]
