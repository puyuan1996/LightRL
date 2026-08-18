"""Canonical action views for observational tool-call bundles."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .metadata import redact_sensitive_jsonable


ACTION_VIEW_SCHEMA = "tool_call_bundle_v1"
_CALL_BLOCK = re.compile(
    r'<call index=(\d+) name=("(?:\\.|[^"\\])*")>\n(.*?)\n</call>',
    re.DOTALL,
)


def render_tool_call_bundle(calls: Sequence[Mapping[str, Any]]) -> str:
    """Render ordered tool names and canonical arguments without assistant prose."""

    blocks: list[str] = []
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise TypeError("tool-call bundle entries must be mappings")
        name = str(call.get("tool_name") or call.get("name") or "").strip()
        if not name:
            raise ValueError("tool-call bundle entry is missing a tool name")
        arguments = redact_sensitive_jsonable(
            call.get("args", call.get("arguments", {}))
        )
        rendered_name = json.dumps(name, ensure_ascii=False)
        rendered_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        blocks.append(
            f"<call index={index} name={rendered_name}>\n"
            f"{rendered_arguments}\n"
            "</call>"
        )
    if not blocks:
        raise ValueError("tool-call bundle requires at least one call")
    return "\n\n".join(blocks)


def parse_tool_call_bundle(text: str) -> tuple[dict[str, Any], ...]:
    """Parse and canonicalize a strict ``tool_call_bundle_v1`` string."""

    if not isinstance(text, str) or not text:
        raise ValueError("tool-call bundle must be a non-empty string")
    calls: list[dict[str, Any]] = []
    position = 0
    for match in _CALL_BLOCK.finditer(text):
        if text[position : match.start()].strip():
            raise ValueError("tool-call bundle contains text outside call blocks")
        index = int(match.group(1))
        if index != len(calls):
            raise ValueError("tool-call indices must be contiguous and ordered")
        try:
            name = json.loads(match.group(2))
            arguments = json.loads(match.group(3))
        except json.JSONDecodeError as exc:
            raise ValueError("tool-call bundle contains invalid JSON") from exc
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool-call name must be a non-empty string")
        calls.append({"tool_name": name, "args": arguments})
        position = match.end()
    if text[position:].strip():
        raise ValueError("tool-call bundle contains text outside call blocks")
    if not calls:
        raise ValueError("tool-call bundle contains no call blocks")
    if render_tool_call_bundle(calls) != text:
        raise ValueError("tool-call bundle is not in canonical form")
    return tuple(calls)
