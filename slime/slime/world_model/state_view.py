from __future__ import annotations

import json
from typing import Any, Sequence

from .metadata import redact_sensitive_jsonable, redact_sensitive_text, stable_hash, truncate_head_tail_text


FULL_CONTEXT_V1 = "full_context_v1"
BELIEF_VIEW_V1 = "belief_view_v1"
STATE_VIEW_CHOICES = (FULL_CONTEXT_V1, BELIEF_VIEW_V1)
BELIEF_VIEW_POOLING = "suffix_last_v1"
BELIEF_VIEW_ALLOWLIST = ("role", "name", "content")
BELIEF_VIEW_MAX_CHARS = 512


def _content_text(value: Any, max_chars: int) -> str:
    value = redact_sensitive_jsonable(value)
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    return truncate_head_tail_text(redact_sensitive_text(text), max_chars)


def _task_anchor(messages: Sequence[dict[str, Any]], max_chars: int) -> str:
    for message in messages:
        if str(message.get("role") or "").casefold() == "user":
            return _content_text(message.get("content", ""), max_chars)
    return ""


def _visible_events(
    messages: Sequence[dict[str, Any]],
    *,
    max_events: int,
    max_chars: int,
) -> list[dict[str, str]]:
    first_user_seen = False
    events: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "unknown").casefold()
        if role == "user" and not first_user_seen:
            first_user_seen = True
            continue
        if role not in {"tool", "environment", "user"}:
            continue
        content = _content_text(message.get("content", ""), max_chars)
        if not content:
            continue
        event = {"role": role, "content": content}
        name = message.get("name")
        if name is not None:
            event["name"] = truncate_head_tail_text(redact_sensitive_text(name), 256)
        events.append(event)
    return events[-max_events:]


def belief_view_parts(
    messages: Sequence[dict[str, Any]],
    *,
    max_events: int = 3,
    max_chars: int = BELIEF_VIEW_MAX_CHARS,
) -> tuple[str, str]:
    """Return causal conditioning and the versioned dynamic suffix.

    Only environment-visible fields enter the dynamic block. Assistant actions,
    reward/eval fields, rollout identifiers and arbitrary message metadata are
    excluded by construction.
    """

    if max_events <= 0 or max_chars <= 0:
        raise ValueError("belief view limits must be positive")
    safe_messages = [
        redact_sensitive_jsonable(message)
        for message in messages
        if isinstance(message, dict)
    ]
    anchor = _task_anchor(safe_messages, max_chars)
    events = _visible_events(
        safe_messages,
        max_events=max_events,
        max_chars=max_chars,
    )
    conditioning = (
        "<TASK_ANCHOR>\n"
        f"{anchor}\n"
        "</TASK_ANCHOR>\n"
    )
    payload = {
        "events": events,
        "observation_count": len(events),
        "state": "observed" if events else "initial",
        "version": BELIEF_VIEW_V1,
    }
    suffix = (
        f'<STATE_VIEW version="{BELIEF_VIEW_V1}">\n'
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n</STATE_VIEW>"
    )
    return conditioning, suffix


def belief_view_text(
    messages: Sequence[dict[str, Any]],
    *,
    max_events: int = 3,
    max_chars: int = BELIEF_VIEW_MAX_CHARS,
) -> str:
    conditioning, suffix = belief_view_parts(
        messages,
        max_events=max_events,
        max_chars=max_chars,
    )
    return conditioning + suffix


def belief_view_metadata(
    current_messages: Sequence[dict[str, Any]],
    next_messages: Sequence[dict[str, Any]] | None,
    *,
    max_events: int = 3,
    max_chars: int = BELIEF_VIEW_MAX_CHARS,
) -> dict[str, Any]:
    current = belief_view_text(
        current_messages,
        max_events=max_events,
        max_chars=max_chars,
    )
    next_text = (
        belief_view_text(next_messages, max_events=max_events, max_chars=max_chars)
        if next_messages
        else None
    )
    return {
        "state_view": BELIEF_VIEW_V1,
        "state_view_pooling": BELIEF_VIEW_POOLING,
        "state_view_allowlist": list(BELIEF_VIEW_ALLOWLIST),
        "state_view_max_events": max_events,
        "state_view_max_chars": max_chars,
        "state_view_hash": stable_hash(current),
        "next_state_view_hash": stable_hash(next_text) if next_text is not None else None,
    }
