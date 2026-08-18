"""Canonical target views for observational tool-result bundles."""

from __future__ import annotations

import re
from typing import Sequence


RESULT_VIEW_SCHEMA = "result_only_v1"
_OPEN = re.compile(
    r'<tool_result(?:\s+name=(?:"[^"]*"|\'[^\']*\'|[^>\s]+))?\s*>\n?'
)
_CLOSE = re.compile(r"\n?</tool_result>")
_RESULT_BLOCK = re.compile(
    r"<result index=(\d+)>\n(.*?)\n</result>",
    re.DOTALL,
)


def parse_tool_result_bundle(feedback_text: str) -> tuple[str, ...]:
    """Extract logged result bodies while rejecting unparsed wrapper content."""

    if not isinstance(feedback_text, str) or not feedback_text.strip():
        raise ValueError("tool-result feedback must be a non-empty string")
    results: list[str] = []
    position = 0
    while position < len(feedback_text):
        opening = _OPEN.search(feedback_text, position)
        if opening is None:
            if feedback_text[position:].strip():
                raise ValueError("feedback contains text outside tool_result wrappers")
            break
        if feedback_text[position : opening.start()].strip():
            raise ValueError("feedback contains text outside tool_result wrappers")
        closing = _CLOSE.search(feedback_text, opening.end())
        if closing is None:
            raise ValueError("tool_result wrapper is not closed")
        body = feedback_text[opening.end() : closing.start()]
        results.append(body)
        position = closing.end()
    if not results:
        raise ValueError("feedback contains no tool_result wrapper")
    return tuple(results)


def render_result_only_view(results: Sequence[str]) -> str:
    """Render an order-preserving target with no tool-name wrapper leakage."""

    values = [str(value) for value in results]
    if not values:
        raise ValueError("result-only view requires at least one result")
    return "\n\n".join(
        f"<result index={index}>\n{value}\n</result>"
        for index, value in enumerate(values)
    )


def parse_result_only_view(text: str) -> tuple[str, ...]:
    """Parse a strict, canonical ``result_only_v1`` target."""

    if not isinstance(text, str) or not text:
        raise ValueError("result-only view must be a non-empty string")
    values: list[str] = []
    position = 0
    for match in _RESULT_BLOCK.finditer(text):
        if text[position : match.start()].strip():
            raise ValueError("result-only view contains text outside result blocks")
        if int(match.group(1)) != len(values):
            raise ValueError("result indices must be contiguous and ordered")
        values.append(match.group(2))
        position = match.end()
    if text[position:].strip():
        raise ValueError("result-only view contains text outside result blocks")
    if not values:
        raise ValueError("result-only view contains no result blocks")
    if render_result_only_view(values) != text:
        raise ValueError("result-only view is not in canonical form")
    return tuple(values)


def result_only_view(feedback_text: str) -> str:
    return render_result_only_view(parse_tool_result_bundle(feedback_text))
