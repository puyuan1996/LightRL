"""Serving contract consumed by agent harnesses."""

from __future__ import annotations

from typing import Any, Protocol

from agentic_rl.types import Interaction


class TurnClient(Protocol):
    tokenizer: Any
    sampling_params: dict[str, Any]
    tool_call_parser: str | None
    session_id: str | None
    url: str
    request_timeout: float | None

    async def generate_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        turn_idx: int,
    ) -> tuple[Any, Interaction]: ...

    def _truncate_input_ids(self, input_ids: list[int]) -> list[int]: ...

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[int]: ...


__all__ = ["TurnClient"]
