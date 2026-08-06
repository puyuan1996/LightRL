"""Tau2 task instruction text extraction, shared by the in-process runtime
(environments/tau2/runtime.py) and the dataset converter
(data/convert_tau2_to_dataset.py)."""

from __future__ import annotations

from typing import Any


def _structured_instruction_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()

    lines: list[str] = []
    for label, attr in (
        ("Domain", "domain"),
        ("Reason", "reason_for_call"),
        ("Known info", "known_info"),
        ("Unknown info", "unknown_info"),
        ("Task instructions", "task_instructions"),
    ):
        raw = getattr(value, attr, None)
        if raw:
            lines.append(f"{label}: {raw}")
    return "\n".join(lines).strip()


def task_instruction(task: Any) -> str:
    ticket = getattr(task, "ticket", None)
    if ticket:
        return str(ticket).strip()

    user_scenario = getattr(task, "user_scenario", None)
    if user_scenario is not None:
        instructions = getattr(user_scenario, "instructions", None)
        structured = _structured_instruction_text(instructions)
        if structured:
            return structured
        if instructions is not None:
            return str(instructions).strip()

    description = getattr(task, "description", None)
    if description is not None:
        for attr in ("notes", "purpose"):
            raw = getattr(description, attr, None)
            if raw:
                return str(raw).strip()

    return str(getattr(task, "id", "unknown")).strip()
