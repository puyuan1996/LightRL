from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

from .metadata import stable_hash
from .seta_dataset import load_terminal_transitions
from .state_view import BELIEF_VIEW_V1, belief_view_parts


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "p50": 0, "p95": 0, "max": 0}
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def audit(
    input_path: str | Path,
    *,
    max_events: int,
    tokenizer_name_or_path: str | None = None,
    max_context_tokens: int = 1536,
) -> dict[str, Any]:
    transitions = load_terminal_transitions(input_path, require_tool_feedback=True)
    current_hashes: list[str] = []
    next_hashes: list[str] = []
    current_lengths: list[int] = []
    next_lengths: list[int] = []
    unchanged = 0
    identifier_leaks: list[str] = []
    suffixes: list[str] = []
    forbidden_literals = ("reward_score", "eval_status", "rollout_id", "sample_index")
    forbidden_hits: Counter[str] = Counter()

    for row in transitions:
        current_parts = belief_view_parts(row.context_messages, max_events=max_events)
        current = "".join(current_parts)
        suffixes.append(current_parts[1])
        current_hash = stable_hash(current)
        current_hashes.append(current_hash)
        current_lengths.append(len(current))
        for literal in forbidden_literals:
            forbidden_hits[literal] += int(literal in current)
        for identifier in (
            row.trajectory_id,
            row.task_id,
        ):
            if identifier and len(identifier) >= 12 and identifier in current:
                identifier_leaks.append(stable_hash(identifier))

        if not row.has_next or row.next_context_messages is None:
            continue
        future_parts = belief_view_parts(
            row.next_context_messages,
            max_events=max_events,
        )
        future = "".join(future_parts)
        suffixes.append(future_parts[1])
        future_hash = stable_hash(future)
        next_hashes.append(future_hash)
        next_lengths.append(len(future))
        unchanged += int(current_hash == future_hash)
        for literal in forbidden_literals:
            forbidden_hits[literal] += int(literal in future)

    def duplicate_rate(values: Sequence[str]) -> float:
        return 1.0 - (len(set(values)) / max(len(values), 1))

    suffix_token_length = None
    oversized_suffix_count = 0
    if tokenizer_name_or_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        token_lengths: list[int] = []
        for start in range(0, len(suffixes), 256):
            tokenized = tokenizer(
                suffixes[start : start + 256],
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            token_lengths.extend(len(ids) for ids in tokenized)
        suffix_token_length = _summary(token_lengths)
        oversized_suffix_count = sum(length > max_context_tokens for length in token_lengths)

    gate = {
        "passed": (
            not identifier_leaks
            and not any(forbidden_hits.values())
            and not oversized_suffix_count
            and bool(next_hashes)
        ),
        "identifier_leak_count": len(identifier_leaks),
        "forbidden_field_hits": dict(forbidden_hits),
        "has_next_count": len(next_hashes),
        "oversized_suffix_count": oversized_suffix_count,
    }
    return {
        "schema_version": "openclaw_belief_view_audit_v1",
        "state_view": BELIEF_VIEW_V1,
        "max_events": max_events,
        "record_count": len(transitions),
        "current_char_length": _summary(current_lengths),
        "next_char_length": _summary(next_lengths),
        "suffix_token_length": suffix_token_length,
        "max_context_tokens": max_context_tokens,
        "current_duplicate_rate": duplicate_rate(current_hashes),
        "next_duplicate_rate": duplicate_rate(next_hashes),
        "unchanged_current_next_rate": unchanged / max(len(next_hashes), 1),
        "gate": gate,
        "claim_boundary": (
            "This structural audit checks deterministic construction and obvious metadata leakage. "
            "It does not replace tokenizer-length, representation-leakage or downstream heldout tests."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    args = parser.parse_args(argv)
    if args.max_events <= 0:
        raise ValueError("--max-events must be positive")
    if args.max_context_tokens <= 0:
        raise ValueError("--max-context-tokens must be positive")
    result = audit(
        args.input,
        max_events=args.max_events,
        tokenizer_name_or_path=args.tokenizer,
        max_context_tokens=args.max_context_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    if not result["gate"]["passed"]:
        raise ValueError(f"belief view audit failed: {result['gate']}")
    print(f"wrote belief-view audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
