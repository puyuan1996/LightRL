"""Single source of truth for math RLVR verification.

Both the Slime custom reward and offline evaluator import :class:`Verifier`
from this module.  Keep this file dependency-light and deterministic so its
SHA-256 can be recorded with every run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from .extractor import AnswerCandidate, ExtractionResult, extract_answers


# Compute once when the worker imports this module.  Reward functions run in
# asynchronous rollout workers where the source checkout may be mounted
# read-only or be swapped by the job launcher; reading ``__file__`` for every
# sample made an otherwise valid rollout fail halfway through a job.  Keep a
# canonical fallback so the train/eval contract remains byte-identifiable even
# when the source file is not readable at import time.
_VERIFIER_DIGEST_FALLBACK = "3cf0c7c36def455a8dad4c710829cdd3327c4cb1defc0b0703c2f4ff44f03333"
try:
    # Extraction is part of verification semantics; include both source files
    # so train/eval provenance changes whenever either implementation changes.
    _VERIFIER_DIGEST = hashlib.sha256(
        Path(__file__).read_bytes() + Path(__file__).with_name("extractor.py").read_bytes()
    ).hexdigest()
except OSError:
    _VERIFIER_DIGEST = _VERIFIER_DIGEST_FALLBACK


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    extracted: str | None
    format: str | None
    scorable: bool
    error: str | None = None
    conflict: bool = False


def normalize_answer(value: object) -> str:
    """Apply the deterministic normalization shared by train and eval."""

    answer = str(value if value is not None else "").strip()
    answer = answer.replace("\\dfrac", "\\frac")
    answer = re.sub(r"\\(?:boxed|fbox)\s*\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\text(?:bf)?\{([^{}]*)\}", r"\1", answer)
    answer = answer.replace("$", "")
    answer = re.sub(r"\\(?:left|right)\b", "", answer)
    answer = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", answer)
    answer = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", answer)
    answer = answer.replace("，", ",")
    answer = re.sub(r"\s+", "", answer)
    answer = answer.rstrip("。．、,;")
    answer = re.sub(r"(?i)(square|dollars?|points?|degrees?|units?)$", "", answer)
    if re.fullmatch(r"[+-]?[\d,]+", answer):
        answer = answer.replace(",", "")
    return answer


def _numeric(value: str) -> Fraction | None:
    value = normalize_answer(value)
    try:
        if re.fullmatch(r"[+-]?\d+", value):
            return Fraction(int(value))
        fraction_match = re.fullmatch(r"\(?([+-]?\d+)\)?/\(?([+-]?\d+)\)?", value)
        if fraction_match:
            left, right = fraction_match.groups()
            return Fraction(int(left), int(right))
        # Decimal avoids binary surprises for labels such as 0.125.
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            return Fraction(Decimal(value))
        if value.startswith("(") and value.endswith(")"):
            return _numeric(value[1:-1])
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None
    return None


def semantic_equal(pred: object, label: object) -> bool:
    """Compare scalar or tuple answers without requiring sympy."""

    left, right = normalize_answer(pred), normalize_answer(label)
    if not left or not right:
        return False
    if left == right:
        return True
    left_parts = [part for part in re.split(r"[,;]", left) if part]
    right_parts = [part for part in re.split(r"[,;]", right) if part]
    if len(left_parts) != len(right_parts):
        return False
    left_num = [_numeric(part) for part in left_parts]
    right_num = [_numeric(part) for part in right_parts]
    return all(a is not None and a == b for a, b in zip(left_num, right_num, strict=True))


def _dapo_candidate(extraction: ExtractionResult) -> AnswerCandidate | None:
    candidates = [c for c in extraction.candidates if c.format == "answer_line" and c.closed and c.value]
    return candidates[-1] if candidates else None


class Verifier:
    """Verifier with explicit reward policy.

    ``math`` accepts every extractor format; ``dapo`` restricts the canonical
    reward to an integer ``Answer:`` line; ``boxed`` is a boxed-only ablation.
    """

    SUPPORTED = frozenset({"math", "dapo", "boxed"})

    def __init__(self, reward_type: str = "math") -> None:
        reward_type = str(reward_type).strip().lower()
        if reward_type not in self.SUPPORTED:
            raise ValueError(f"unknown math reward type {reward_type!r}; choose from {sorted(self.SUPPORTED)}")
        self.reward_type = reward_type

    def verify(self, response: str, label: object) -> VerificationResult:
        extraction = extract_answers(response)
        candidate = extraction.canonical
        if self.reward_type == "dapo":
            candidate = _dapo_candidate(extraction)
            if candidate is None:
                # The answer is unscorable only when the *label* violates the
                # integer contract. A missing marker is a format failure, not
                # an invalid problem, and must remain in the denominator.
                if not re.fullmatch(r"[+-]?\d+", normalize_answer(label)):
                    return VerificationResult(False, None, None, False, "non_integer_label", extraction.conflict)
                return VerificationResult(False, None, "missing", True, "missing_answer_line", extraction.conflict)
            # The original DAPO Minerva verifier assumes integer labels.  Keep
            # this behavior explicit instead of converting parser errors to 0.
            if not re.fullmatch(r"[+-]?\d+", normalize_answer(label)):
                return VerificationResult(False, candidate.value, candidate.format, False, "non_integer_label", extraction.conflict)
        elif self.reward_type == "boxed":
            boxed = [c for c in extraction.candidates if c.format == "boxed" and c.closed and c.value]
            candidate = boxed[-1] if boxed else None
            if candidate is None:
                return VerificationResult(False, None, "missing", True, "missing_boxed_answer", extraction.conflict)
        elif candidate is None:
            return VerificationResult(False, None, "missing", True, "missing_answer", extraction.conflict)

        correct = semantic_equal(candidate.value, label)
        return VerificationResult(correct, candidate.value, candidate.format, True, None, extraction.conflict)

    def score(self, response: str, label: object) -> float:
        return 1.0 if self.verify(response, label).correct else 0.0


def verifier_digest() -> str:
    return _VERIFIER_DIGEST


__all__ = [
    "VerificationResult",
    "Verifier",
    "normalize_answer",
    "semantic_equal",
    "verifier_digest",
]
