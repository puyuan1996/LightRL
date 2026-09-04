"""Format-aware answer extraction for math RLVR.

Extraction is intentionally separate from semantic verification.  A result
keeps every candidate and its provenance so a format gain can be distinguished
from a capability gain after a run has finished.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerCandidate:
    value: str
    format: str
    start: int
    end: int
    closed: bool = True


@dataclass(frozen=True)
class ExtractionResult:
    canonical: AnswerCandidate | None
    candidates: tuple[AnswerCandidate, ...]
    conflict: bool

    @property
    def format(self) -> str | None:
        return self.canonical.format if self.canonical else None

    @property
    def value(self) -> str | None:
        return self.canonical.value if self.canonical else None


class AnswerExtractor:
    """Small state-free adapter useful when injecting an extractor in config."""

    def extract(self, text: str) -> ExtractionResult:
        return extract_answers(text)


# Do not use ``\bAnswer`` alone: a model may write a Markdown heading such as
# ``**Answer:** 42``.  The line anchor avoids grabbing prose mentioning answer.
_ANSWER_LINE = re.compile(
    r"(?im)^[ \t]*(?:[*_`#>\-]+[ \t]*)?answer[ \t]*:[ \t]*(.+?)[ \t]*$"
)
_NATURAL = re.compile(
    r"(?is)(?:final[ \t]+answer|the[ \t]+answer|therefore[ ,:;\-]*answer|答案)"
    r"[ \t]*(?:is|=|:|为|是)?[ \t]*([^\n.!?。！？;；]+)"
)


def _boxed_candidates(text: str) -> list[AnswerCandidate]:
    found: list[AnswerCandidate] = []
    # A scanner handles nested braces, unlike a non-greedy regex.  It also
    # records an unclosed marker, which is useful evidence for truncation or
    # malformed output.
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        begin = match.start()
        body_start = match.end()
        depth = 1
        i = body_start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        closed = depth == 0
        end = i if closed else len(text)
        value_end = i - 1 if closed else len(text)
        found.append(
            AnswerCandidate(
                text[body_start:value_end].strip(),
                "boxed",
                begin,
                end,
                closed,
            )
        )
    return found


def _strip_markup(value: str) -> str:
    value = value.strip()
    # Markdown around an answer should not turn a correct answer into ``**``.
    value = re.sub(r"^[*_`]+|[*_`]+$", "", value).strip()
    return value


def extract_answers(text: str) -> ExtractionResult:
    """Extract all supported answer forms and choose a canonical candidate.

    Priority is last complete ``Answer:`` line, then last complete boxed value,
    then last natural-language final-answer phrase.  A malformed marker is
    retained as evidence but cannot become canonical.
    """

    text = str(text or "")
    candidates: list[AnswerCandidate] = []
    for match in _ANSWER_LINE.finditer(text):
        candidates.append(
            AnswerCandidate(_strip_markup(match.group(1)), "answer_line", match.start(), match.end())
        )
    candidates.extend(_boxed_candidates(text))
    for match in _NATURAL.finditer(text):
        value = _strip_markup(match.group(1))
        if value:
            candidates.append(AnswerCandidate(value, "natural_language", match.start(), match.end()))

    complete = [c for c in candidates if c.closed and c.value]
    rank = {"answer_line": 3, "boxed": 2, "natural_language": 1}
    canonical = max(complete, key=lambda c: (rank[c.format], c.start), default=None)

    # Compare format candidates after lightweight whitespace/markup cleanup;
    # semantic equivalence is intentionally left to verifier.py.
    distinct = {re.sub(r"\s+", "", c.value).lower() for c in complete}
    conflict = len(distinct) > 1
    return ExtractionResult(canonical, tuple(candidates), conflict)


def extract_answer(text: str) -> str | None:
    """Return only the canonical value (legacy convenience API)."""

    return extract_answers(text).value


__all__ = ["AnswerCandidate", "AnswerExtractor", "ExtractionResult", "extract_answer", "extract_answers"]
