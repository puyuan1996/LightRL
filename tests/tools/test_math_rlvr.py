from __future__ import annotations

import math

from tools.evaluation.math_rlvr.data import MathExample, deduplicate_rows
from tools.evaluation.math_rlvr.extractor import extract_answers
from tools.evaluation.math_rlvr.scorer import ScoreConfig, score_group, score_sample, summarize
from tools.evaluation.math_rlvr.stats import compare_payloads
from tools.evaluation.math_rlvr.scorer import compliance_rate
from tools.evaluation.math_rlvr.verifier import Verifier


def test_extractor_priority_and_conflict():
    result = extract_answers("\\boxed{41}\n**Answer:** 42")
    assert result.value == "42"
    assert result.format == "answer_line"
    assert result.conflict


def test_extractor_nested_box_and_natural_language():
    nested = extract_answers(r"final answer is \boxed{\frac{1}{2}}")
    assert nested.value == r"\frac{1}{2}"
    assert not nested.conflict
    assert extract_answers("因此答案为 17。").value == "17"


def test_natural_language_equation_uses_terminal_scalar():
    # The reasoning trace may end without ``Answer:`` or ``\\boxed{}``.
    # The terminal value of a conclusion equation is still the submitted
    # answer and must be scored identically by train and eval.
    assert Verifier("math").verify("Therefore m+n=106", "106").correct
    assert Verifier("math").verify("Thus, the answer is 3/4.", "3/4").correct
    assert not extract_answers(
        "Therefore, the conclusion is that the greedy algorithm is optimal "
        "when the remainder has last digit >=5, so the number is 9"
    ).candidates


def test_verifier_tracks_format_without_format_learning():
    assert Verifier("math").verify(r"\boxed{42}", "42").correct
    strict = score_sample(r"\boxed{42}", "42", config=ScoreConfig("math", 100))
    assert strict["lenient_correct"] and not strict["strict_correct"]
    assert strict["format_penalty"]


def test_cap_and_zero_variance_are_observable():
    group = [score_sample("Answer: 1", "1", completion_tokens=8, config=ScoreConfig("math", 8)) for _ in range(2)]
    info = score_group(group)
    assert info["zero_variance"]
    summary = summarize([{"id": "p", "label": "1", "samples": group, "zero_variance_group": True}], config=ScoreConfig("math", 8))
    assert summary["truncated_count"] == 2
    assert summary["zero_variance_group_rate"] == 1.0


def test_deduplicate_questions_stably():
    rows = [MathExample("a", "same", "1", "x"), MathExample("b", "same", "1", "y"), MathExample("c", "other", "2", "x")]
    assert [row.id for row in deduplicate_rows(rows)] == ["a", "c"]


def test_paired_stats_requires_same_sample_population():
    base = {"problems": [{"id": "p", "samples": [{"lenient_correct": False}, {"lenient_correct": True}]}]}
    candidate = {"problems": [{"id": "p", "samples": [{"lenient_correct": True}, {"lenient_correct": True}]}]}
    result = compare_payloads(base, candidate)
    assert result["mean_delta"] == 0.5
    assert result["wins"] == 1


def test_format_compliance_excludes_unscorable_labels():
    assert math.isnan(compliance_rate([{"strict_scorable": False, "format_compliant": True}]))


def test_score_sample_tracks_match_direct_verifiers():
    # score_sample reuses one extraction for all tracks; each track must still
    # match an independent Verifier call on the raw response.
    response = "Some reasoning.\n\\boxed{41}\n**Answer:** 42"
    record = score_sample(response, "42", config=ScoreConfig("math", 100))
    assert record["configured_correct"] == Verifier("math").verify(response, "42").correct
    assert record["lenient_correct"] == Verifier("math").verify(response, "42").correct
    assert record["boxed_correct"] == Verifier("boxed").verify(response, "42").correct
    assert record["strict_correct"] == Verifier("dapo").verify(response, "42").correct
    assert record["lenient_correct"] and record["strict_correct"] and not record["boxed_correct"]
