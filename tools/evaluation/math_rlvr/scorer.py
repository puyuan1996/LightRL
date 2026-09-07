"""Scoring and observable diagnostics for sampled math responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .verifier import Verifier


@dataclass(frozen=True)
class ScoreConfig:
    reward_type: str = "math"
    response_cap: int = 32768

    def __post_init__(self) -> None:
        if self.response_cap <= 0:
            raise ValueError("response_cap must be positive")
        Verifier(self.reward_type)  # validate early


def _truncated(completion_tokens: int | None, finish_reason: str | None, cap: int) -> bool:
    return str(finish_reason or "").lower() in {"length", "max_tokens", "max_length"} or (
        completion_tokens is not None and int(completion_tokens or 0) >= cap
    )


def score_sample(
    response: str,
    label: object,
    *,
    completion_tokens: int | None = None,
    finish_reason: str | None = None,
    config: ScoreConfig | None = None,
) -> dict[str, Any]:
    """Return one JSON-serializable record with all three score tracks."""

    config = config or ScoreConfig()
    configured = Verifier(config.reward_type).verify(response, label)
    lenient = Verifier("math").verify(response, label)
    boxed = Verifier("boxed").verify(response, label)
    strict = Verifier("dapo").verify(response, label)
    truncated = _truncated(completion_tokens, finish_reason, config.response_cap)
    return {
        "reward": 1.0 if configured.correct else 0.0,
        "configured_correct": configured.correct,
        "reward_type": config.reward_type,
        "strict_correct": strict.correct,
        "strict_scorable": strict.scorable,
        "format_compliant": bool(strict.scorable and strict.format == "answer_line"),
        "lenient_correct": lenient.correct,
        "boxed_correct": boxed.correct,
        "strict_acc": strict.correct,
        "lenient_acc": lenient.correct,
        "boxed_acc": boxed.correct,
        "format_penalty": bool(lenient.correct and not strict.correct and strict.scorable),
        "format": lenient.format,
        "extracted": lenient.extracted,
        "conflict": lenient.conflict,
        "scorable": configured.scorable,
        "verifier_error": configured.error,
        "strict_verifier_error": strict.error,
        "completion_tokens": int(completion_tokens or 0),
        "finish_reason": finish_reason,
        "truncated": truncated,
        "response_tail": str(response or "")[-1000:],
    }


def score_group(records: list[dict[str, Any]], *, reward_key: str = "configured_correct") -> dict[str, Any]:
    """Annotate a rollout group and report zero-variance collapse."""

    values = [float(bool(record.get(reward_key))) for record in records]
    unique = len(set(values)) if values else 0
    variance = (sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)) if values else 0.0
    collapsed = len(values) > 0 and unique <= 1
    for record in records:
        record["group_reward_variance"] = variance
        record["zero_variance_group"] = collapsed
    return {"size": len(records), "variance": variance, "zero_variance": collapsed}


def avg_pass_at_k(
    per_problem: Iterable[dict[str, Any]],
    *,
    key: str,
    keep: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[float, float]:
    problems = list(per_problem)
    keep = keep or (lambda _: True)
    groups = [sample for problem in problems for sample in problem.get("samples", [])]
    if not problems or not groups:
        return 0.0, 0.0
    hits = [bool(sample.get(key)) and keep(sample) for sample in groups]
    solved = [any(bool(sample.get(key)) and keep(sample) for sample in problem.get("samples", [])) for problem in problems]
    return sum(hits) / len(hits), sum(solved) / len(solved)


def summarize(per_problem: list[dict[str, Any]], *, config: ScoreConfig, elapsed_s: float | None = None) -> dict[str, Any]:
    all_samples = [sample for problem in per_problem for sample in problem.get("samples", [])]
    avg, passed = avg_pass_at_k(per_problem, key="configured_correct")
    strict_avg, strict_pass = avg_pass_at_k(per_problem, key="strict_correct")
    lenient_avg, lenient_pass = avg_pass_at_k(per_problem, key="lenient_correct")
    boxed_avg, boxed_pass = avg_pass_at_k(per_problem, key="boxed_correct")
    scoreable = [sample for sample in all_samples if sample.get("scorable")]
    penalty = [sample for sample in scoreable if sample.get("format_penalty") and not sample.get("truncated")]
    zero_groups = [problem for problem in per_problem if problem.get("zero_variance_group")]
    format_rate = len(penalty) / len(scoreable) if scoreable else 0.0
    trunc_rate = sum(bool(sample.get("truncated")) for sample in all_samples) / len(all_samples) if all_samples else 0.0
    zero_rate = len(zero_groups) / len(per_problem) if per_problem else 0.0
    format_scoreable = [sample for sample in all_samples if sample.get("strict_scorable", sample.get("scorable"))]
    format_compliant = sum(bool(sample.get("format_compliant")) for sample in format_scoreable)
    return {
        "schema_version": 2,
        "reward_type": config.reward_type,
        "response_cap": config.response_cap,
        "num_problems": len(per_problem),
        "num_samples": len(all_samples),
        "avg_at_k": avg,
        "pass_at_k": passed,
        # Short aliases retained for existing result-processing notebooks.
        "avg": avg,
        "pass": passed,
        "strict_avg_at_k": strict_avg,
        "strict_pass_at_k": strict_pass,
        "lenient_avg_at_k": lenient_avg,
        "lenient_pass_at_k": lenient_pass,
        "boxed_avg_at_k": boxed_avg,
        "boxed_pass_at_k": boxed_pass,
        "format_penalty_count": len(penalty),
        "format_penalty_rate": format_rate,
        "format_mismatch_rate": format_rate,
        "format_mismatch_count": sum(bool(sample.get("format_penalty")) for sample in scoreable),
        "format_compliance_count": format_compliant,
        "format_compliance_rate": format_compliant / len(format_scoreable) if format_scoreable else 0.0,
        "truncated_count": sum(bool(sample.get("truncated")) for sample in all_samples),
        "truncation_rate": trunc_rate,
        "truncated_rate": trunc_rate,
        "zero_variance_group_count": len(zero_groups),
        "zero_variance_group_rate": zero_rate,
        "zero_variance_collapse_rate": zero_rate,
        "scorable_count": len(scoreable),
        "verifier_error_count": sum(not sample.get("scorable") for sample in all_samples),
        "elapsed_s": elapsed_s,
    }


def compliance_rate(records: list[dict[str, Any]]) -> float:
    """Return format compliance over scoreable records."""

    scoreable = [record for record in records if record.get(
        "strict_scorable", record.get("scorable", record.get("strict_pred") != "[STRICT_ERROR]"))]
    if not scoreable:
        return float("nan")
    def compliant(record: dict[str, Any]) -> bool:
        if "format_compliant" in record:
            return bool(record["format_compliant"])
        if "format" in record:
            return record["format"] == "answer_line"
        return record.get("strict_pred", record.get("pred")) not in (None, "[INVALID]", "[STRICT_ERROR]")
    return sum(compliant(record) for record in scoreable) / len(scoreable)


def rescore_detail(
    payload: dict[str, Any], *, reward_type: str | None = None, response_cap: int | None = None
) -> dict[str, Any]:
    """Recompute all score tracks from stored responses without model calls."""

    config = ScoreConfig(
        reward_type=reward_type or payload.get("reward_type", "math"),
        response_cap=int(response_cap or payload.get("response_cap", 32768)),
    )
    problems = []
    for problem in payload.get("problems", payload.get("per_problem", [])):
        samples = []
        for sample in problem.get("samples", []):
            scored = dict(sample)
            scored.update(score_sample(
                sample.get("response", sample.get("text", "")), problem.get("label", ""),
                completion_tokens=sample.get("completion_tokens"),
                finish_reason=sample.get("finish_reason"), config=config,
            ))
            samples.append(scored)
        group = score_group(samples)
        problems.append({**problem, "samples": samples, "zero_variance_group": group["zero_variance"]})
    return {"summary": summarize(problems, config=config), "problems": problems}


def rescore_file(
    path: str | Path, *, output: str | Path | None = None,
    reward_type: str | None = None, response_cap: int | None = None,
) -> Path:
    source = Path(path)
    result = rescore_detail(json.loads(source.read_text(encoding="utf-8")), reward_type=reward_type, response_cap=response_cap)
    target = Path(output) if output else source.with_name(source.stem + ".rescore.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def rescore_directory(
    results_dir: str | Path, *, reward_type: str | None = None, response_cap: int | None = None
) -> list[Path]:
    return [rescore_file(source, reward_type=reward_type, response_cap=response_cap)
            for source in sorted(Path(results_dir).glob("*.detail.json"))]


__all__ = [
    "ScoreConfig", "avg_pass_at_k", "compliance_rate", "rescore_detail", "rescore_directory",
    "rescore_file", "score_group", "score_sample", "summarize",
]
