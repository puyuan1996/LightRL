"""Post-hoc rescoring at a new cap or reward policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scorer import ScoreConfig, score_group, summarize


def rescore_detail(payload: dict[str, Any], *, reward_type: str | None = None, response_cap: int | None = None) -> dict[str, Any]:
    config = ScoreConfig(
        reward_type=reward_type or payload.get("reward_type", "math"),
        response_cap=int(response_cap or payload.get("response_cap", 32768)),
    )
    problems = []
    for problem in payload.get("problems", payload.get("per_problem", [])):
        samples = []
        for sample in problem.get("samples", []):
            scored = dict(sample)
            scored.update(score_sample_from_record(sample, problem.get("label", ""), config=config))
            samples.append(scored)
        group = score_group(samples)
        problems.append({**problem, "samples": samples, "zero_variance_group": group["zero_variance"]})
    summary = summarize(problems, config=config)
    return {"summary": summary, "problems": problems}


def score_sample_from_record(record: dict[str, Any], label: object, *, config: ScoreConfig) -> dict[str, Any]:
    # Import lazily to keep the module useful for tools that only inspect JSON.
    from .scorer import score_sample

    return score_sample(
        record.get("response", record.get("text", "")),
        label,
        completion_tokens=record.get("completion_tokens"),
        finish_reason=record.get("finish_reason"),
        config=config,
    )


def rescore_file(path: str | Path, *, output: str | Path | None = None, reward_type: str | None = None, response_cap: int | None = None) -> Path:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = rescore_detail(payload, reward_type=reward_type, response_cap=response_cap)
    target = Path(output) if output else source.with_name(source.stem + ".rescore.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def rescore_directory(results_dir: str | Path, *, reward_type: str | None = None, response_cap: int | None = None) -> list[Path]:
    """Rescore every detail artifact in a run directory."""

    directory = Path(results_dir)
    outputs = []
    for source in sorted(directory.glob("*.detail.json")):
        outputs.append(rescore_file(source, reward_type=reward_type, response_cap=response_cap))
    return outputs


__all__ = ["rescore_detail", "rescore_directory", "rescore_file"]
