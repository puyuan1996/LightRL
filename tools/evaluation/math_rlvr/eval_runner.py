"""Batch evaluator for an OpenAI-compatible SGLang chat endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .data import load_dataset, resolve_dataset
from .scorer import ScoreConfig, score_group, score_sample, summarize
from .verifier import verifier_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--dataset", dest="data", required=True, help="dataset alias or JSON/JSONL path")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model", required=True, help="served model name; deliberately has no default")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--base-url", default=None, help="override http://127.0.0.1:<port>/v1/chat/completions")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default=None)
    parser.add_argument("--tag", default="T1.0")
    parser.add_argument("--n", type=int, default=16, help="samples per problem")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", "--max-response-len", dest="max_tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reward-type", "--rm-type", dest="reward_type", choices=("math", "dapo", "boxed"), default="math")
    parser.add_argument("--no-thinking", action="store_true")
    return parser


async def _request(session: Any, url: str, model: str, messages: list[dict[str, str]], args: argparse.Namespace, sem: asyncio.Semaphore) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if not args.no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    async with sem:
        for attempt in range(args.retries + 1):
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()
                choice = body.get("choices", [{}])[0]
                message = choice.get("message") or {}
                return {
                    "response": message.get("content") or "",
                    "reasoning_content": message.get("reasoning_content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": int((body.get("usage") or {}).get("completion_tokens") or 0),
                    "prompt_tokens": int((body.get("usage") or {}).get("prompt_tokens") or 0),
                }
            except Exception as exc:  # transient endpoint failures are retried
                if attempt >= args.retries:
                    return {
                        "response": "",
                        "reasoning_content": "",
                        "finish_reason": f"error:{type(exc).__name__}",
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    }
                await asyncio.sleep(min(30.0, 2.0 ** attempt))
    raise AssertionError("unreachable")


async def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.n <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        raise ValueError("n, concurrency and max-tokens must be positive")
    rows = load_dataset(args.data, data_root=args.data_root)
    if not rows:
        raise ValueError(f"dataset is empty: {args.data}")
    config = ScoreConfig(args.reward_type, args.max_tokens)
    url = args.base_url or f"http://127.0.0.1:{args.port}/v1/chat/completions"
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("eval_math requires aiohttp; install `pip install -e '.[rollout]'`") from exc

    started = time.time()
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        tasks = []
        for row in rows:
            messages = row.prompt if isinstance(row.prompt, list) else [{"role": "user", "content": str(row.prompt)}]
            for _ in range(args.n):
                tasks.append(_request(session, url, args.model, messages, args, semaphore))
        responses = await asyncio.gather(*tasks)

    problems = []
    for index, row in enumerate(rows):
        samples = []
        for sample_index, response_index in enumerate(range(index * args.n, (index + 1) * args.n)):
            raw = responses[response_index]
            scored = score_sample(
                raw["response"], row.label,
                completion_tokens=raw["completion_tokens"],
                finish_reason=raw["finish_reason"],
                config=config,
            )
            samples.append({"sample_index": sample_index, **raw, **scored})
        group = score_group(samples)
        problems.append({"id": row.id, "prompt": row.prompt, "label": row.label, "source": row.source, "samples": samples, **group, "zero_variance_group": group["zero_variance"]})

    detail = {
        "schema_version": 2,
        "dataset": args.data,
        "data_path": str(resolve_dataset(args.data, args.data_root)) if not args.data.startswith("hf://") else args.data,
        "model": args.model,
        "tag": args.tag,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "response_cap": args.max_tokens,
        "reward_type": args.reward_type,
        "verifier_sha256": verifier_digest(),
        "problems": problems,
    }
    detail["summary"] = summarize(problems, config=config, elapsed_s=time.time() - started)
    return detail, detail["summary"]


def write_outputs(detail: dict[str, Any], summary: dict[str, Any], *, output_dir: str | Path, dataset: str, tag: str, n: int) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(dataset).stem.replace("_", "-")
    detail_path = target_dir / f"{stem}_{tag}_n{n}.detail.json"
    summary_path = target_dir / f"{stem}_{tag}_n{n}.summary.json"
    detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return detail_path, summary_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    detail, summary = asyncio.run(evaluate(args))
    output_dir = args.output_dir or str(Path(args.data_root or "benchmarks/math") / "eval_results")
    detail_path, summary_path = write_outputs(detail, summary, output_dir=output_dir, dataset=args.data, tag=args.tag, n=args.n)
    print(json.dumps({"detail": str(detail_path), "summary": str(summary_path), **summary}, ensure_ascii=False))
    return 0


__all__ = ["build_parser", "evaluate", "main", "write_outputs"]
