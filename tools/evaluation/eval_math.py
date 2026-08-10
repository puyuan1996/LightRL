#!/usr/bin/env python3
"""Avg@k / Pass@k evaluation for math RLVR sets against an sglang OpenAI-compatible server.

Scoring uses the SAME verifier the training run will use (slime rm_hub
compute_score_dapo, i.e. rm_type="dapo"), so base numbers and in-training eval
numbers are directly comparable.

Records per-sample completion_tokens + finish_reason so we can answer
"what would accuracy be if max_response_len were 8192?" post-hoc on the very
same samples, without re-generating.

Usage:
  python eval_math.py --data aime-2025/aime-2025.jsonl --n 16 \
      --temperature 1.0 --top-p 1.0 --max-tokens 32768 --tag T1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

# tools/evaluation/eval_math.py -> repo root is two levels up. Resolving from
# __file__ keeps this runnable from any checkout and any cwd; the verifier has to
# be the vendored slime one, because the whole point is that base numbers and
# in-training eval numbers come from the same scoring code.
REPO = Path(__file__).resolve().parents[2]
if not (REPO / "slime" / "slime" / "rollout" / "rm_hub").is_dir():
    raise SystemExit(
        f"vendored slime not found under {REPO / 'slime'}; "
        "run this from a LightRL checkout, or set PYTHONPATH to one"
    )
sys.path.insert(0, str(REPO / "slime"))
from slime.rollout.rm_hub import compute_score_dapo  # noqa: E402
from slime.rollout.rm_hub import extract_boxed_answer, grade_answer_verl  # noqa: E402


def lenient_acc(text: str, gt: str) -> bool:
    """Accept a correct answer given EITHER as 'Answer: X' or as the last \\boxed{X}.

    The strict path (compute_score_dapo) is what the training reward sees. Qwen3
    thinking models very often answer with \\boxed{} instead, which the strict
    verifier scores [INVALID]. Reporting both separates "cannot do the math" from
    "did not follow the output format".
    """
    # NOTE: grade_answer_verl(solution_str, gt) does the \boxed{} extraction
    # itself. Passing an already-extracted answer makes it return False for
    # everything, since extract_answer("279") is None.
    if not extract_boxed_answer(text):
        return False
    try:
        return bool(grade_answer_verl(text, gt))
    except Exception:
        return False

# Where prepare_math_data.py wrote the JSONL sets. Override with MATH_DATA_ROOT.
DATA_ROOT = Path(os.environ.get("MATH_DATA_ROOT", REPO / "benchmarks" / "math"))


def avg_and_pass_at_k(per_problem, keep_fn, key="acc"):
    """Avg@k and Pass@k over already-scored samples.

    Avg@k averages over every problem x sample; Pass@k is the share of problems
    with at least one hit. ``keep_fn`` is what makes the token-cap counterfactual
    possible without regenerating: a sample longer than the cap could not have
    emitted its answer, so it is scored wrong rather than dropped -- dropping it
    would shrink the denominator and inflate the result.
    """
    all_samples = [s for p in per_problem for s in p["samples"]]
    if not all_samples:
        return 0.0, 0.0
    hits = [bool(s[key]) and keep_fn(s) for s in all_samples]
    avg = sum(hits) / len(hits)
    solved = sum(
        1 for p in per_problem if any(bool(s[key]) and keep_fn(s) for s in p["samples"])
    )
    return avg, solved / len(per_problem)


async def one_sample(session, url, model, messages, sp, sem, retries=3):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": sp["temperature"],
        "top_p": sp["top_p"],
        "max_tokens": sp["max_tokens"],
        # Qwen3 chat template switch: keep the thinking chain ON. Without this the
        # template can fall back to non-thinking, which is fatal on AIME.
        "chat_template_kwargs": {"enable_thinking": True},
    }
    async with sem:
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                    body = await resp.json()
                choice = body["choices"][0]
                text = choice["message"]["content"] or ""
                reasoning = choice["message"].get("reasoning_content") or ""
                return {
                    "text": text,
                    "reasoning_len_chars": len(reasoning),
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": body.get("usage", {}).get("completion_tokens", 0),
                    "prompt_tokens": body.get("usage", {}).get("prompt_tokens", 0),
                }
            except Exception as exc:  # transient server hiccup -> retry
                if attempt == retries - 1:
                    return {
                        "text": "",
                        "reasoning_len_chars": 0,
                        "finish_reason": f"error:{type(exc).__name__}",
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    }
                await asyncio.sleep(5 * (attempt + 1))


async def run(args):
    data_path = DATA_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    print(f"[data] {data_path.name}: {len(rows)} problems x n={args.n} = {len(rows) * args.n} samples")

    sp = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for row in rows:
            for _ in range(args.n):
                tasks.append(one_sample(session, url, args.model, row["prompt"], sp, sem))
        flat = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    # ── score ────────────────────────────────────────────────────────────
    per_problem = []
    for i, row in enumerate(rows):
        samples = flat[i * args.n : (i + 1) * args.n]
        gt = str(row["label"]).strip()
        scored = []
        for s in samples:
            # compute_score_dapo hard-assumes an integer ground truth:
            #   math_dapo_utils.py:210  gt = str(int(float(gt)))
            # so it raises ValueError on LaTeX answers (MATH-500 has coordinate
            # pairs, fractions, intervals). Treat that as "strict not computable"
            # rather than crashing the whole sweep, and count it.
            try:
                res = compute_score_dapo(s["text"], gt)
                strict_err = False
            except Exception:  # noqa: BLE001
                res = {"acc": False, "pred": "[STRICT_ERROR]"}
                strict_err = True
            strict = bool(res["acc"])
            scored.append(
                {
                    **{k: v for k, v in s.items() if k != "text"},
                    "acc": strict,
                    "strict_error": strict_err,
                    # lenient = strict OR a correct last \boxed{}
                    "acc_lenient": strict or lenient_acc(s["text"], gt),
                    "pred": res["pred"],
                    "response_chars": len(s["text"]),
                    # keep the tail for auditability (the answer region)
                    "text_tail": s["text"][-700:],
                    # The pure-boxed track is recomputed downstream from text_tail,
                    # so record the same judgement on the FULL text here. Without
                    # it the two cannot be compared, and a \boxed{} sitting further
                    # back than the tail would be silently scored as absent --
                    # unrecoverably, since the full text is not kept.
                    "boxed_full": bool(grade_answer_verl(s["text"], gt)),
                    "boxed_tail": bool(grade_answer_verl(s["text"][-700:], gt)),
                }
            )
        per_problem.append({"label": gt, "samples": scored, "problem_idx": row.get("problem_idx")})

    all_s = [s for p in per_problem for s in p["samples"]]
    n_all = len(all_s)

    def metrics(keep_fn, key="acc"):
        """keep_fn(sample) -> was the answer actually emitted under this cap?"""
        return avg_and_pass_at_k(per_problem, keep_fn, key)

    avg_full, pass_full = metrics(lambda s: True)
    # Counterfactual: if the cap had been 8192 tokens, everything longer would
    # have been cut off before the "Answer:" line and scored INVALID.
    avg_8k, pass_8k = metrics(lambda s: s["completion_tokens"] <= 8192)
    avg_16k, pass_16k = metrics(lambda s: s["completion_tokens"] <= 16384)
    avg_len, pass_len = metrics(lambda s: True, key="acc_lenient")
    avg_len_8k, pass_len_8k = metrics(lambda s: s["completion_tokens"] <= 8192, key="acc_lenient")
    finish_dist: dict[str, int] = {}
    for s in all_s:
        finish_dist[str(s["finish_reason"])] = finish_dist.get(str(s["finish_reason"]), 0) + 1

    trunc = sum(1 for s in all_s if s["finish_reason"] == "length") / n_all
    errs = sum(1 for s in all_s if str(s["finish_reason"]).startswith("error")) / n_all
    over_8k = sum(1 for s in all_s if s["completion_tokens"] > 8192) / n_all
    over_16k = sum(1 for s in all_s if s["completion_tokens"] > 16384) / n_all
    toks = sorted(s["completion_tokens"] for s in all_s)
    mean_tok = sum(toks) / len(toks)
    p50, p90, p99 = toks[len(toks) // 2], toks[int(len(toks) * 0.9)], toks[int(len(toks) * 0.99)]
    invalid = sum(1 for s in all_s if s["pred"] == "[INVALID]") / n_all

    summary = {
        "dataset": data_path.name,
        "tag": args.tag,
        "model": args.model,
        "n_problems": len(rows),
        "n_samples_per_problem": args.n,
        "sampling": sp,
        f"avg@{args.n}": round(avg_full * 100, 2),
        f"pass@{args.n}": round(pass_full * 100, 2),
        f"avg@{args.n}_if_cap_8192": round(avg_8k * 100, 2),
        f"pass@{args.n}_if_cap_8192": round(pass_8k * 100, 2),
        f"avg@{args.n}_if_cap_16384": round(avg_16k * 100, 2),
        f"pass@{args.n}_if_cap_16384": round(pass_16k * 100, 2),
        # lenient = accept a correct last \boxed{} too. Gap vs strict = the cost
        # of output-format non-compliance under rm_type=dapo.
        f"avg@{args.n}_lenient": round(avg_len * 100, 2),
        f"pass@{args.n}_lenient": round(pass_len * 100, 2),
        f"avg@{args.n}_lenient_if_cap_8192": round(avg_len_8k * 100, 2),
        f"pass@{args.n}_lenient_if_cap_8192": round(pass_len_8k * 100, 2),
        "finish_reason_dist": finish_dist,
        # Correct answer + response finished + still scored wrong == pure format cost.
        "format_penalty_count": sum(
            1 for s in all_s if s["acc_lenient"] and not s["acc"] and s["finish_reason"] == "stop"
        ),
        "format_penalty_denominator": n_all,
        "format_penalty_rate": round(
            100 * sum(1 for s in all_s if s["acc_lenient"] and not s["acc"] and s["finish_reason"] == "stop") / n_all, 2
        ),
        "compliance_rate": round(100 * sum(1 for s in all_s if s["pred"] != "[INVALID]") / n_all, 2),
        # >0 means rm_type=dapo cannot score this dataset (non-integer ground truth).
        "strict_scoring_error_rate": round(100 * sum(1 for s in all_s if s.get("strict_error")) / n_all, 2),
        "truncated_no_answer_rate": round(100 * sum(1 for s in all_s if s["finish_reason"] == "length") / n_all, 2),
        "truncation_rate_at_cap": round(trunc * 100, 2),
        "frac_over_8192_tokens": round(over_8k * 100, 2),
        "frac_over_16384_tokens": round(over_16k * 100, 2),
        "invalid_answer_rate": round(invalid * 100, 2),
        "request_error_rate": round(errs * 100, 2),
        "completion_tokens": {"mean": round(mean_tok, 1), "p50": p50, "p90": p90, "p99": p99},
        "wall_clock_sec": round(elapsed, 1),
    }

    out_dir = Path(args.out) if args.out else DATA_ROOT / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{data_path.stem}_{args.tag}_n{args.n}"
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{stem}.detail.json").write_text(json.dumps(per_problem, indent=1))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--port", type=int, default=30000)
    # No default: the served name identifies what is being evaluated, and a wrong
    # one silently scores a different model.
    ap.add_argument("--model", required=True,
                    help="model name as served by sglang (--model-path or --served-model-name)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: $MATH_DATA_ROOT/eval_results)")
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--tag", default="T1.0")
    asyncio.run(run(ap.parse_args()))
