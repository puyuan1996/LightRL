# Math RLVR evaluation (AIME / AMC23 / MATH-500)

## TL;DR

This is the tooling for training a math RLVR run and evaluating it on AIME2025, AIME2024, AMC23 and MATH-500. The one thing to take away before running anything: **on this task the verifier definition, not the model, dominates the number you read.** The same Qwen3-8B samples score `Avg@16 = 20.62%` on AIME2025 under the strict `rm_type=dapo` rule and `67.71%` when a correct `\boxed{}` is also accepted — a 3.3x gap, of which every point is a sample that was correct and finished and still scored wrong. So every metric here is reported on **two tracks at once**, and a single-track number is not interpretable. The second thing: `rm_type=dapo` cannot score ~30% of MATH-500 at all, and an 8192-token cap costs AIME2025 `Pass@16` 76.67% → 26.67%. Pipeline is five commands: `prepare_math_data.py` → `launch_sglang_math.sh` → `run_math_base_evals.sh` → `rescore_math_eval.py` → `math_paired_stats.py`. Measurements behind all of this: [issue #35](https://github.com/HansBug/OpenClaw-RL/issues/35), [#36](https://github.com/HansBug/OpenClaw-RL/issues/36), [#37](https://github.com/HansBug/OpenClaw-RL/issues/37).

## 1. The two tracks, and why both

`rm_type=dapo` extracts the answer with `(?i)Answer\s*:\s*([^\n]+)` and scores anything else `[INVALID]`. Qwen3 almost always answers with `\boxed{}` instead — which is what its own model card recommends for math. The result is that the training reward measures output format at least as much as it measures arithmetic.

| track | rule | what it measures |
|---|---|---|
| strict | `compute_score_dapo`, i.e. `rm_type=dapo` | exactly the reward the DAPO run optimises |
| lenient | strict, **or** a correct last `\boxed{}` via `grade_answer_verl` | a lower bound on math ability, insensitive to which format was used |
| pure boxed | `\boxed{}` only | the reward a `rm_type=math` run optimises |

`eval_math.py` records strict and lenient for every sample plus `boxed_full` / `boxed_tail`, so all three can be recomputed later from stored records without regenerating.

Two things the lenient track is not. It is a **lower** bound, not an upper one: a sample answering `**70` fails both tracks. And it saturates — AMC23 `Pass@16` is already 100.00% on base, so it cannot resolve improvements there.

## 2. What the format rule costs

`format_penalty_rate` counts samples that are lenient-correct, finished (`finish_reason == "stop"`), strict-wrong, and scorable. On AIME2025 it is 47.08% — identical to the 47.08pp gap between the two tracks, which is an identity rather than an approximation: the whole gap is carried by finished, correct answers.

| dataset | k | strict Avg@k | lenient Avg@k | format penalty | no `Answer:` line | wrote it, regex mis-grabbed |
|---|---:|---:|---:|---:|---:|---:|
| AIME2025 | 16 | 20.62 | 67.71 | 47.08% | 40.00% | 7.08% |
| AIME2024 | 16 | 20.42 | 78.33 | 57.71% | 50.21% | 7.50% |
| AMC23 | 16 | 42.19 | 93.44 | 51.25% | 45.78% | 5.47% |
| MATH-500 | 4 | 36.70 † | 86.20 | 27.45% | 24.00% | 3.45% |

† also contains unscorable samples, see §3; k=4 rather than 16, so it is not comparable across this column. This row is over all 500 problems. §5 and §8 quote MATH-500 on the **349 scorable problems** instead, where the lenient baseline is 92.0 rather than 86.20 — the two populations must not be mixed, and a comparison that changes population silently degrades the lenient track into a boxed-only one.

The mis-grab case is the nastier one: `**Answer:**\n$$\boxed{588}$$` parses to `pred='**'`. The model complied and still lost the point.

## 3. `rm_type=dapo` cannot score every dataset

`is_correct_minerva` in `slime/slime/rollout/rm_hub/math_dapo_utils.py` does `gt = str(int(float(gt)))  # in dapo, all answers are integers`, which raises on LaTeX ground truths. MATH-500 has coordinate pairs, fractions and intervals: **30.20%** of it cannot be scored strictly.

This tooling does not patch that function — it is the training reward, and changing it would move every existing run's reward distribution. `eval_math.py` catches the exception, scores the sample `[STRICT_ERROR]`, and reports `strict_scoring_error_rate` so the loss is visible instead of silently landing in the wrong-answer bucket. **For a MATH-500 column, use `rm_type=math`.**

## 4. The response cap is a first-order parameter

8192 tokens is not a safe default for competition math. AIME2025 answers average 17,015 tokens with p90 at 31,714, and **79.79%** exceed 8192. Re-scoring the same samples under shorter caps:

| dataset | Avg@k cap 8k | cap 16k | cap 32k | Pass@k cap 8k | cap 16k | cap 32k |
|---|---:|---:|---:|---:|---:|---:|
| AIME2025 (k=16) | 8.12 | 17.29 | **20.62** | 26.67 | 60.00 | **76.67** |
| AIME2024 (k=16) | 10.62 | 18.96 | 20.42 | 36.67 | 73.33 | 80.00 |
| AMC23 (k=16) | 33.44 | 41.56 | 42.19 | 75.00 | 95.00 | 97.50 |
| MATH-500 (k=4) | 34.90 | 36.60 | 36.70 | 54.80 | 58.80 | 59.00 |

Read this as an **upper bound** on the cost of a small cap, not a conservative estimate: it simulates inference-time truncation on samples generated at 32768, and a run trained at a smaller cap would have learned shorter answers. What is not an extrapolation: at 32768 the real truncation rate is only 8.54%, and Qwen3-8B's model card suggests 38,912 output tokens for competition math.

There is also a mechanism worth watching rather than a measurement: truncation → no answer line → reward −1 → a whole group at −1 → zero group std → no gradient. That is the zero-variance collapse the exploration work targets.

## 5. What RL actually moved

Across three seeds on AIME2025 as the training set, evaluated on held-out AIME2024: the training reward (pure boxed) rose **+8.82pp** while the capability proxy (either format accepted) moved **−1.32pp, not significant**. At most 75–90% of the gain is explained by closing the format gap — an upper bound, since any gain not exceeding the gap is attributed to format regardless of its real cause. Meanwhile the strict track collapsed from 20.42% to 0.35%, because the model dropped the `Answer:` habit it no longer needed.

In a separate sweep of four configurations at **one seed each** (#36), each verifier trained the model into its own format and destroyed the other — directional evidence, not a statistical result, since those intervals carry problem-level sampling only and no training randomness. An `Answer:` reward took compliance from 34% to 98.75% and pushed boxed correctness from 68.12% down to **2.50%**; a boxed reward did the reverse.

Two consequences for anyone using this as a baseline. Report all three tracks, and make the verifier byte-identical on both sides of a comparison — a single-track number is not interpretable. And on AIME2025 differences below about 5pp are not readable: the seed-to-seed range alone is 5.00pp and re-evaluating one model twice differs by 2.92pp. That is a floor, not a safety line; estimated from n=3 it could itself lie anywhere in 2.6–31pp.

Where capability did move measurably: MATH-500 over its 349 scorable problems, **+3.41pp**, robust under any multiple-comparison correction (Holm 0.00012, CI [+1.89, +5.18]). So "format explains most of the gain" is the accurate statement; "RLVR does nothing for math" is not.

## 6. Running it

```bash
# 1. Datasets -> $MATH_DATA_ROOT (default: benchmarks/math). Downloads from HF.
#    DAPO-Math-17k ships ~1.79M rows because each prompt is pre-duplicated ~103x
#    for verl's multi-epoch loop; this deduplicates to 17,255 unique problems.
python tools/evaluation/prepare_math_data.py

# 2. Serve the checkpoint. CTX must exceed the eval --max-tokens.
MODEL=/path/to/Qwen3-8B TP=8 bash tools/evaluation/launch_sglang_math.sh

# 3. Sweep. T=1.0/top_p=1.0 matches slime's in-training eval defaults, so these
#    numbers line up with the eval curve a training run emits.
MODEL=/path/to/Qwen3-8B bash tools/evaluation/run_math_base_evals.sh

# 4. Recompute derived fields from stored per-sample records (no generation), and
#    compare two runs with paired bootstrap / permutation tests.
python tools/evaluation/rescore_math_eval.py
python tools/evaluation/math_paired_stats.py --help
```

Training is `examples/training/train_qwen3_8b_dapo_math.sh`. Note it trains on **AIME2025's 30 problems** with AIME2024 held out, not on the 17,255-prompt DAPO-Math-17k that step 1 also prepares; point `PROMPT_DATA` at the 17k set for the larger run. `DRY_RUN=1` prints the resolved argv without launching, `SMOKE=1` runs two steps without checkpointing. It requires `HF_CKPT` and `REF_LOAD` rather than defaulting them, because a stale default silently trains a different model. It defaults to `rm_type=math`, not `dapo`: with `dapo` roughly the first 60 steps go into learning to emit `Answer:`, which on a 30-problem set would dominate the whole run and turn any DAPO-vs-variant comparison into "who learns the format faster".

`eval_math.py` writes `<dataset>_<tag>_n<k>.summary.json` and `.detail.json` per run. The detail file keeps per-sample `completion_tokens`, `finish_reason`, both track verdicts and the last 700 characters of the response, which is what makes every table above recomputable without regenerating.

## 7. Sampling temperature

T=1.0/top_p=1.0 is the default here for one reason: it matches the in-training eval, so offline and curve numbers are comparable. It is not a quality claim. Against Qwen3's recommended T=0.6/top_p=0.95 the strict `Pass@16` differs by 10.00pp, but that is 3 problems at paired McNemar p = 0.250, while the **lenient `Pass@16` is identical at 86.67% with zero discordant pairs**. So this data cannot resolve a capability difference, and the most parsimonious reading of the strict gap is format-compliance fluctuation. Lenient `Avg@16` does differ slightly (67.71 vs 66.04), and that pair was not tested.

## 8. Limits

n=30 is small. The strict AIME2024−AIME2025 gap is −0.21pp with 95% CI [−9.79, +9.58]: the two years are not distinguishable from each other, let alone two training configurations. Anything that depends on a small effect needs multiple seeds and paired tests, which is what `math_paired_stats.py` is for.

AMC23 overlaps the training set by at least 40%, so a gain there should not carry weight. The starting point is post-trained Qwen3-8B, not pretrained weights, and its lenient baselines are already at 67.7 / 78.3 / 93.4 (and 92.0 on MATH-500's 349 scorable problems) — "capability barely moved" means "these configurations did not extract more from an already-saturated start", not "RLVR does not improve math".

This document ships the tooling and the previously measured numbers; nothing here was re-run as part of adding it. The raw per-sample records, environment fingerprints and figures live with the three issues linked at the top.
