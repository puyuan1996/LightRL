# Math RLVR toolkit

The implementation is split into five dependency-light modules:

| module | responsibility |
|---|---|
| `data` | portable roots, aliases, JSON/JSONL/HuggingFace loading, stable deduplication, manifests |
| `extractor` | `Answer:`, boxed and natural-language candidates with provenance |
| `verifier` | byte-identical semantic/strict verification used by train and eval |
| `scorer` | strict/lenient/boxed tracks and format/cap/collapse diagnostics |
| `stats` | paired deltas and deterministic bootstrap confidence intervals |

Post-hoc rescoring is part of `scorer` and is exposed by
`tools/evaluation/rescore_math_eval.py`; this keeps the scoring policy in one
place and guarantees that replay uses the same verifier as online evaluation.

Typical local flow:

```bash
# Optional override.  By default the resolver uses the shared canonical root
# `/mnt/shared-storage-user/puyuan/data/math_rlvr`, then repository-local data,
# and finally the repository benchmark directory.
export MATH_DATA_ROOT=/mnt/shared-storage-user/puyuan/data/math_rlvr
python tools/evaluation/prepare_math_data.py \
  --source hf://open-r1/DAPO-Math-17k --dataset dapo-math-17k
MODEL_PATH=/shared/ckpt/step-0 \
  bash tools/evaluation/launch_sglang_math.sh
MODEL=Qwen3-8B DATASETS='aime-2025 aime-2024 amc23 math-500' \
  bash tools/evaluation/run_math_base_evals.sh
python tools/evaluation/rescore_math_eval.py \
  /shared/math/eval_results/aime-2025_T1.0_n16.detail.json \
  --response-cap 65536
```

`--model` is required by `eval_math.py`; `MODEL_PATH`, `HF_CKPT` and `REF_LOAD`
are required by the serving/training entrypoints. This prevents a stale model
name from silently changing a result.
