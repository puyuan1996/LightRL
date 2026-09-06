#!/usr/bin/env bash
# Qwen3-8B Math RLVR/DAPO recipe.
#
# This entrypoint is intentionally independent of the terminal-environment
# recipe.  It can run inside an rjob worker and accepts the same environment
# variables when called directly.  Use DRY_RUN=1 to inspect the exact Slime
# argv without starting Ray or allocating a GPU.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Silent checkpoint fallbacks are dangerous for a capability experiment.
: "${HF_CKPT:?set HF_CKPT to the checkpoint being trained}"
: "${REF_LOAD:?set REF_LOAD to the reference checkpoint}"
MATH_DATA_ROOT="${MATH_DATA_ROOT:-${REPO_ROOT}/benchmarks/math}"
TRAIN_DATASET="${TRAIN_DATASET:-aime-2025}"
REWARD_TYPE="${REWARD_TYPE:-math}"
RESPONSE_CAP="${RESPONSE_CAP:-32768}"
if [[ -z "${ROLLOUT_BATCH_SIZE:-}" ]]; then
  if [[ "${TRAIN_DATASET}" == "dapo" || "${TRAIN_DATASET}" == "dapo-math-17k" ]]; then
    ROLLOUT_BATCH_SIZE=256
  else
    ROLLOUT_BATCH_SIZE=30
  fi
fi
N_SAMPLES="${N_SAMPLES:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES))}"
EVAL_DATASETS="${EVAL_DATASETS:-aime-2025,aime-2024}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
SEED="${SEED:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
COLOCATE="${COLOCATE:-0}"
TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
MODEL_TRANSFORMER_IMPL="${MODEL_TRANSFORMER_IMPL:-transformer_engine}"
NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
MODEL_NUM_LAYERS="${MODEL_NUM_LAYERS:-36}"
MODEL_VOCAB_SIZE="${MODEL_VOCAB_SIZE:-151936}"
MODEL_HIDDEN_SIZE="${MODEL_HIDDEN_SIZE:-4096}"
MODEL_NUM_ATTENTION_HEADS="${MODEL_NUM_ATTENTION_HEADS:-32}"
MODEL_FFN_HIDDEN_SIZE="${MODEL_FFN_HIDDEN_SIZE:-12288}"
MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-40960}"
MODEL_NUM_QUERY_GROUPS="${MODEL_NUM_QUERY_GROUPS:-8}"
MODEL_NORM_EPSILON="${MODEL_NORM_EPSILON:-1e-6}"
MODEL_ROTARY_BASE="${MODEL_ROTARY_BASE:-1000000}"
SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
RUN_ID="${RUN_ID:-math-dapo-${TRAIN_DATASET}-seed${SEED}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/training/${RUN_ID}}"

dataset_path() {
  case "$1" in
    aime-2025|aime2025) echo "${MATH_DATA_ROOT}/aime-2025.jsonl" ;;
    aime-2024|aime2024) echo "${MATH_DATA_ROOT}/aime-2024.jsonl" ;;
    amc23|amc-23) echo "${MATH_DATA_ROOT}/amc23.jsonl" ;;
    math-500|math500) echo "${MATH_DATA_ROOT}/math-500.jsonl" ;;
    dapo|dapo-math-17k) echo "${MATH_DATA_ROOT}/dapo-math-17k.jsonl" ;;
    *) echo "$1" ;;
  esac
}
if [[ -z "${TRAIN_DATA:-}" ]]; then
  TRAIN_DATA="$(dataset_path "${TRAIN_DATASET}")"
fi
export TRAIN_DATA TRAIN_DATASET
[[ -f "${TRAIN_DATA}" ]] || { echo "[math-dapo] TRAIN_DATA does not exist: ${TRAIN_DATA}" >&2; exit 2; }

case "${REWARD_TYPE}" in math|dapo|boxed) ;; *) echo "[math-dapo] invalid REWARD_TYPE=${REWARD_TYPE}" >&2; exit 2 ;; esac
(( RESPONSE_CAP > 0 && ROLLOUT_BATCH_SIZE > 0 && N_SAMPLES > 0 )) || { echo "[math-dapo] cap/batch/n must be positive" >&2; exit 2; }
[[ -d "${HF_CKPT}" || -f "${HF_CKPT}" ]] || { echo "[math-dapo] HF_CKPT does not exist: ${HF_CKPT}" >&2; exit 2; }
[[ -d "${REF_LOAD}" || -f "${REF_LOAD}" ]] || { echo "[math-dapo] REF_LOAD does not exist: ${REF_LOAD}" >&2; exit 2; }

ROW_COUNT="$(PYTHONPATH="${REPO_ROOT}" "${TRAIN_PYTHON}" -c 'import sys; from tools.evaluation.math_rlvr.data import load_dataset; print(len(load_dataset(sys.argv[1], deduplicate=True)))' "${TRAIN_DATA}")"
[[ "${ROW_COUNT}" =~ ^[0-9]+$ && "${ROW_COUNT}" -gt 0 ]] || { echo "[math-dapo] empty training data" >&2; exit 2; }
if [[ "${TRAIN_DATASET}" == "dapo-math-17k" || "${TRAIN_DATASET}" == "dapo" ]]; then
  echo "[math-dapo] DAPO-Math-17k unique rows=${ROW_COUNT}; batch=${ROLLOUT_BATCH_SIZE}; eval=${EVAL_DATASETS}"
fi

EVAL_ARGS=()
IFS=',' read -r -a _EVAL_NAMES <<< "${EVAL_DATASETS}"
for eval_name in "${_EVAL_NAMES[@]}"; do
  eval_path="$(dataset_path "${eval_name}")"
  [[ -f "${eval_path}" ]] || { echo "[math-dapo] eval dataset does not exist: ${eval_path}" >&2; exit 2; }
  EVAL_ARGS+=(--eval-prompt-data "${eval_name}" "${eval_path}")
done

CMD=("${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py"
  --hf-checkpoint "${HF_CKPT}" --ref-load "${REF_LOAD}"
  --prompt-data "${TRAIN_DATA}" --input-key prompt --label-key label
  --reward-key score --rm-type "${REWARD_TYPE}"
  --custom-rm-path tools.evaluation.math_rlvr.reward.reward_func
  --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}" --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --rollout-max-response-len "${RESPONSE_CAP}" --rollout-max-context-len "$((RESPONSE_CAP + 4096))"
  --rollout-temperature 1.0 --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28
  --calculate-per-token-loss --eval-interval "${EVAL_INTERVAL}"
  --n-samples-per-eval-prompt "${EVAL_N_SAMPLES}" --eval-max-response-len "${RESPONSE_CAP}"
  --eval-input-key prompt --eval-label-key label --eval-reward-key score
  --eval-top-p "${EVAL_TOP_P}" --train-backend "${TRAIN_BACKEND}"
  --transformer-impl "${MODEL_TRANSFORMER_IMPL}"
  --num-layers "${MODEL_NUM_LAYERS}" --vocab-size "${MODEL_VOCAB_SIZE}" --hidden-size "${MODEL_HIDDEN_SIZE}"
  --num-attention-heads "${MODEL_NUM_ATTENTION_HEADS}" --ffn-hidden-size "${MODEL_FFN_HIDDEN_SIZE}"
  --max-position-embeddings "${MODEL_MAX_POSITION_EMBEDDINGS}"
  --normalization RMSNorm --norm-epsilon "${MODEL_NORM_EPSILON}"
  --position-embedding-type rope --rotary-base "${MODEL_ROTARY_BASE}"
  --group-query-attention --num-query-groups "${MODEL_NUM_QUERY_GROUPS}"
  --swiglu --disable-bias-linear --untie-embeddings-and-output-weights
  --actor-num-nodes 1 --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --seed "${SEED}" --save "${RUN_DIR}/checkpoints" --save-interval "${SAVE_INTERVAL:-20}"
  "${EVAL_ARGS[@]}")

if [[ "${COLOCATE}" == "1" ]]; then
  CMD+=(--colocate)
else
  (( ROLLOUT_GPUS > 0 )) || { echo "[math-dapo] ROLLOUT_GPUS must be positive when COLOCATE=0" >&2; exit 2; }
  CMD+=(--rollout-num-gpus "${ROLLOUT_GPUS}")
fi

mkdir -p "${RUN_DIR}/config" "${RUN_DIR}/logs"
PYTHONPATH="${REPO_ROOT}" "${TRAIN_PYTHON}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"train_data":sys.argv[2],"train_rows":int(sys.argv[3]),"train_batch_size":int(sys.argv[4]),"eval_datasets":sys.argv[5].split(","),"reward_type":sys.argv[6],"response_cap":int(sys.argv[7]),"seed":int(sys.argv[8])}, indent=2)+"\n")' "${RUN_DIR}/config/math_rlvr.json" "${TRAIN_DATA}" "${ROW_COUNT}" "${ROLLOUT_BATCH_SIZE}" "${EVAL_DATASETS}" "${REWARD_TYPE}" "${RESPONSE_CAP}" "${SEED}"
export MATH_RLVR_REWARD_TYPE="${REWARD_TYPE}" MATH_RLVR_RESPONSE_CAP="${RESPONSE_CAP}"
export NVTE_FUSED_ATTN

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
exec "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/logs/train.log"
