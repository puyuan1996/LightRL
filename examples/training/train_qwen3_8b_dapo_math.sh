#!/usr/bin/env bash
# ============================================================================
# DAPO baseline with AIME2025 AS THE TRAINING SET — Qwen3-8B, 8x H200
#
#   train    : AIME2025 (30 problems)  <-- deliberately tiny; one step = one epoch
#   held-out : AIME2024 (30) in-training; AMC23 + MATH-500 offline on checkpoints
#   verifier : rm_type=math (boxed), NOT dapo
#   algo     : DAPO clip 0.2/0.28, token-level loss, KL off, dynamic sampling ON
#
# Why this differs from train_dapo_math.sh, all three changes forced by the
# 2026-07-28 run (issue #35 + its result comment):
#
#  1. rm_type=math instead of dapo. With rm_type=dapo the model spends its first
#     ~25-60 steps just learning to emit "Answer:" (+41..53pp of pure format
#     compliance, format_penalty_rate 47%->0%). On a 30-problem set the total step
#     count is small, so that format phase would dominate the whole run and any
#     DAPO-vs-DIVE-PO difference would be "who learns the format faster", not
#     "who explores better". rm_type=math scores the model's native \boxed{}
#     output, so reward measures math from step 1.
#     CAUTION: rm_type=math returns a SCALAR (0/1), not a dict, so --reward-key
#     and --eval-reward-key must stay UNSET or get_reward_value() does
#     scalar[key] -> TypeError. Rewards are therefore 0/1, not -1/+1.
#
#  2. Dynamic sampling ON. base already solves 26/30 AIME2025 problems at least
#     once (lenient Pass@16 = 86.67%), so all-correct groups pile up fast. The
#     17k run showed zero-std groups going 52% -> 72% within 40 steps with it
#     OFF; on 30 problems that would saturate sooner and harder.
#
#  3. group size 16 (not 8) and eval every 5 steps (not 25). Larger groups delay
#     zero-std and make pass@16 - the metric DIVE-PO must move - statistically
#     usable. The useful window is expected to be tens of steps, so the 17k run's
#     25-step eval spacing would miss it entirely.
#
# Usage:
#   bash train_dapo_aime25.sh                 # full run
#   SMOKE=1 bash train_dapo_aime25.sh         # 2-step smoke, no ckpt
#   DRY_RUN=1 bash train_dapo_aime25.sh       # print the command only
# ============================================================================
set -euo pipefail

SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${REPO_ROOT}/Megatron-LM}"
DATA_DIR="${MATH_DATA_ROOT:-${REPO_ROOT}/benchmarks/math}"

# CUDA_ENV_PREFIX is for hosts whose only CUDA toolkit lives inside a conda env,
# where cuDNN is a pip wheel that transformer_engine dlopen()s via ctypes and so
# ignores torch's RPATH. Both must also reach the Ray workers, hence
# RUNTIME_ENV_JSON below. Leave it unset when /usr/local/cuda exists.
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
if [[ -n "${CUDA_ENV_PREFIX:-}" ]]; then
  export CUDA_HOME="${CUDA_ENV_PREFIX}"
  export CUDA_PATH="${CUDA_ENV_PREFIX}"
  export PATH="${CUDA_ENV_PREFIX}/bin:${PATH}"
  NVIDIA_LIB_DIRS="$(ls -d "${CUDA_ENV_PREFIX}"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${CUDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  TRAIN_PYTHON="${CUDA_ENV_PREFIX}/bin/python"
fi
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TP_SIZE="${TP_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"

# Required: these identify what is being trained. A stale default would silently
# train a different model, and the resulting curve would be uninterpretable.
: "${HF_CKPT:?HF_CKPT is required (HF checkpoint directory)}"
: "${REF_LOAD:?REF_LOAD is required (Megatron torch_dist checkpoint directory)}"
# THE CHANGE: AIME2025 is now the training set.
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/aime-2025/aime-2025.jsonl}"
# In-training held-out. AIME2024 only: it is the same distribution as the training
# set and only 30 problems. AMC23 / MATH-500 are evaluated offline on checkpoints
# (MATH-500 at n=16 would be 8000 samples per eval).
EVAL_HELDOUT="${DATA_DIR}/aime-2024/aime-2024.jsonl"

SEED="${SEED:-1234}"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H%M%S)}"
if [[ "${SMOKE}" == "1" ]]; then
  RUN_ID="${RUN_ID:-dapo-aime25_qwen3-8b_SMOKE_${RUN_TAG}}"
else
  RUN_ID="${RUN_ID:-aime25-baseline_s${SEED}_${RUN_TAG}}"
fi
RUN_DIR="${DATA_DIR}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs"

# 30 problems: rollout_batch_size 30 => one step is exactly one epoch.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-30}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-20480}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
# The 17k run cashed the format bonus in 25-60 steps; with only 30 distinct
# prompts the useful window should be shorter still. 40 epochs is a ceiling, not
# a target -- watch the curves and stop when held-out stops moving.
NUM_ROLLOUT="${NUM_ROLLOUT:-21}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
# slime deletes older checkpoints by default (--max-ckpt-keep 1); intermediate
# ones are needed to score the curve offline at a mid-point, so keep a few.
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"

if [[ "${SMOKE}" == "1" ]]; then
  NUM_ROLLOUT="${SMOKE_NUM_ROLLOUT:-2}"
  EVAL_INTERVAL=999999
  SAVE_CKPT=""
else
  SAVE_CKPT="${SAVE_CKPT:-${DATA_DIR}/ckpt/${RUN_ID}}"
fi

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"   # MODEL_ARGS

CKPT_ARGS=(--hf-checkpoint "${HF_CKPT}" --ref-load "${REF_LOAD}")
if [[ -n "${SAVE_CKPT}" ]]; then
  # save every 5 steps: slime keeps only the LATEST checkpoint, so a frequent
  # cadence plus offline conversion is the only way to keep intermediate anchors.
  CKPT_ARGS+=(--save "${SAVE_CKPT}" --save-interval "${SAVE_INTERVAL}" --max-ckpt-keep "${MAX_CKPT_KEEP}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": true}'
  --rollout-shuffle
  # rm_type=math -> scalar 0/1 reward. Do NOT add --reward-key/--eval-reward-key.
  --rm-type math
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
  --balance-data
)

EVAL_ARGS=(
  --eval-interval "${EVAL_INTERVAL}"
  --eval-prompt-data aime2025_train "${PROMPT_DATA}" aime2024_heldout "${EVAL_HELDOUT}"
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
  --eval-max-response-len 32768
  --eval-max-context-len 40960
  --eval-top-p 1
)

DAPO_ARGS=(
  --seed "${SEED}"
  --rollout-seed "${SEED}"
  --advantage-estimator grpo
  --dynamic-history
  --eps-clip 0.2
  --eps-clip-high 0.28
  --calculate-per-token-loss
  --entropy-coef 0.00
)

# Dynamic sampling ON by default here (opposite of the 17k baseline).
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-1}"
DAPO_DYNAMIC_FILTER_PATH="${DAPO_DYNAMIC_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
DAPO_OVER_SAMPLING_BATCH_SIZE="${DAPO_OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 2))}"
if [[ "${DAPO_DYNAMIC_SAMPLING}" == "1" ]]; then
  DAPO_ARGS+=(
    --dynamic-sampling-filter-path "${DAPO_DYNAMIC_FILTER_PATH}"
    --over-sampling-batch-size "${DAPO_OVER_SAMPLING_BATCH_SIZE}"
  )
  echo "[config] dynamic sampling ON (over_sampling=${DAPO_OVER_SAMPLING_BATCH_SIZE})"
else
  echo "[config] dynamic sampling OFF"
fi

OPTIMIZER_ARGS=(
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.98
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}" --sequence-parallel
  --pipeline-model-parallel-size 1 --context-parallel-size 1
  --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
  --attention-dropout 0.0 --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
  --attention-backend flash
  --log-multi-turn
  # --log-passrate omitted: incompatible with --balance-data
  # (megatron_utils/data.py:605 -> utils/metric_utils.py:27 asserts
  #  len(raw_reward) == rollout_batch_size * n_samples_per_prompt).
)

# Fail before Ray starts rather than after, and keep the banner readable when the
# data has not been prepared yet (tools/evaluation/prepare_math_data.py).
if [[ "${DRY_RUN}" != "1" ]]; then
  for f in "${PROMPT_DATA}" "${EVAL_HELDOUT}"; do
    [[ -r "${f}" ]] || { echo "[ERROR] missing dataset ${f}; run tools/evaluation/prepare_math_data.py" >&2; exit 1; }
  done
fi
N_TRAIN_PROBLEMS="$(wc -l < "${PROMPT_DATA}" 2>/dev/null || echo '?')"

echo "==================================================================="
echo " RUN_ID     : ${RUN_ID}   $([[ "${SMOKE}" == "1" ]] && echo '(SMOKE)')"
echo " TRAIN SET  : ${PROMPT_DATA} (${N_TRAIN_PROBLEMS} problems)  <-- AIME2025"
echo " held-out   : aime2024 (30) in-training @ n=${N_SAMPLES_PER_EVAL_PROMPT}, every ${EVAL_INTERVAL} steps"
echo " verifier   : rm_type=math (boxed), rewards are 0/1"
echo " batch      : ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} = $((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT)) rollouts/step (= 1 epoch)"
echo " num_rollout: ${NUM_ROLLOUT} epochs (ceiling; stop when held-out plateaus)"
echo " ckpt       : ${SAVE_CKPT:-<disabled>} every ${SAVE_INTERVAL}"
echo "==================================================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" --num-gpus-per-node "${NUM_GPUS}" \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${DAPO_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
  echo; exit 0
fi

echo "[cleanup] stopping stale sglang / ray"
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 5

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"CUDA_HOME\": \"${CUDA_HOME}\",
    \"CUDA_PATH\": \"${CUDA_PATH}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\",
    \"HF_HOME\": \"${HF_HOME}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"PYTHONUNBUFFERED\": \"1\"
  }
}"

echo "${RUN_ID}" > "${RUN_DIR}/RUN_ID"
env | sort > "${RUN_DIR}/logs/launch_env.txt"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
  "${DAPO_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/logs/train.log"
