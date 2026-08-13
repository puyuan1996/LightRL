#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${WM_TRAJECTORIES:?Set WM_TRAJECTORIES to a SETA directory, records JSONL, or verified replay .pt}"
if [[ ! -e "${WM_TRAJECTORIES}" ]]; then
  echo "[wm-seta-latent] input does not exist: ${WM_TRAJECTORIES}" >&2
  exit 2
fi

WM_OUTPUT_DIR="${WM_OUTPUT_DIR:-${REPO_ROOT}/runs/world_model_seta_latent/$(date +%Y%m%d_%H%M%S)}"
WM_ENCODER="${WM_ENCODER:-hash}"
WM_HF_MODEL="${WM_HF_MODEL:-}"
WM_MAX_TRAJECTORIES="${WM_MAX_TRAJECTORIES:-}"
WM_MAX_TRANSITIONS="${WM_MAX_TRANSITIONS:-}"
WM_EPOCHS="${WM_EPOCHS:-3}"
WM_TRAIN_BATCHES_PER_EPOCH="${WM_TRAIN_BATCHES_PER_EPOCH:-0}"
WM_VALIDATION_BATCHES_PER_EPOCH="${WM_VALIDATION_BATCHES_PER_EPOCH:-0}"
WM_MIN_TRAIN_SECONDS="${WM_MIN_TRAIN_SECONDS:-0}"
WM_MAX_EPOCHS="${WM_MAX_EPOCHS:-}"
WM_VALIDATION_INTERVAL_EPOCHS="${WM_VALIDATION_INTERVAL_EPOCHS:-1}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-16}"
WM_ENCODE_BATCH_SIZE="${WM_ENCODE_BATCH_SIZE:-2}"
WM_MICROBATCH_QUEUE_SIZE="${WM_MICROBATCH_QUEUE_SIZE:-0}"
WM_CHECKPOINT_SELECTION="${WM_CHECKPOINT_SELECTION:-final_epoch}"
WM_LATENT_DIM="${WM_LATENT_DIM:-128}"
WM_PREDICTOR_TYPE="${WM_PREDICTOR_TYPE:-adaln}"
WM_PREDICTOR_NUM_HEADS="${WM_PREDICTOR_NUM_HEADS:-4}"
WM_PREDICTOR_DEPTH="${WM_PREDICTOR_DEPTH:-2}"
WM_PREDICTOR_MLP_RATIO="${WM_PREDICTOR_MLP_RATIO:-4.0}"
WM_PREDICTION_TARGET="${WM_PREDICTION_TARGET:-feedback}"
WM_PREDICTOR_INPUT_MODE="${WM_PREDICTOR_INPUT_MODE:-observed}"
WM_PREDICTION_FORM="${WM_PREDICTION_FORM:-direct}"
if [[ "${WM_PREDICTION_TARGET}" == "next_state" ]]; then
  WM_PRED_LOSS_TYPE="${WM_PRED_LOSS_TYPE:-scaled_mse}"
  WM_OBJECTIVE_POPULATION="${WM_OBJECTIVE_POPULATION:-has_next}"
  WM_ACTION_CONTRAST_COEF="${WM_ACTION_CONTRAST_COEF:-0}"
  WM_ALIGNMENT_COEF="${WM_ALIGNMENT_COEF:-0}"
else
  WM_PRED_LOSS_TYPE="${WM_PRED_LOSS_TYPE:-mse}"
  WM_OBJECTIVE_POPULATION="${WM_OBJECTIVE_POPULATION:-all}"
  WM_ACTION_CONTRAST_COEF="${WM_ACTION_CONTRAST_COEF:-0.1}"
  WM_ALIGNMENT_COEF="${WM_ALIGNMENT_COEF:-0.1}"
fi
WM_FEEDBACK_AUX_COEF="${WM_FEEDBACK_AUX_COEF:-0}"
WM_VAL_RATIO="${WM_VAL_RATIO:-0.2}"
WM_SPLIT_GROUP_KEY="${WM_SPLIT_GROUP_KEY:-task_id}"
WM_SPLIT_MANIFEST="${WM_SPLIT_MANIFEST:-}"
WM_VALUE_COEF="${WM_VALUE_COEF:-0}"
WM_LR="${WM_LR:-1e-4}"
WM_WEIGHT_DECAY="${WM_WEIGHT_DECAY:-0.01}"
WM_SIGREG_COEF="${WM_SIGREG_COEF:-0.09}"
WM_SIGREG_SCOPE="${WM_SIGREG_SCOPE:-state}"
WM_SEED="${WM_SEED:-42}"
WM_SPLIT_SEED="${WM_SPLIT_SEED:-}"
WM_BACKPROP_TO_LLM="${WM_BACKPROP_TO_LLM:-0}"
WM_SAVE_UPDATED_LLM="${WM_SAVE_UPDATED_LLM:-0}"
WM_FIXED_TARGET_BACKBONE="${WM_FIXED_TARGET_BACKBONE:-0}"
WM_LLM_TRAIN_MODE="${WM_LLM_TRAIN_MODE:-full}"
WM_LORA_RANK="${WM_LORA_RANK:-16}"
WM_LORA_ALPHA="${WM_LORA_ALPHA:-32}"
WM_LORA_DROPOUT="${WM_LORA_DROPOUT:-0.05}"
WM_LORA_TARGET_MODULES="${WM_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
WM_LLM_LR="${WM_LLM_LR:-1e-6}"
WM_USE_DAPO_REPLAY_BUFFER="${WM_USE_DAPO_REPLAY_BUFFER:-0}"
WM_REPLAY_BUFFER_SIZE="${WM_REPLAY_BUFFER_SIZE:-2048}"
WM_ALLOW_UNVERIFIED_REPLAY="${WM_ALLOW_UNVERIFIED_REPLAY:-0}"
WM_ALLOW_UNVERIFIED_WORLD_MODEL_RECORDS="${WM_ALLOW_UNVERIFIED_WORLD_MODEL_RECORDS:-0}"
WM_HIDDEN_CACHE_INPUT="${WM_HIDDEN_CACHE_INPUT:-}"
WM_ALLOW_CACHE_ENCODER_MISMATCH="${WM_ALLOW_CACHE_ENCODER_MISMATCH:-0}"
WM_ALLOW_UNVERIFIED_VALUE_LABELS="${WM_ALLOW_UNVERIFIED_VALUE_LABELS:-0}"
WM_HF_ALLOW_DOWNLOAD="${WM_HF_ALLOW_DOWNLOAD:-0}"
WM_HF_TRUST_REMOTE_CODE="${WM_HF_TRUST_REMOTE_CODE:-0}"
WM_ALLOW_APPROXIMATE_ACTION_BOUNDARY="${WM_ALLOW_APPROXIMATE_ACTION_BOUNDARY:-0}"
WM_REQUIRE_TOOL_FEEDBACK="${WM_REQUIRE_TOOL_FEEDBACK:-0}"
WM_STOP_GRAD_TARGET="${WM_STOP_GRAD_TARGET:-0}"
WM_TARGET_EMA_DECAY="${WM_TARGET_EMA_DECAY:-0.996}"
WM_TARGET_GEOMETRY="${WM_TARGET_GEOMETRY:-learned_shared_v2}"
WM_FIXED_TARGET_SEED="${WM_FIXED_TARGET_SEED:-20260731}"
WM_DEVICE="${WM_DEVICE:-auto}"
WM_HF_DTYPE="${WM_HF_DTYPE:-auto}"
WM_HIDDEN_LAYER="${WM_HIDDEN_LAYER:--1}"
WM_ACTION_POOL="${WM_ACTION_POOL:-mean}"
WM_MAX_CONTEXT_TOKENS="${WM_MAX_CONTEXT_TOKENS:-1536}"
WM_MAX_ACTION_TOKENS="${WM_MAX_ACTION_TOKENS:-512}"
WM_MAX_FEEDBACK_TOKENS="${WM_MAX_FEEDBACK_TOKENS:-512}"
WM_ENCODER_LONG_TEXT_MODE="${WM_ENCODER_LONG_TEXT_MODE:-tail_v1}"
WM_CHUNK_FORWARD_BATCH_SIZE="${WM_CHUNK_FORWARD_BATCH_SIZE:-16}"
WM_STATE_VIEW="${WM_STATE_VIEW:-full_context_v1}"
WM_BELIEF_MAX_EVENTS="${WM_BELIEF_MAX_EVENTS:-3}"
WM_MAX_TEXT_CHARS="${WM_MAX_TEXT_CHARS:-4096}"
WM_READY_FILE="${WM_READY_FILE:-}"
WM_START_FILE="${WM_START_FILE:-}"
WM_PHASE_FILE="${WM_PHASE_FILE:-}"
WM_BARRIER_TIMEOUT_SECONDS="${WM_BARRIER_TIMEOUT_SECONDS:-1800}"
PYTHON_BIN="${PYTHON_BIN:-${WM_PYTHON_BIN:-python}}"
WM_REQUIRE_PYTEST="${WM_REQUIRE_PYTEST:-0}"

python_is_usable() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    return 1
  fi
  if [[ "${candidate}" != /* ]]; then
    if [[ -x "/${candidate}" ]]; then
      candidate="/${candidate}"
    else
      candidate="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
  fi
  [[ -n "${candidate}" && -x "${candidate}" ]] || return 1

  local import_stmt="import numpy; import torch; import transformers"
  if [[ "${WM_REQUIRE_PYTEST}" == "1" ]]; then
    import_stmt="${import_stmt}; import pytest"
  fi
  "${candidate}" -c "${import_stmt}"
}

resolve_python() {
  local requested="${PYTHON_BIN:-}" resolved
  for candidate in \
    "${requested}" \
    "${CONDA_PREFIX:-}/bin/python" \
    /root/miniconda3/bin/python /opt/conda/bin/python \
    /usr/local/bin/python /usr/local/bin/python3 \
    /usr/bin/python /usr/bin/python3 python3 python
  do
    [[ -n "${candidate}" ]] || continue
    if python_is_usable "${candidate}"; then
      if [[ "${candidate}" == /* ]]; then
        resolved="${candidate}"
      else
        resolved="$(command -v "${candidate}" 2>/dev/null || true)"
        if [[ -z "${resolved}" && -x "/${candidate}" ]]; then
          resolved="/${candidate}"
        fi
      fi
      if [[ -n "${resolved}" ]]; then
        printf '%s\n' "${resolved}"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(resolve_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[wm-seta-latent] no Python can import required runtime modules" >&2
  echo "[wm-seta-latent] candidate tested: numpy/torch/transformers" >&2
  echo "[wm-seta-latent] set PYTHON_BIN to a working interpreter or fix environment" >&2
  exit 2
fi

args=(
  --input "${WM_TRAJECTORIES}"
  --output-dir "${WM_OUTPUT_DIR}"
  --encoder "${WM_ENCODER}"
  --epochs "${WM_EPOCHS}"
  --train-batches-per-epoch "${WM_TRAIN_BATCHES_PER_EPOCH}"
  --validation-batches-per-epoch "${WM_VALIDATION_BATCHES_PER_EPOCH}"
  --min-train-seconds "${WM_MIN_TRAIN_SECONDS}"
  --validation-interval-epochs "${WM_VALIDATION_INTERVAL_EPOCHS}"
  --batch-size "${WM_BATCH_SIZE}"
  --encode-batch-size "${WM_ENCODE_BATCH_SIZE}"
  --microbatch-queue-size "${WM_MICROBATCH_QUEUE_SIZE}"
  --checkpoint-selection "${WM_CHECKPOINT_SELECTION}"
  --latent-dim "${WM_LATENT_DIM}"
  --predictor-type "${WM_PREDICTOR_TYPE}"
  --predictor-num-heads "${WM_PREDICTOR_NUM_HEADS}"
  --predictor-depth "${WM_PREDICTOR_DEPTH}"
  --predictor-mlp-ratio "${WM_PREDICTOR_MLP_RATIO}"
  --prediction-target "${WM_PREDICTION_TARGET}"
  --predictor-input-mode "${WM_PREDICTOR_INPUT_MODE}"
  --prediction-form "${WM_PREDICTION_FORM}"
  --pred-loss-type "${WM_PRED_LOSS_TYPE}"
  --objective-population "${WM_OBJECTIVE_POPULATION}"
  --val-ratio "${WM_VAL_RATIO}"
  --split-group-key "${WM_SPLIT_GROUP_KEY}"
  --value-coef "${WM_VALUE_COEF}"
  --lr "${WM_LR}"
  --llm-lr "${WM_LLM_LR}"
  --weight-decay "${WM_WEIGHT_DECAY}"
  --sigreg-coef "${WM_SIGREG_COEF}"
  --sigreg-scope "${WM_SIGREG_SCOPE}"
  --action-contrast-coef "${WM_ACTION_CONTRAST_COEF}"
  --alignment-coef "${WM_ALIGNMENT_COEF}"
  --feedback-aux-coef "${WM_FEEDBACK_AUX_COEF}"
  --seed "${WM_SEED}"
  --replay-buffer-size "${WM_REPLAY_BUFFER_SIZE}"
  --device "${WM_DEVICE}"
  --hf-dtype "${WM_HF_DTYPE}"
  --hidden-layer "${WM_HIDDEN_LAYER}"
  --action-pool "${WM_ACTION_POOL}"
  --max-context-tokens "${WM_MAX_CONTEXT_TOKENS}"
  --max-action-tokens "${WM_MAX_ACTION_TOKENS}"
  --max-feedback-tokens "${WM_MAX_FEEDBACK_TOKENS}"
  --encoder-long-text-mode "${WM_ENCODER_LONG_TEXT_MODE}"
  --chunk-forward-batch-size "${WM_CHUNK_FORWARD_BATCH_SIZE}"
  --state-view "${WM_STATE_VIEW}"
  --belief-max-events "${WM_BELIEF_MAX_EVENTS}"
  --max-text-chars "${WM_MAX_TEXT_CHARS}"
  --target-ema-decay "${WM_TARGET_EMA_DECAY}"
  --target-geometry "${WM_TARGET_GEOMETRY}"
  --fixed-target-seed "${WM_FIXED_TARGET_SEED}"
  --llm-train-mode "${WM_LLM_TRAIN_MODE}"
  --lora-rank "${WM_LORA_RANK}"
  --lora-alpha "${WM_LORA_ALPHA}"
  --lora-dropout "${WM_LORA_DROPOUT}"
  --lora-target-modules "${WM_LORA_TARGET_MODULES}"
  --barrier-timeout-seconds "${WM_BARRIER_TIMEOUT_SECONDS}"
)

if [[ -n "${WM_MAX_EPOCHS}" ]]; then
  args+=(--max-epochs "${WM_MAX_EPOCHS}")
fi
if [[ -n "${WM_READY_FILE}" || -n "${WM_START_FILE}" ]]; then
  [[ -n "${WM_READY_FILE}" && -n "${WM_START_FILE}" ]] || {
    echo "[wm-seta-latent] WM_READY_FILE and WM_START_FILE must be set together" >&2
    exit 2
  }
  args+=(--ready-file "${WM_READY_FILE}" --start-file "${WM_START_FILE}")
fi
if [[ -n "${WM_PHASE_FILE}" ]]; then
  args+=(--phase-file "${WM_PHASE_FILE}")
fi

if [[ -n "${WM_SPLIT_SEED}" ]]; then
  args+=(--split-seed "${WM_SPLIT_SEED}")
fi
if [[ -n "${WM_SPLIT_MANIFEST}" ]]; then
  [[ -s "${WM_SPLIT_MANIFEST}" ]] || {
    echo "[wm-seta-latent] split manifest is missing: ${WM_SPLIT_MANIFEST}" >&2
    exit 2
  }
  args+=(--split-manifest "${WM_SPLIT_MANIFEST}")
fi

if [[ -n "${WM_MAX_TRAJECTORIES}" ]]; then
  args+=(--max-trajectories "${WM_MAX_TRAJECTORIES}")
fi
if [[ -n "${WM_MAX_TRANSITIONS}" ]]; then
  args+=(--max-transitions "${WM_MAX_TRANSITIONS}")
fi
if [[ "${WM_ENCODER}" == "hf-policy" ]]; then
  : "${WM_HF_MODEL:?Set WM_HF_MODEL when WM_ENCODER=hf-policy}"
  args+=(--hf-model "${WM_HF_MODEL}")
  if [[ "${WM_HF_ALLOW_DOWNLOAD}" == "1" ]]; then
    args+=(--hf-allow-download)
  else
    args+=(--hf-local-files-only)
  fi
fi
if [[ "${WM_HF_TRUST_REMOTE_CODE}" == "1" ]]; then
  args+=(--hf-trust-remote-code)
fi
if [[ "${WM_ALLOW_APPROXIMATE_ACTION_BOUNDARY}" == "1" ]]; then
  args+=(--allow-approximate-action-boundary)
fi
if [[ "${WM_BACKPROP_TO_LLM}" == "1" ]]; then
  args+=(--backprop-to-llm)
fi
if [[ "${WM_SAVE_UPDATED_LLM}" == "1" ]]; then
  args+=(--save-updated-llm)
fi
if [[ "${WM_FIXED_TARGET_BACKBONE}" == "1" ]]; then
  args+=(--fixed-target-backbone)
fi
if [[ "${WM_USE_DAPO_REPLAY_BUFFER}" == "1" ]]; then
  args+=(--use-dapo-replay-buffer)
fi
if [[ "${WM_ALLOW_UNVERIFIED_REPLAY}" == "1" ]]; then
  args+=(--allow-unverified-replay)
fi
if [[ "${WM_ALLOW_UNVERIFIED_WORLD_MODEL_RECORDS}" == "1" ]]; then
  args+=(--allow-unverified-world-model-records)
fi
if [[ -n "${WM_HIDDEN_CACHE_INPUT}" ]]; then
  args+=(--hidden-cache-input "${WM_HIDDEN_CACHE_INPUT}")
fi
if [[ "${WM_ALLOW_CACHE_ENCODER_MISMATCH}" == "1" ]]; then
  args+=(--allow-cache-encoder-mismatch)
fi
if [[ "${WM_ALLOW_UNVERIFIED_VALUE_LABELS}" == "1" ]]; then
  args+=(--allow-unverified-value-labels)
fi
if [[ "${WM_REQUIRE_TOOL_FEEDBACK}" == "1" ]]; then
  args+=(--require-tool-feedback)
fi
if [[ "${WM_STOP_GRAD_TARGET}" == "1" ]]; then
  args+=(--stop-grad-target)
fi

echo "[wm-seta-latent] input:  ${WM_TRAJECTORIES}"
echo "[wm-seta-latent] output: ${WM_OUTPUT_DIR}"
echo "[wm-seta-latent] encoder=${WM_ENCODER} split=${WM_SPLIT_GROUP_KEY} val_ratio=${WM_VAL_RATIO}"
echo "[wm-seta-latent] prediction_target=${WM_PREDICTION_TARGET} pred_loss=${WM_PRED_LOSS_TYPE} population=${WM_OBJECTIVE_POPULATION}"
echo "[wm-seta-latent] predictor_input_mode=${WM_PREDICTOR_INPUT_MODE}"
echo "[wm-seta-latent] prediction_form=${WM_PREDICTION_FORM}"
echo "[wm-seta-latent] state_view=${WM_STATE_VIEW} belief_max_events=${WM_BELIEF_MAX_EVENTS}"
echo "[wm-seta-latent] long_text=${WM_ENCODER_LONG_TEXT_MODE} chunk_batch=${WM_CHUNK_FORWARD_BATCH_SIZE}"
echo "[wm-seta-latent] target_geometry=${WM_TARGET_GEOMETRY} fixed_target_seed=${WM_FIXED_TARGET_SEED}"
echo "[wm-seta-latent] sigreg_scope=${WM_SIGREG_SCOPE} sigreg_coef=${WM_SIGREG_COEF}"
echo "[wm-seta-latent] microbatch_queue_size=${WM_MICROBATCH_QUEUE_SIZE} checkpoint_selection=${WM_CHECKPOINT_SELECTION}"
echo "[wm-seta-latent] feedback_aux_coef=${WM_FEEDBACK_AUX_COEF}"
echo "[wm-seta-latent] min_train_seconds=${WM_MIN_TRAIN_SECONDS} max_epochs=${WM_MAX_EPOCHS:-unbounded}"
echo "[wm-seta-latent] train_batches_per_epoch=${WM_TRAIN_BATCHES_PER_EPOCH} validation_batches_per_epoch=${WM_VALIDATION_BATCHES_PER_EPOCH}"
echo "[wm-seta-latent] fixed_target_backbone=${WM_FIXED_TARGET_BACKBONE} backprop_to_llm=${WM_BACKPROP_TO_LLM}"
echo "[wm-seta-latent] llm_train_mode=${WM_LLM_TRAIN_MODE} lora_rank=${WM_LORA_RANK}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/slime${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN}" -m slime.world_model.train_latent "${args[@]}"
