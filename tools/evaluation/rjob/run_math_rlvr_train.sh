#!/usr/bin/env bash
# Submit the math recipe to rjob.  The command is configurable because rjob
# installations differ in queue/resource flags; this script never invents a
# checkpoint or dataset path.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${HF_CKPT:?set HF_CKPT before submitting}"
: "${REF_LOAD:?set REF_LOAD before submitting}"
: "${RJOB_NAME:?set RJOB_NAME before submitting}"
RJOB_BIN="${RJOB_BIN:-rjob}"
RJOB_ARGS=(submit --name "${RJOB_NAME}")
if [[ -n "${RJOB_GPU:-}" ]]; then RJOB_ARGS+=(--gpu "${RJOB_GPU}"); fi
if [[ -n "${RJOB_CPU:-}" ]]; then RJOB_ARGS+=(--cpu "${RJOB_CPU}"); fi
if [[ -n "${RJOB_MEMORY:-}" ]]; then RJOB_ARGS+=(--memory "${RJOB_MEMORY}"); fi
TRAIN_ENV=(env HF_CKPT="${HF_CKPT}" REF_LOAD="${REF_LOAD}")
for _name in MATH_DATA_ROOT TRAIN_DATA TRAIN_DATASET REWARD_TYPE RESPONSE_CAP ROLLOUT_BATCH_SIZE N_SAMPLES NUM_ROLLOUT GLOBAL_BATCH_SIZE EVAL_DATASETS EVAL_N_SAMPLES EVAL_INTERVAL EVAL_TOP_P SEED RUN_ID RUN_DIR; do
  if [[ -n "${!_name:-}" ]]; then TRAIN_ENV+=("${_name}=${!_name}"); fi
done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- "${TRAIN_ENV[@]}" bash "${ROOT}/examples/training/train_qwen3_8b_dapo_math.sh"
  printf '\n'
  exit 0
fi
exec "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- "${TRAIN_ENV[@]}" bash "${ROOT}/examples/training/train_qwen3_8b_dapo_math.sh"
