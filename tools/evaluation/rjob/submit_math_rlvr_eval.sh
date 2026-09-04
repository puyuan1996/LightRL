#!/usr/bin/env bash
# Submit the evaluation payload to rjob while forwarding explicit model paths.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${MODEL_PATH:?set MODEL_PATH before submitting}"
: "${MODEL:?set MODEL before submitting}"
: "${RJOB_NAME:?set RJOB_NAME before submitting}"
RJOB_BIN="${RJOB_BIN:-rjob}"
RJOB_ARGS=(submit --name "${RJOB_NAME}")
if [[ -n "${RJOB_GPU:-}" ]]; then RJOB_ARGS+=(--gpu "${RJOB_GPU}"); fi
if [[ -n "${RJOB_CPU:-}" ]]; then RJOB_ARGS+=(--cpu "${RJOB_CPU}"); fi
if [[ -n "${RJOB_MEMORY:-}" ]]; then RJOB_ARGS+=(--memory "${RJOB_MEMORY}"); fi
EVAL_ENV=(env MODEL_PATH="${MODEL_PATH}" MODEL="${MODEL}")
for _name in MATH_DATA_ROOT DATASETS OUTPUT_DIR N MAX_TOKENS TEMPERATURE TOP_P CONCURRENCY REWARD_TYPE RUN_DIR; do
  if [[ -n "${!_name:-}" ]]; then EVAL_ENV+=("${_name}=${!_name}"); fi
done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- "${EVAL_ENV[@]}" bash "${ROOT}/tools/evaluation/rjob/run_math_rlvr_eval.sh"
  printf '\n'
  exit 0
fi
exec "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- "${EVAL_ENV[@]}" bash "${ROOT}/tools/evaluation/rjob/run_math_rlvr_eval.sh"
