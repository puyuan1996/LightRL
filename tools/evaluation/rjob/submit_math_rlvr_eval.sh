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
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- env MODEL_PATH="${MODEL_PATH}" MODEL="${MODEL}" bash "${ROOT}/tools/evaluation/rjob/run_math_rlvr_eval.sh"
  printf '\n'
  exit 0
fi
exec "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- env MODEL_PATH="${MODEL_PATH}" MODEL="${MODEL}" bash "${ROOT}/tools/evaluation/rjob/run_math_rlvr_eval.sh"
