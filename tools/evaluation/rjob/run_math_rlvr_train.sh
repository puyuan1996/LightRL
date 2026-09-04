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
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- bash "${ROOT}/examples/training/train_qwen3_8b_dapo_math.sh"
  printf '\n'
  exit 0
fi
exec "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- bash "${ROOT}/examples/training/train_qwen3_8b_dapo_math.sh"
