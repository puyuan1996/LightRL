#!/usr/bin/env bash
# Submit the evaluation payload to rjob while forwarding explicit model paths.
set -euo pipefail
ROOT="${LIGHTRL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RJOB_CONFIG_FILE="${RJOB_CONFIG_FILE:-${ROOT}/local/rjob/rjob.env}"
if [[ -f "${RJOB_CONFIG_FILE}" ]]; then source "${RJOB_CONFIG_FILE}"; fi
: "${MODEL_PATH:?set MODEL_PATH before submitting}"
: "${MODEL:?set MODEL before submitting}"
: "${RJOB_NAME:?set RJOB_NAME before submitting}"
RJOB_BIN="${RJOB_BIN:-rjob}"
: "${RJOB_NAMESPACE:?set RJOB_NAMESPACE (or RJOB_CONFIG_FILE) before submitting}"
: "${RJOB_GROUP:?set RJOB_GROUP (or RJOB_CONFIG_FILE) before submitting}"
: "${RJOB_IMAGE:?set RJOB_IMAGE (or RJOB_CONFIG_FILE) before submitting}"
: "${RJOB_MOUNTS:?set RJOB_MOUNTS (or RJOB_CONFIG_FILE) before submitting}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_PRIORITY="${RJOB_PRIORITY:-9}"
RJOB_ARGS=(submit --namespace "${RJOB_NAMESPACE}" --group "${RJOB_GROUP}" --name "${RJOB_NAME}"
  --charged-group "${RJOB_GROUP}" --private-machine "${RJOB_PRIVATE_MACHINE}"
  --priority "${RJOB_PRIORITY}" --auto-delete-duration "720h" --image "${RJOB_IMAGE}"
  --share-host-shm True)
RJOB_GPU="${RJOB_GPU:-4}"
RJOB_CPU="${RJOB_CPU:-50}"
RJOB_MEMORY="${RJOB_MEMORY:-560000}"
RJOB_ARGS+=(--gpu "${RJOB_GPU}" --cpu "${RJOB_CPU}" --memory "${RJOB_MEMORY}")
read -r -a _mounts <<< "${RJOB_MOUNTS}"
for _mount in "${_mounts[@]}"; do RJOB_ARGS+=(--mount="${_mount}"); done
if [[ -n "${RJOB_FOLDER:-}" ]]; then RJOB_ARGS+=(--folder "${RJOB_FOLDER}"); fi
EVAL_ENV=(env LIGHTRL_ROOT="${LIGHTRL_ROOT}" MODEL_PATH="${MODEL_PATH}" MODEL="${MODEL}")
for _name in MATH_DATA_ROOT DATASETS OUTPUT_DIR N MAX_TOKENS TEMPERATURE TOP_P CONCURRENCY REWARD_TYPE RUN_DIR TP_SIZE MEM_FRACTION SGLANG_EXTRA_ARGS READY_TIMEOUT TAG; do
  if [[ -n "${!_name:-}" ]]; then EVAL_ENV+=("${_name}=${!_name}"); fi
done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- "${EVAL_ENV[@]}" bash "${ROOT}/local/rjob/run_math_rlvr_eval.sh"
  printf '\n'
  exit 0
fi
exec "${RJOB_BIN}" "${RJOB_ARGS[@]}" -- "${EVAL_ENV[@]}" bash "${ROOT}/local/rjob/run_math_rlvr_eval.sh"
