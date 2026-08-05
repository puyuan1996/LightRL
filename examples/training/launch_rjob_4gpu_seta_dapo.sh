#!/usr/bin/env bash
# Launch the full 4-GPU Qwen3-8B + SETA + DAPO recipe from inside an rjob.
# It reuses a running remote Docker worker; it does not start Docker locally.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

# The healthy pool server on pu-dev-2. Override when using another worker.
WORKER_URLS="${WORKER_URLS:-http://100.98.75.44:18081}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-seta-dapo-4g-${RUN_TIMESTAMP}}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/launcher.log}"
FOREGROUND="${FOREGROUND:-0}"

die() {
  printf '[rjob-seta-dapo] ERROR: %s\n' "$*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null || die "nvidia-smi is required inside the rjob"
gpu_count="$(nvidia-smi -L | sed -n '/^GPU /p' | wc -l)"
[[ "${gpu_count}" == "4" ]] || die "expected 4 visible GPUs, found ${gpu_count}"

command -v curl >/dev/null || die "curl is required for the worker health check"
curl --noproxy '*' --fail --silent --show-error --max-time 10 \
  "${WORKER_URLS}/healthz" >/dev/null \
  || die "Docker worker is not healthy: ${WORKER_URLS}/healthz"

mkdir -p "${RUN_DIR}"

train_cmd=(
  env
  "RUN_ID=${RUN_ID}"
  "RUN_NAME=${RUN_NAME:-${RUN_ID}}"
  "WORKER_URLS=${WORKER_URLS}"
  "NUM_GPUS=${NUM_GPUS:-4}"
  "ACTOR_GPUS=${ACTOR_GPUS:-2}"
  "ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}"
  "TP_SIZE=${TP_SIZE:-2}"
  "ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
  bash examples/training/train_qwen3_8b_seta_dapo.sh
)

if [[ "${FOREGROUND}" == "1" ]]; then
  printf '[rjob-seta-dapo] starting foreground run_id=%s worker=%s\n' "${RUN_ID}" "${WORKER_URLS}"
  exec "${train_cmd[@]}"
fi

if ! mkdir "${RUN_DIR}/.launch-once" 2>/dev/null; then
  die "${RUN_ID} was already launched; choose another RUN_ID or inspect ${LOG_FILE}"
fi

nohup "${train_cmd[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
pid="$!"
printf '%s\n' "${pid}" >"${RUN_DIR}/launcher.pid"
printf '[rjob-seta-dapo] started run_id=%s pid=%s\n' "${RUN_ID}" "${pid}"
printf '[rjob-seta-dapo] log=%s\n' "${LOG_FILE}"
