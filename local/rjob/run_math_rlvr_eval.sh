#!/usr/bin/env bash
# rjob payload: serve one checkpoint and run all configured math eval sets.
set -euo pipefail
ROOT="${LIGHTRL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${MODEL_PATH:?set MODEL_PATH to the checkpoint to serve}"
: "${MODEL:?set MODEL to the served model name}"
PORT="${PORT:-30000}"
SERVER_LOG="${SERVER_LOG:-${RUN_DIR:-${ROOT}/runs/evaluation/math-rjob}/sglang.log}"
mkdir -p "$(dirname -- "${SERVER_LOG}")"
MODEL_PATH="${MODEL_PATH}" MODEL="${MODEL}" PORT="${PORT}" \
  bash "${ROOT}/tools/evaluation/launch_sglang_math.sh" >"${SERVER_LOG}" 2>&1 &
server_pid=$!
cleanup() { kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 "${READY_TIMEOUT:-300}"); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null
exec env MODEL="${MODEL}" PORT="${PORT}" bash "${ROOT}/tools/evaluation/run_math_base_evals.sh"
