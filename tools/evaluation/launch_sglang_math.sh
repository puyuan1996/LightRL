#!/usr/bin/env bash
# Start an OpenAI-compatible SGLang server for Math RLVR evaluation.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?set MODEL_PATH to the checkpoint to serve (no implicit model)}"
SERVED_NAME="${SERVED_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
GPU_IDS="${GPU_IDS:-}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"

cmd=("${SGLANG_PYTHON}" -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_NAME}"
  --host "${HOST}" --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --mem-fraction-static "${MEM_FRACTION}"
  --trust-remote-code)
if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi
if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
  read -r -a _extra_args <<< "${SGLANG_EXTRA_ARGS}"
  cmd+=("${_extra_args[@]}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[dry-run] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi
exec "${cmd[@]}"
