#!/usr/bin/env bash
# Serve a checkpoint over sglang for the math RLVR eval sweep.
#
# Required:
#   MODEL        HF checkpoint directory, or a served model name
#
# Optional:
#   TP=8            tensor parallel size
#   PORT=30000      HTTP port; eval_math.py talks to 127.0.0.1:$PORT
#   CTX=40960       context length; must exceed the eval --max-tokens (32768)
#   MEM_FRAC=0.85   static VRAM fraction
#   PYTHON=python   interpreter to use; ignored when CUDA_ENV_PREFIX is set
#   MATH_DATA_ROOT  default <repo>/benchmarks/math; only used to place the log
#   LOG             log file path
#   CUDA_ENV_PREFIX when set, its bin/python is used and CUDA_HOME points at it.
#                   Needed on hosts whose only CUDA toolkit lives inside a conda
#                   env: without CUDA_HOME, sglang's CUDA-graph capture fails
#                   with "Capture cuda graph failed: Could not find CUDA
#                   installation." Leave unset when /usr/local/cuda exists.
set -euo pipefail

: "${MODEL:?MODEL is required (HF checkpoint dir or served model name)}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MATH_DATA_ROOT="${MATH_DATA_ROOT:-${REPO}/benchmarks/math}"

TP="${TP:-8}"
PORT="${PORT:-30000}"
CTX="${CTX:-40960}"
MEM_FRAC="${MEM_FRAC:-0.85}"
LOG="${LOG:-${MATH_DATA_ROOT}/logs/sglang_$(date +%Y%m%d_%H%M%S).log}"

PYTHON_BIN="${PYTHON:-python}"
if [[ -n "${CUDA_ENV_PREFIX:-}" ]]; then
  export CUDA_HOME="${CUDA_ENV_PREFIX}"
  export CUDA_PATH="${CUDA_ENV_PREFIX}"
  export PATH="${CUDA_ENV_PREFIX}/bin:${PATH}"
  PYTHON_BIN="${CUDA_ENV_PREFIX}/bin/python"
fi
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "${LOG}")"
echo "[launch] model=${MODEL} tp=${TP} port=${PORT} ctx=${CTX} log=${LOG}"
nohup "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp "${TP}" \
  --port "${PORT}" \
  --context-length "${CTX}" \
  --mem-fraction-static "${MEM_FRAC}" \
  > "${LOG}" 2>&1 &
echo "[launch] pid=$!"
echo "${LOG}"
