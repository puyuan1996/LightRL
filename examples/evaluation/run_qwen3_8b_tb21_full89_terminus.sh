#!/usr/bin/env bash
# One-click Terminal-Bench 2.1 full-89 evaluation:
# Qwen3-8B + Harbor terminus-2 on a four-GPU SGLang server.
#
# This is a pure evaluation path.  It does not start a Slime actor or Ray
# training job, so all four GPUs are available to the single rollout server.
# Paths and the Harbor/SGLang environment remain overridable for different
# workers or pre-provisioned TBv2.1 bundles.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/evaluation/run_qwen3_8b_tb21_full89_terminus.sh
  bash examples/evaluation/run_qwen3_8b_tb21_full89_terminus.sh --dry-run

Important overrides:
  TB21_TASKS_DIR      Terminal-Bench 2.1 tasks directory (must contain 89 task dirs)
  HF_MODEL_PATH       Qwen3-8B HuggingFace checkpoint directory
  HARBOR_BIN          Harbor CLI (or TBV21_HOME/bin/harbor)
  EVAL_PYTHON         Python with tools.evaluation and SGLang dependencies
  WORKER_URLS         optional external SETA worker; not used by terminus-2
  MODEL_API_BASE      use an existing OpenAI-compatible endpoint instead of managed SGLang
  GPU_IDS             managed SGLang GPU IDs (default 0,1,2,3)
  SGLANG_PORT         managed SGLang port (default 30000)
  TB21_CONCURRENCY    Harbor concurrency (default 2; increase after storage validation)
  TB21_OUTPUT_DIR     output directory (default runs/evaluation/<job>)

The default managed mode starts SGLang on all four GPUs and then runs all 89
Terminal-Bench 2.1 tasks with the terminus-2 Harbor agent.
EOF
}

DRY_RUN="${DRY_RUN:-0}"
case "${1:-}" in
  --dry-run) DRY_RUN=1; shift ;;
  --help|-h) usage; exit 0 ;;
  "") ;;
  *) echo "[tb21-full89] ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac
[[ "$#" -eq 0 ]] || { echo "[tb21-full89] ERROR: unexpected arguments" >&2; exit 2; }

die() { echo "[tb21-full89] ERROR: $*" >&2; exit 1; }
warn() { echo "[tb21-full89] WARN: $*" >&2; }

first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

TB21_TASKS_DIR="${TB21_TASKS_DIR:-}"
if [[ -z "${TB21_TASKS_DIR}" ]]; then
  TB21_TASKS_DIR="$(first_existing_dir \
    "/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1/datasets/terminal-bench-2-1/tasks" \
    "/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/tbv2.1/datasets/terminal-bench-2-1/tasks" \
    "${REPO_ROOT}/benchmarks/datasets/terminal-bench-2-1/tasks" || true)"
fi
[[ -n "${TB21_TASKS_DIR}" && -d "${TB21_TASKS_DIR}" ]] || die \
  "TB21_TASKS_DIR is missing; set it to the Terminal-Bench 2.1 tasks directory"

task_count="$(find "${TB21_TASKS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '.' 2>/dev/null | wc -c | tr -d ' ')"
if [[ "${task_count}" != "89" && "${ALLOW_TASK_COUNT_MISMATCH:-0}" != "1" ]]; then
  die "expected 89 task directories under ${TB21_TASKS_DIR}, found ${task_count}; set ALLOW_TASK_COUNT_MISMATCH=1 only for a deliberate subset"
fi

HF_MODEL_PATH="${HF_MODEL_PATH:-}"
if [[ -z "${HF_MODEL_PATH}" ]]; then
  HF_MODEL_PATH="$(first_existing_dir \
    "${REPO_ROOT}/../slime/Qwen3-8B" \
    "${REPO_ROOT}/models/Qwen3-8B" \
    "/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B" || true)"
fi
[[ -n "${HF_MODEL_PATH}" && -r "${HF_MODEL_PATH}/config.json" ]] || die \
  "HF_MODEL_PATH is missing or has no readable config.json; set it to Qwen3-8B"

TBV21_HOME="${TBV21_HOME:-/mnt/shared-storage-user/narmodel/zhangshaoang/tbv2.1}"
HARBOR_BIN="${HARBOR_BIN:-}"
if [[ -z "${HARBOR_BIN}" && -x "${TBV21_HOME}/bin/harbor" ]]; then
  HARBOR_BIN="${TBV21_HOME}/bin/harbor"
fi
if [[ -z "${HARBOR_BIN}" ]]; then
  HARBOR_BIN="$(command -v harbor 2>/dev/null || true)"
fi
[[ -n "${HARBOR_BIN}" && -x "${HARBOR_BIN}" ]] || die \
  "Harbor CLI not found; set HARBOR_BIN or TBV21_HOME"

EVAL_PYTHON="${EVAL_PYTHON:-}"
if [[ -z "${EVAL_PYTHON}" ]]; then
  for candidate in \
    "${LIGHTRFT_PY312_BIN:+${LIGHTRFT_PY312_BIN}/python3}" \
    "/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python" \
    "/mnt/shared-storage-user/puyuan/.cache/lightrl-seta-worker-py312/bin/python" \
    "$(command -v python3 || true)"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    if PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${candidate}" -c \
      'import yaml, tools.evaluation' >/dev/null 2>&1; then
      EVAL_PYTHON="${candidate}"
      break
    fi
  done
fi
[[ -n "${EVAL_PYTHON}" && -x "${EVAL_PYTHON}" ]] || die \
  "no compatible EVAL_PYTHON found; need Python with PyYAML and tools.evaluation"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${EVAL_PYTHON}" -c \
  'import yaml, tools.evaluation' >/dev/null 2>&1 || die \
  "EVAL_PYTHON cannot import yaml/tools.evaluation: ${EVAL_PYTHON}"

MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
SGLANG_MODE="${SGLANG_MODE:-managed}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.70}"
MODEL_API_BASE="${MODEL_API_BASE:-http://127.0.0.1:${SGLANG_PORT}/v1}"
TB21_CONCURRENCY="${TB21_CONCURRENCY:-2}"
TB21_MAX_RETRIES="${TB21_MAX_RETRIES:-1}"
TB21_TIMEOUT_MULTIPLIER="${TB21_TIMEOUT_MULTIPLIER:-1.0}"
TB21_MAX_INPUT_TOKENS="${TB21_MAX_INPUT_TOKENS:-32768}"
TB21_MAX_OUTPUT_TOKENS="${TB21_MAX_OUTPUT_TOKENS:-8192}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
JOB_NAME="${JOB_NAME:-qwen3-8b-tb21-full89-terminus-${RUN_TIMESTAMP}}"
OUTPUT_DIR="${TB21_OUTPUT_DIR:-${REPO_ROOT}/runs/evaluation/${JOB_NAME}}"
CONFIG_PATH="${OUTPUT_DIR}/${JOB_NAME}.yaml"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname -- "${EVAL_PYTHON}"):${PATH}"
# The config writer below runs in a child Python process.  Export every value
# it consumes before invoking it (shell locals are not inherited by children).
export CONFIG_PATH OUTPUT_DIR JOB_NAME TB21_TASKS_DIR HF_MODEL_PATH HARBOR_BIN
export GPU_IDS MODEL_NAME SGLANG_MODE SGLANG_PORT TP_SIZE SGLANG_MEM_FRACTION MODEL_API_BASE
export TB21_CONCURRENCY TB21_MAX_RETRIES TB21_TIMEOUT_MULTIPLIER TB21_MAX_INPUT_TOKENS TB21_MAX_OUTPUT_TOKENS

"${EVAL_PYTHON}" - "${CONFIG_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
gpu_ids = [int(x) for x in os.environ["GPU_IDS"].split(",") if x.strip()]
proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy")
task_env = {
    "DEBIAN_FRONTEND": "noninteractive",
    "TZ": "Etc/UTC",
    "UV_LINK_MODE": "copy",
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy"),
    "OPENAI_BASE_URL": os.environ["MODEL_API_BASE"],
}
for key in proxy_keys:
    if os.environ.get(key):
        task_env[key] = os.environ[key]

config = {
    "harness": "terminus-2",
    "job_name": os.environ["JOB_NAME"],
    "output_dir": os.environ["OUTPUT_DIR"],
    "dataset": {"path": os.environ["TB21_TASKS_DIR"], "task_names": None},
    "run": {
        "n_attempts": 1,
        "concurrency": int(os.environ["TB21_CONCURRENCY"]),
        "max_retries": int(os.environ["TB21_MAX_RETRIES"]),
        "timeout_multiplier": float(os.environ["TB21_TIMEOUT_MULTIPLIER"]),
        "max_input_tokens": int(os.environ["TB21_MAX_INPUT_TOKENS"]),
        "max_output_tokens": int(os.environ["TB21_MAX_OUTPUT_TOKENS"]),
    },
    "serving": {
        "mode": os.environ["SGLANG_MODE"],
        "model_path": os.environ["HF_MODEL_PATH"],
        "model_name": os.environ["MODEL_NAME"],
        "port": int(os.environ["SGLANG_PORT"]),
        "gpu_ids": gpu_ids,
        "tp_size": int(os.environ["TP_SIZE"]),
        "mem_fraction": float(os.environ["SGLANG_MEM_FRACTION"]),
        "api_base": os.environ["MODEL_API_BASE"],
        "health_timeout_s": 900,
    },
    "environment": task_env,
    "extra": {
        "harbor_bin": os.environ["HARBOR_BIN"],
        "openai_api_key": task_env["OPENAI_API_KEY"],
        "no_proxy": os.environ.get("NO_PROXY", "localhost,127.0.0.1,::1"),
        "process_env": {key: value for key, value in task_env.items() if key in proxy_keys},
    },
}
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

echo "[tb21-full89] job=${JOB_NAME}"
echo "[tb21-full89] tasks=${TB21_TASKS_DIR} count=${task_count}"
echo "[tb21-full89] model=${HF_MODEL_PATH} name=${MODEL_NAME}"
echo "[tb21-full89] harness=terminus-2 concurrency=${TB21_CONCURRENCY} retries=${TB21_MAX_RETRIES}"
echo "[tb21-full89] output=${OUTPUT_DIR}"
echo "[tb21-full89] config=${CONFIG_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${EVAL_PYTHON}" -m tools.evaluation run --config "${CONFIG_PATH}" --dry-run \
    --poll-interval "${POLL_INTERVAL}"
  exit 0
fi

"${EVAL_PYTHON}" -m tools.evaluation run --config "${CONFIG_PATH}" \
  --poll-interval "${POLL_INTERVAL}"
echo "[tb21-full89] completed; result=${OUTPUT_DIR}/eval_result.json"
