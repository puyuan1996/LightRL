#!/usr/bin/env bash
# Small, reproducible SETA+DAPO validation run for a single 4-GPU rjob.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/LightRL}"
cd "${REPO_ROOT}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-lightrl-refactor-seta-dapo-4g-${RUN_TIMESTAMP}}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
OUTER_LOG="${RUN_DIR}/rjob_outer.log"
SMOKE_TASK_NAME="${SMOKE_TASK_NAME:-307}"
SMOKE_DATA="${RUN_DIR}/config/seta_smoke_${SMOKE_TASK_NAME}.jsonl"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${RUN_DIR}/config/worker_urls.txt}"

mkdir -p "${RUN_DIR}/config"
exec > >(tee -a "${OUTER_LOG}") 2>&1

echo "[rjob] host=$(hostname) commit=$(git rev-parse --short HEAD) run_id=${RUN_ID}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
free -h
df -h / "${REPO_ROOT}"

if docker info >/dev/null 2>&1; then
  echo "[rjob] Docker server ready"
else
  # SETA task containers are owned by WORKER_URLS. The trainer itself does not
  # need privileged Docker access; local Docker availability is diagnostic.
  echo "[rjob] Docker unavailable in trainer; using remote SETA worker"
fi

python3 - "${REPO_ROOT}/benchmarks/seta_env_convert/train.jsonl" \
  "${SMOKE_DATA}" "${SMOKE_TASK_NAME}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
task_name = sys.argv[3]
matched = []
with source.open(encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        if str(record.get("metadata", {}).get("task_name")) == task_name:
            matched.append(line)
if len(matched) != 1:
    raise SystemExit(f"expected one SETA task {task_name}, found {len(matched)}")
output.write_text(matched[0], encoding="utf-8")
print(f"[rjob] smoke task={task_name} data={output}")
PY

export RUN_ID
export RUN_NAME="${RUN_NAME:-${RUN_ID}}"
export DATASET="${DATASET:-seta}"
export ALGO="${ALGO:-dapo}"
export EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-off}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export DEBUG_MODE="${DEBUG_MODE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES="${N_SAMPLES:-2}"
export MAX_TURN="${MAX_TURN:-10}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-8192}"
export ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SMOKE_DATA}}"
export WORKER_URLS="${WORKER_URLS:-http://100.96.26.133:18081}"
export WORKER_URLS_FILE
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"
export SAVE_CKPT="${SAVE_CKPT:-}"
export RESUME_LOAD="${RESUME_LOAD:-}"
export TERMINAL_RL_GENERATE_FAILURE_TRACEBACK="${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK:-1}"
export ENV_RESET_MAX_RETRIES="${ENV_RESET_MAX_RETRIES:-2}"
export ROLLOUT_GENERATION_MAX_RETRIES="${ROLLOUT_GENERATION_MAX_RETRIES:-2}"
export ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF="${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF:-20}"
export ROLLOUT_GENERATION_RETRY_MAX_BACKOFF="${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF:-60}"

exec python3 -m agentic_rl.cli train \
  --config configs/experiment/qwen3_8b_seta_dapo.yaml
