#!/usr/bin/env bash
# Bounded end-to-end smoke run for the public DIVE-PO and mixed examples.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/LightRL}"
EXPERIMENT="${EXPERIMENT:-dive_po}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-lightrl-4gpu-${EXPERIMENT}-smoke-${RUN_TIMESTAMP}}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
SMOKE_TASK_NAME="${SMOKE_TASK_NAME:-307}"
PROMPT_DATA="${RUN_DIR}/config/${EXPERIMENT}_smoke.jsonl"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${RUN_DIR}/config/worker_urls.txt}"

case "${EXPERIMENT}" in
  dive_po)
    EXAMPLE_SCRIPT="examples/train_qwen3_8b_seta_dive_po.sh"
    DEFAULT_NUM_ROLLOUT=3
    DEFAULT_ROLLOUT_BATCH_SIZE=1
    ;;
  mixed)
    EXAMPLE_SCRIPT="examples/train_qwen3_8b_mixed_dapo.sh"
    DEFAULT_NUM_ROLLOUT=2
    DEFAULT_ROLLOUT_BATCH_SIZE=3
    ;;
  *)
    echo "[rjob] ERROR: EXPERIMENT must be dive_po or mixed, got ${EXPERIMENT}" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}/config"
exec > >(tee -a "${RUN_DIR}/rjob_outer.log") 2>&1

echo "[rjob] host=$(hostname) commit=$(git rev-parse --short HEAD) experiment=${EXPERIMENT} run_id=${RUN_ID}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
free -h
df -h / "${REPO_ROOT}"

python3 - "${EXPERIMENT}" "${SMOKE_TASK_NAME}" "${PROMPT_DATA}" <<'PY'
import json
import sys
from pathlib import Path

experiment, task_name, output_raw = sys.argv[1:]
output = Path(output_raw)
repo = Path.cwd()

def read_first(path: Path, predicate=lambda _record: True):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if predicate(record):
                return record
    raise SystemExit(f"no matching smoke record in {path}")

seta_path = repo / "benchmarks/seta_env_convert/train.jsonl"
seta = read_first(
    seta_path,
    lambda record: str(record.get("metadata", {}).get("task_name")) == task_name,
)
records = [seta]
if experiment == "mixed":
    records.extend(
        [
            read_first(repo / "benchmarks/agent_safetybench_convert/train.jsonl"),
            read_first(repo / "benchmarks/agentharm_convert/train.jsonl"),
        ]
    )
with output.open("w", encoding="utf-8") as stream:
    for record in records:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
print(
    "[rjob] smoke data="
    + str(output)
    + " sources="
    + ",".join(str(record["metadata"]["data_source"]) for record in records)
)
PY

export RUN_ID
export RUN_NAME="${RUN_NAME:-${RUN_ID}}"
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export DEBUG_MODE="${DEBUG_MODE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-${DEFAULT_NUM_ROLLOUT}}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${DEFAULT_ROLLOUT_BATCH_SIZE}}"
export N_SAMPLES="${N_SAMPLES:-2}"
export MAX_TURN="${MAX_TURN:-10}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-8192}"
export ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${PROMPT_DATA}}"
export WORKER_URLS="${WORKER_URLS:-http://100.96.26.133:18081}"
export WORKER_URLS_FILE
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_ENABLE="${WANDB_ENABLE:-0}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"
export SAVE_CKPT="${SAVE_CKPT:-}"
export RESUME_LOAD="${RESUME_LOAD:-}"
export TERMINAL_RL_GENERATE_FAILURE_TRACEBACK="${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK:-1}"
export ENV_RESET_MAX_RETRIES="${ENV_RESET_MAX_RETRIES:-2}"
export ROLLOUT_GENERATION_MAX_RETRIES="${ROLLOUT_GENERATION_MAX_RETRIES:-2}"
export ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF="${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF:-20}"
export ROLLOUT_GENERATION_RETRY_MAX_BACKOFF="${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF:-60}"

echo "[rjob] launch=${EXAMPLE_SCRIPT} rollouts=${NUM_ROLLOUT} batch=${ROLLOUT_BATCH_SIZE} samples=${N_SAMPLES}"
exec bash "${EXAMPLE_SCRIPT}"
