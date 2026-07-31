#!/usr/bin/env bash
# One-command end-to-end SETA+DAPO validation for an already-running 4-GPU rjob.
#
# Defaults exercise the complete path with a deliberately small training shape.
# Increase NUM_ROLLOUT only after the default validation succeeds.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/LightRL}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
ALLOW_GPU_COUNT_MISMATCH="${ALLOW_GPU_COUNT_MISMATCH:-0}"
RUN_STATIC_CHECKS="${RUN_STATIC_CHECKS:-1}"
RUN_IMPORT_SMOKE="${RUN_IMPORT_SMOKE:-1}"
RUN_RELEVANT_TESTS="${RUN_RELEVANT_TESTS:-1}"
RUN_CLI_DRY_RUN="${RUN_CLI_DRY_RUN:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
DEFAULT_WORKER_URLS="${DEFAULT_WORKER_URLS:-http://100.96.29.69:18081}"
WORKER_URLS="${WORKER_URLS:-${DEFAULT_WORKER_URLS}}"
WORKER_HEALTH_TIMEOUT="${WORKER_HEALTH_TIMEOUT:-10}"
MIN_REPO_FREE_GB="${MIN_REPO_FREE_GB:-10}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-lightrl-4gpu-seta-dapo-validation-${RUN_TIMESTAMP}}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
VALIDATION_LOG="${RUN_DIR}/validation_driver.log"

# Safe 4-GPU topology and bounded end-to-end training defaults.
NUM_GPUS="${NUM_GPUS:-4}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
TP_SIZE="${TP_SIZE:-2}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
DEBUG_MODE="${DEBUG_MODE:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-2}"
SMOKE_TASK_NAME="${SMOKE_TASK_NAME:-307}"
MAX_TURN="${MAX_TURN:-10}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-8192}"
WANDB_MODE="${WANDB_MODE:-disabled}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"

log() {
  printf '[4gpu-seta-dapo] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

run_stage() {
  local name="$1"
  shift
  log "BEGIN ${name}"
  "$@"
  log "PASS  ${name}"
}

check_gpu_topology() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
  local gpu_count
  gpu_count="$(nvidia-smi -L | sed -n '/^GPU /p' | wc -l)"
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  if [[ "${gpu_count}" != "${EXPECTED_GPUS}" \
    && "${ALLOW_GPU_COUNT_MISMATCH}" != "1" ]]; then
    die "expected ${EXPECTED_GPUS} visible GPUs, found ${gpu_count}"
  fi
}

check_resources() {
  free -h
  df -h / "${REPO_ROOT}"
  local available_kb
  available_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
  if [[ -n "${available_kb}" ]] \
    && (( available_kb < MIN_REPO_FREE_GB * 1024 * 1024 )); then
    die "repo filesystem has less than ${MIN_REPO_FREE_GB} GiB free"
  fi
}

check_worker() {
  command -v curl >/dev/null 2>&1 || die "curl is required"
  local first_worker="${WORKER_URLS%%,*}"
  first_worker="${first_worker%% *}"
  [[ -n "${first_worker}" ]] || die "WORKER_URLS is empty"
  log "Checking worker health at ${first_worker}/healthz"
  curl --noproxy '*' --silent --show-error --fail \
    --max-time "${WORKER_HEALTH_TIMEOUT}" \
    "${first_worker%/}/healthz"
  printf '\n'
}

static_checks() {
  python3 -m compileall -q agentic_rl tests/agentic_rl tools
  git diff --check
  local legacy_module_pattern='agentic_rl\.rollout\.'"trajectory"'\.(policy|store)'
  local legacy_dir_pattern='agentic_rl/rollout/'"trajectory/"
  if rg -n "${legacy_module_pattern}|${legacy_dir_pattern}" \
    agentic_rl tests tools configs docs examples deploy; then
    die "legacy trajectory module path remains"
  fi
}

import_smoke() {
  python3 - <<'PY'
from agentic_rl import REGISTRY, load_config
from agentic_rl.rollout import trajectory_store

assert callable(trajectory_store._trajectory_save_decision)
assert callable(trajectory_store._save_rollout_artifacts)
assert "dapo" in REGISTRY.names("algorithms")
config = load_config("configs/experiment/qwen3_8b_seta_dapo.yaml")
assert config["algorithm"]["base"]["name"] == "dapo"
assert config["environment"]["name"] == "seta"
print("RJOB_IMPORT_SMOKE_OK")
PY
}

relevant_tests() {
  python3 -m pytest -q \
    tests/agentic_rl/test_rollout_log_metrics.py \
    tests/agentic_rl/test_harness_option_routing.py \
    tests/agentic_rl/test_agent_runner_harness_option.py
}

cli_dry_run() {
  NUM_GPUS="${NUM_GPUS}" \
  ACTOR_GPUS="${ACTOR_GPUS}" \
  ROLLOUT_GPUS="${ROLLOUT_GPUS}" \
  TP_SIZE="${TP_SIZE}" \
  ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
    bash examples/train_qwen3_8b_seta_dapo.sh --dry-run
}

run_training() {
  export REPO_ROOT RUN_TIMESTAMP RUN_ID WORKER_URLS
  export NUM_GPUS ACTOR_GPUS ROLLOUT_GPUS TP_SIZE
  export ROLLOUT_NUM_GPUS_PER_ENGINE DEBUG_MODE
  export NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES SMOKE_TASK_NAME
  export MAX_TURN ROLLOUT_MAX_RESPONSE_LEN ROLLOUT_MAX_CONTEXT_LEN
  export WANDB_MODE MAX_CKPT_KEEP
  export DATASET=seta
  export ALGO=dapo
  export DAPO_DYNAMIC_SAMPLING=0
  export EXPLORATION_PROFILE=off

  log "Starting real SETA+DAPO validation: task=${SMOKE_TASK_NAME} rollouts=${NUM_ROLLOUT} batch=${ROLLOUT_BATCH_SIZE} samples=${N_SAMPLES}"
  bash tools/rjob/run_seta_dapo_refactor_smoke.sh
}

post_training_checks() {
  local metrics_path="${RUN_DIR}/logs/metrics.jsonl"
  if rg -n \
    'ModuleNotFoundError|NameError:.*trajectory|CUDA out of memory|RayActorError' \
    "${RUN_DIR}"; then
    die "fatal error signature found under ${RUN_DIR}"
  fi
  [[ -s "${metrics_path}" ]] \
    || die "training exited without a non-empty ${metrics_path}"
  python3 - "${metrics_path}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not records:
    raise SystemExit("metrics file contains no JSON records")

finite_numeric = 0
for record in records:
    for value in record.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                finite_numeric += 1
if finite_numeric == 0:
    raise SystemExit("metrics records contain no finite numeric values")

print(
    "TRAINING_METRICS_OK",
    f"records={len(records)}",
    f"finite_numeric_values={finite_numeric}",
)
print("LAST_METRICS_RECORD", json.dumps(records[-1], sort_keys=True))
PY
}

main() {
  [[ -d "${REPO_ROOT}/agentic_rl" ]] || die "LightRL repo not found: ${REPO_ROOT}"
  cd "${REPO_ROOT}"
  mkdir -p "${RUN_DIR}"
  exec > >(tee -a "${VALIDATION_LOG}") 2>&1

  log "host=$(hostname) commit=$(git rev-parse --short HEAD) run_id=${RUN_ID}"
  run_stage "GPU topology" check_gpu_topology
  run_stage "resource guard" check_resources

  if [[ "${RUN_STATIC_CHECKS}" == "1" ]]; then
    run_stage "static checks" static_checks
  fi
  if [[ "${RUN_IMPORT_SMOKE}" == "1" ]]; then
    run_stage "import smoke" import_smoke
  fi
  if [[ "${RUN_RELEVANT_TESTS}" == "1" ]]; then
    run_stage "relevant pytest" relevant_tests
  fi
  if [[ "${RUN_CLI_DRY_RUN}" == "1" ]]; then
    run_stage "CLI dry-run" cli_dry_run
  fi
  if [[ "${RUN_TRAINING}" == "1" ]]; then
    run_stage "worker health" check_worker
    run_stage "SETA+DAPO training" run_training
    run_stage "training artifacts" post_training_checks
  fi

  log "All enabled validation stages passed"
  if [[ "${RUN_TRAINING}" == "0" ]]; then
    log "RUN_TRAINING=0, so no real SETA+DAPO rollout was started"
  fi
}

main "$@"
