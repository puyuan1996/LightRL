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
DEFAULT_WORKER_URLS="${DEFAULT_WORKER_URLS:-http://100.96.26.133:18081}"
WORKER_URLS="${WORKER_URLS:-${DEFAULT_WORKER_URLS}}"
WORKER_HEALTH_TIMEOUT="${WORKER_HEALTH_TIMEOUT:-10}"
MIN_REPO_FREE_GB="${MIN_REPO_FREE_GB:-10}"
LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin}"

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
  local search_pattern="${legacy_module_pattern}|${legacy_dir_pattern}"
  local search_roots=(agentic_rl tests tools configs docs examples deploy)
  if command -v rg >/dev/null 2>&1; then
    if rg -n "${search_pattern}" "${search_roots[@]}"; then
      die "legacy trajectory module path remains"
    fi
  elif grep -R -I -n -E "${search_pattern}" "${search_roots[@]}"; then
    die "legacy trajectory module path remains"
  fi
}

import_smoke() {
  python3 - <<'PY'
from agentic_rl.algorithms import ALGORITHMS
from agentic_rl.rollout import entrypoint, trajectory_store

assert callable(trajectory_store._trajectory_save_decision)
assert callable(trajectory_store._save_rollout_artifacts)
for name in (
    "_EXPLORE_CDE_ACTOR_ENABLED",
    "_EXPLORE_CDE_ACTOR_REWARD_GATE",
    "_EXPLORE_INTRINSIC_COEF",
    "_EXPLORE_INTRINSIC_DECAY_STEPS",
    "_EXPLORE_INTRINSIC_ENABLED",
    "_EXPLORE_INTRINSIC_GRANULARITY",
    "_EXPLORE_INTRINSIC_REDUCER",
    "_EXPLORE_INTRINSIC_SCHEDULE",
    "_EXPLORE_INTRINSIC_SCOPE",
    "_EXPLORE_LPRND_COEF",
    "_EXPLORE_LPRND_DECAY_STEPS",
    "_EXPLORE_LPRND_ENABLED",
    "_EXPLORE_LPRND_SCHEDULE",
    "_EXPLORE_SAFETY_FILTER_ENABLED",
    "_EXPLORE_SCORE_BONUS_COMPONENTS",
):
    assert hasattr(entrypoint, name), f"rollout entrypoint is missing {name}"
assert ALGORITHMS == ("dive_po",)
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
    bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
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
  bash examples/validation/internal/run_seta_dapo_smoke.sh
}

post_training_checks() {
  local metrics_path="${RUN_DIR}/logs/metrics.jsonl"
  local train_log_path="${RUN_DIR}/logs/train.log"
  local fatal_pattern='ModuleNotFoundError|NameError:|CUDA out of memory|RayActorError'
  local fatal_found=1
  if command -v rg >/dev/null 2>&1; then
    rg -n "${fatal_pattern}" "${RUN_DIR}" && fatal_found=0
  else
    grep -R -I -n -E "${fatal_pattern}" "${RUN_DIR}" && fatal_found=0
  fi
  if (( fatal_found == 0 )); then
    die "fatal error signature found under ${RUN_DIR}"
  fi
  [[ -s "${metrics_path}" ]] \
    || die "training exited without a non-empty ${metrics_path}"
  [[ -s "${train_log_path}" ]] \
    || die "training exited without a non-empty ${train_log_path}"
  python3 - "${metrics_path}" "${train_log_path}" "${NUM_ROLLOUT}" <<'PY'
import ast
import json
import math
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
train_log_path = Path(sys.argv[2])
records = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not records:
    raise SystemExit("metrics file contains no JSON records")
expected_rollouts = int(sys.argv[3])
if len(records) < expected_rollouts:
    raise SystemExit(
        f"expected at least {expected_rollouts} metric records, found {len(records)}"
    )

finite_numeric = 0
for record in records:
    for value in record.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                finite_numeric += 1
if finite_numeric == 0:
    raise SystemExit("metrics records contain no finite numeric values")
max_trainable = max(int(record.get("trainable_count") or 0) for record in records)
if max_trainable <= 0:
    raise SystemExit("all metric records have trainable_count=0")

train_records = []
for match in re.finditer(
    r"train-step\s+\d+:\s+(\{[^\n]+\})",
    train_log_path.read_text(encoding="utf-8", errors="replace"),
):
    try:
        train_records.append(ast.literal_eval(match.group(1)))
    except (SyntaxError, ValueError):
        pass
if len(train_records) < expected_rollouts:
    raise SystemExit(
        f"expected at least {expected_rollouts} actor train-step records, "
        f"found {len(train_records)}"
    )
nonzero_updates = [
    record
    for record in train_records
    if math.isfinite(float(record.get("train/loss", float("nan"))))
    and math.isfinite(float(record.get("train/grad_norm", float("nan"))))
    and abs(float(record["train/loss"])) > 0.0
    and abs(float(record["train/grad_norm"])) > 0.0
]
if not nonzero_updates:
    raise SystemExit("actor train steps contain no finite non-zero loss/grad update")

print(
    "TRAINING_METRICS_OK",
    f"records={len(records)}",
    f"finite_numeric_values={finite_numeric}",
    f"max_trainable_count={max_trainable}",
    f"actor_train_steps={len(train_records)}",
    f"nonzero_actor_updates={len(nonzero_updates)}",
)
print("LAST_METRICS_RECORD", json.dumps(records[-1], sort_keys=True))
PY
}

main() {
  [[ -d "${REPO_ROOT}/agentic_rl" ]] || die "LightRL repo not found: ${REPO_ROOT}"
  cd "${REPO_ROOT}"
  if [[ -x "${LIGHTRFT_PY312_BIN}/python3" ]]; then
    export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"
  else
    die "training Python environment is unavailable: ${LIGHTRFT_PY312_BIN}"
  fi
  export PYTHONPATH="${REPO_ROOT}/slime:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
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
