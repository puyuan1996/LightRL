#!/usr/bin/env bash
# One-click 4-GPU evaluation: Qwen3-8B + SETA fixed12 + camel-agent.
#
# This is an evaluation-only invocation of the existing Slime launcher.  The
# launcher starts a local Ray runtime and therefore may stop local Ray/SGLang
# processes; require an explicit acknowledgement before doing that.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  CONFIRM_LOCAL_CLEANUP=1 bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh
  bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh --dry-run

The real run expects a reachable SETA worker at WORKER_URLS (default:
http://127.0.0.1:18081).  Override model/checkpoint, worker, or run settings
through environment variables; see examples/evaluation/README.md.
EOF
}

DRY_RUN="${DRY_RUN:-0}"
BACKGROUND="${BACKGROUND:-1}"
case "${1:-}" in
  --dry-run)
    DRY_RUN=1
    BACKGROUND=0
    shift
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  "") ;;
  *)
    printf '[seta-fixed12] ERROR: unknown argument: %s\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac
[[ "$#" -eq 0 ]] || { printf '[seta-fixed12] ERROR: unexpected arguments\n' >&2; exit 2; }

die() { printf '[seta-fixed12] ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '[seta-fixed12] WARN: %s\n' "$*" >&2; }

# Fixed protocol and 4-GPU topology.  Every value remains overridable so the
# recipe can also be reused on a private worker or a different model mount.
export REPO_ROOT
export WORKER_URLS="${WORKER_URLS:-http://127.0.0.1:18081}"
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/benchmarks/environments}"
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

export MODEL_TAG="${MODEL_TAG:-qwen3-8b}"
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-qwen3-8B}"
export HF_CKPT="${HF_CKPT:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B}"
export REF_LOAD="${REF_LOAD:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B_torch_dist}"
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
export SLIME_ENTRYPOINT="${SLIME_ENTRYPOINT:-${REPO_ROOT}/slime/eval_only.py}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/configs/rollout/rollout_qwen3_think.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/evaluation/seta_fixed12_score_v1.yaml}"

export DATASET="seta"
export ALGO="dapo"
export HARNESS_OPTION="camel-agent"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-0}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
export EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-off}"
export MAX_TURN="${MAX_TURN:-10}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-0}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES="${N_SAMPLES:-1}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-8192}"
export EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-16384}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
export EVAL_TOP_P="${EVAL_TOP_P:-1}"
export EVAL_TOP_K="${EVAL_TOP_K:--1}"
export EVAL_SEED="${EVAL_SEED:-20260809}"
export EVAL_STEPS="${EVAL_STEPS:-0}"

# Evaluation does not need checkpoints or online W&B.  Trajectories remain
# enabled so every one of the 12 fixed tasks can be inspected afterwards.
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"
export TRAJECTORY_SAVE_INTERVAL_SETA="${TRAJECTORY_SAVE_INTERVAL_SETA:-1}"
export TRAJECTORY_SAVE_POLICY="${TRAJECTORY_SAVE_POLICY:-step_interval}"
export EVAL_ROLLOUT_MAX_CONCURRENCY="${EVAL_ROLLOUT_MAX_CONCURRENCY:-3}"
export WANDB_ENABLE="${WANDB_ENABLE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
export RUN_ID="${RUN_ID:-qwen3-8b-seta-fixed12-camel-4gpu-${RUN_TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-${RUN_ID}}"
export BACKGROUND
export DRY_RUN

[[ -f "${SLIME_ENTRYPOINT}" ]] || die "Slime eval entrypoint not found: ${SLIME_ENTRYPOINT}"
[[ -f "${CUSTOM_CONFIG_PATH}" ]] || die "rollout config not found: ${CUSTOM_CONFIG_PATH}"
[[ -f "${EVAL_CONFIG}" ]] || die "SETA eval config not found: ${EVAL_CONFIG}"
[[ -f "${REPO_ROOT}/benchmarks/datasets/seta_env_convert/eval_fixed12.jsonl" ]] \
  || die "SETA fixed12 dataset not found under benchmarks/datasets/seta_env_convert"
[[ -d "${DATASET_DIR}/seta_env" ]] || die "SETA environment directory not found: ${DATASET_DIR}/seta_env"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
launcher_python="python3"
if [[ -n "${LIGHTRFT_PY312_BIN:-}" && -x "${LIGHTRFT_PY312_BIN}/python3" ]]; then
  launcher_python="${LIGHTRFT_PY312_BIN}/python3"
fi
"${launcher_python}" -c 'import yaml' >/dev/null 2>&1 || die \
  "PyYAML is required by the Slime launcher; activate the LightRL runtime or install PyYAML before running"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ "${NUM_GPUS}" -eq 4 && "${ACTOR_GPUS}" -eq 2 && "${ROLLOUT_GPUS}" -eq 2 && "${TP_SIZE}" -eq 2 ]] \
    || die "the fixed recipe requires NUM_GPUS=4, ACTOR_GPUS=2, ROLLOUT_GPUS=2, TP_SIZE=2"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required on the GPU machine"
  gpu_count="$(nvidia-smi -L | sed -n '/^GPU /p' | wc -l)"
  [[ "${gpu_count}" -ge 4 ]] || die "requires 4 visible GPUs, found ${gpu_count}"
  command -v curl >/dev/null 2>&1 || die "curl is required for the SETA worker health check"
  [[ "${CONFIRM_LOCAL_CLEANUP:-0}" == "1" ]] || die \
    "the launcher stops local Ray/SGLang processes; re-run with CONFIRM_LOCAL_CLEANUP=1 on a dedicated machine"
  if [[ "${SKIP_WORKER_HEALTHCHECK:-0}" != "1" ]]; then
    first_worker="${WORKER_URLS%%,*}"
    first_worker="${first_worker%/}"
    curl --noproxy '*' --fail --silent --show-error --max-time 10 \
      "${first_worker}/healthz" >/dev/null \
      || die "SETA worker is unavailable at ${first_worker}/healthz; start a private/co-located worker or override WORKER_URLS. This recipe does not mutate shared workers."
  else
    warn "SKIP_WORKER_HEALTHCHECK=1: worker connectivity will be checked by the launcher"
  fi
fi

printf '[seta-fixed12] run_id=%s\n' "${RUN_ID}"
printf '[seta-fixed12] topology=4 GPU (2 actor + 2 rollout, TP=2)\n'
printf '[seta-fixed12] worker_urls=%s\n' "${WORKER_URLS}"
printf '[seta-fixed12] eval_config=%s\n' "${EVAL_CONFIG}"
printf '[seta-fixed12] output=%s/runs/%s\n' "${REPO_ROOT}" "${RUN_ID}"

if [[ "${DRY_RUN}" == "1" ]]; then
  bash "${REPO_ROOT}/examples/training/train_qwen3_8b_seta_dapo.sh" --dry-run
else
  bash "${REPO_ROOT}/examples/training/train_qwen3_8b_seta_dapo.sh"
  printf '[seta-fixed12] monitor: tail -f %s/runs/%s/launcher.log\n' "${REPO_ROOT}" "${RUN_ID}"
  printf '[seta-fixed12] trajectories: %s/runs/%s/trajectories/\n' "${REPO_ROOT}" "${RUN_ID}"
fi
