#!/usr/bin/env bash
# Collect redacted LWM transitions into an isolated replay during SETA training.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

WM_TRAIN_SCRIPT="${WM_TRAIN_SCRIPT:-examples/training/train_qwen3_8b_seta_dapo.sh}"
WM_REPLAY_BUFFER_SIZE="${WM_REPLAY_BUFFER_SIZE:-4096}"
WM_METADATA_MAX_CHARS="${WM_METADATA_MAX_CHARS:-4096}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-1}"

if [[ "${WM_TRAIN_SCRIPT}" != /* ]]; then
  WM_TRAIN_SCRIPT="${REPO_ROOT}/${WM_TRAIN_SCRIPT}"
fi
[[ -f "${WM_TRAIN_SCRIPT}" ]] || { echo "[lwm-replay] missing training script: ${WM_TRAIN_SCRIPT}" >&2; exit 2; }
[[ "${WM_REPLAY_BUFFER_SIZE}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "[lwm-replay] WM_REPLAY_BUFFER_SIZE must be positive" >&2; exit 2; }
[[ "${MAX_CKPT_KEEP}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "[lwm-replay] MAX_CKPT_KEEP must be positive so replay snapshots are saved" >&2; exit 2; }

WORLD_MODEL_ARGS=(
  --world-model-enable
  --world-model-use-dapo-replay-buffer
  --world-model-replay-buffer-size "${WM_REPLAY_BUFFER_SIZE}"
  --world-model-metadata-max-chars "${WM_METADATA_MAX_CHARS}"
)

export EXTRA_ALGO_ARGS="${EXTRA_ALGO_ARGS:+${EXTRA_ALGO_ARGS} }${WORLD_MODEL_ARGS[*]}"
export MAX_CKPT_KEEP

printf '[lwm-replay] training_script=%s capacity=%s max_chars=%s\n' \
  "${WM_TRAIN_SCRIPT}" "${WM_REPLAY_BUFFER_SIZE}" "${WM_METADATA_MAX_CHARS}"
exec bash "${WM_TRAIN_SCRIPT}" "$@"
