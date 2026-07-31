#!/usr/bin/env bash
# GLM-5.1 + SETA + DAPO integration entry.
# Harness / Model / Algorithm: Camel-Agent / GLM-5.1 / DAPO.
# Required: HF_CKPT, REF_LOAD, MODEL_ARGS_FILE, plus WORKER_URLS[_FILE].
# Defaults: GLM-5.1 rollout parser config, 8 GPUs, max_turn=10, exploration off.
# MODEL_ARGS_FILE must name a compatible file under slime/scripts/models/.
set -euo pipefail

if [[ "${1:-}" != "--dry-run" ]]; then
  : "${HF_CKPT:?HF_CKPT is required for GLM-5.1 training}"
  : "${REF_LOAD:?REF_LOAD is required for GLM-5.1 training}"
  : "${MODEL_ARGS_FILE:?MODEL_ARGS_FILE is required for GLM-5.1 training}"
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/experiment/glm_5_1_seta_dapo.yaml}"
cd "${REPO_ROOT}"
exec python3 -m agentic_rl.cli train --config "${CONFIG_PATH}" "$@"
