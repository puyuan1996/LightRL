#!/usr/bin/env bash
# Generic LightRL config launcher.
# Combination: selected entirely by CONFIG_PATH.
# Required: a valid config file; runtime-specific checkpoint/worker settings may
# be required by that config. Default: the maintained DIVE-PO recipe.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/experiment/dive_po_qwen3_8b_seta.yaml}"

cd "${REPO_ROOT}"
exec python3 -m agentic_rl.cli train --config "${CONFIG_PATH}" "$@"
