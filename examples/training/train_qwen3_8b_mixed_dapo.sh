#!/usr/bin/env bash
# Qwen3-8B + mixed SETA/Agent-SafetyBench/AgentHarm + DAPO.
# Harness / Model / Algorithm: Camel-Agent / Qwen3-8B / DAPO.
# Required for a real run: workers for SETA tasks and installed safety backends.
# Defaults: data ratio 6:2:2, 8 GPUs, max_turn=10, exploration off.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/experiment/qwen3_8b_mixed_dapo.yaml}"
cd "${REPO_ROOT}"
exec python3 -m agentic_rl.platform.cli train --config "${CONFIG_PATH}" "$@"
