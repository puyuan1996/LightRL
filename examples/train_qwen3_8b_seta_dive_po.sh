#!/usr/bin/env bash
# Qwen3-8B + SETA + DIVE-PO v0716 centered-gate experiment.
# Harness / Model / Algorithm: Camel-Agent / Qwen3-8B / DIVE-PO (DAPO base).
# Required for a real run: reachable WORKER_URLS or WORKER_URLS_FILE.
# Defaults: K=6 Agent57 arms, centered quality gate, 8 GPUs, max_turn=10.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/experiment/dive_po_qwen3_8b_seta.yaml}"
cd "${REPO_ROOT}"
exec python3 -m agentic_rl.cli train --config "${CONFIG_PATH}" "$@"
