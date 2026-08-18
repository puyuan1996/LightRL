#!/usr/bin/env bash
# Generic SETA latent world-model entrypoint. Configure it through WM_* variables.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

: "${WM_TRAJECTORIES:?set WM_TRAJECTORIES to a SETA directory, records JSONL, or verified replay .pt}"

exec bash tools/world_model/run_world_model_seta_latent.sh "$@"
