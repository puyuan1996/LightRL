#!/usr/bin/env bash
# Replace the local SETA Docker pool server in one command.
#
# This wrapper intentionally scopes the restart to the listener on
# ENV_SERVER_PORT.  It does not run host-wide docker prune; use
# deploy/ops/docker_storage_gc.py separately when storage reclamation is needed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18082}"
export STOP_EXISTING_SERVER=1
export STOP_EXISTING_TIMEOUT="${STOP_EXISTING_TIMEOUT:-30}"

# Keep startup usable on the current small Docker partition.  Cleanup remains
# namespace-aware and is performed by the pool's normal preflight sweep.
export WORKER_DISK_GUARD_STRICT="${WORKER_DISK_GUARD_STRICT:-0}"
export SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-0}"
export FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-1}"

exec bash "${SCRIPT_DIR}/run_pool_server.sh" "$@"
