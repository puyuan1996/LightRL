#!/usr/bin/env bash
# Start the local Docker daemon (when needed) and the LightRL CPU pool server.
#
# This script is intentionally conservative:
# - it never force-repairs Docker unless AUTO_REPAIR_DOCKER=1;
# - it skips global preflight/final cleanup by default;
# - it keeps the pool server in the foreground so Ctrl-C performs normal cleanup.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/LightRL}"
ENV_SERVER_HOST="${ENV_SERVER_HOST:-0.0.0.0}"
ENV_SERVER_PORT="${ENV_SERVER_PORT:-18081}"
WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-4}"
WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-2}"
WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-1}"
WORKER_MAX_CONCURRENT_RESETS="${WORKER_MAX_CONCURRENT_RESETS:-2}"
WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-4}"
AUTO_REPAIR_DOCKER="${AUTO_REPAIR_DOCKER:-0}"
DOCKER_START_TIMEOUT="${DOCKER_START_TIMEOUT:-60}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-}"
PROXY_URL="${PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-1}"
FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-0}"
TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-lightrl-validation}"

log() {
  printf '[local-docker-validation] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

docker_ready() {
  timeout 10 docker info >/dev/null 2>&1
}

wait_for_docker() {
  local elapsed=0
  while (( elapsed < DOCKER_START_TIMEOUT )); do
    if docker_ready; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

start_docker_daemon() {
  if docker_ready; then
    log "Docker daemon is already ready"
    return 0
  fi

  log "Docker daemon is not ready; trying the host service manager"
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl start docker || true
  elif command -v service >/dev/null 2>&1; then
    sudo service docker start || true
  fi

  if wait_for_docker; then
    log "Docker daemon became ready"
    return 0
  fi

  if [[ "${AUTO_REPAIR_DOCKER}" != "1" ]]; then
    die "Docker is still unavailable. Inspect 'systemctl status docker' or rerun with AUTO_REPAIR_DOCKER=1 for the repository repair workflow."
  fi

  local repair_root="${DOCKER_DATA_ROOT}"
  if [[ -z "${repair_root}" ]]; then
    repair_root="$(timeout 10 docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    repair_root="${repair_root:-/data}"
  fi
  log "Running explicit Docker repair with data root ${repair_root}"
  sudo env \
    DOCKER_DATA_ROOT="${repair_root}" \
    PROXY_URL="${PROXY_URL}" \
    bash "${REPO_ROOT}/deploy/workers/fix_dockerd_and_proxy.sh"
  wait_for_docker || die "Docker repair completed but the daemon is still unavailable"
}

existing_server_ready() {
  curl --noproxy '*' --silent --show-error --fail --max-time 5 \
    "http://127.0.0.1:${ENV_SERVER_PORT}/healthz" >/dev/null 2>&1
}

main() {
  [[ -d "${REPO_ROOT}/agentic_rl" ]] || die "LightRL repo not found: ${REPO_ROOT}"
  command -v docker >/dev/null 2>&1 || die "docker CLI is not installed"
  command -v curl >/dev/null 2>&1 || die "curl is required for health checks"
  command -v timeout >/dev/null 2>&1 || die "timeout is required"
  cd "${REPO_ROOT}"

  start_docker_daemon
  docker compose version >/dev/null

  if existing_server_ready; then
    log "Pool server is already healthy at http://127.0.0.1:${ENV_SERVER_PORT}"
    log "WORKER_URLS=http://$(hostname -I | awk '{print $1}'):${ENV_SERVER_PORT}"
    return 0
  fi

  if command -v ss >/dev/null 2>&1 \
    && ss -ltn "( sport = :${ENV_SERVER_PORT} )" | grep -q ":${ENV_SERVER_PORT}"; then
    die "Port ${ENV_SERVER_PORT} is occupied, but its /healthz is not healthy"
  fi

  log "Starting LightRL CPU pool server in the foreground"
  log "capacity=$((WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK)) port=${ENV_SERVER_PORT}"
  log "After startup, use WORKER_URLS=http://$(hostname -I | awk '{print $1}'):${ENV_SERVER_PORT}"
  log "Press Ctrl-C to stop the server"

  export ENV_SERVER_HOST ENV_SERVER_PORT
  export WORKER_MAX_TASKS WORKER_MAX_RUNS_PER_TASK
  export WORKER_MAX_CONCURRENT_BUILDS WORKER_MAX_CONCURRENT_RESETS
  export WORKER_MAX_CONCURRENT_CLOSES
  export SKIP_PREFLIGHT_CLEANUP FINAL_DOCKER_CLEANUP
  export TERMINAL_RL_POOL_NAMESPACE

  exec bash deploy/workers/run_pool_server_pu_v2.sh
}

main "$@"
