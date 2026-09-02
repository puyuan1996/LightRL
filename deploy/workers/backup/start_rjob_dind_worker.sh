#!/usr/bin/env bash
# Start a private Docker daemon and LightRL SETA pool inside one privileged RJob Pod.
#
# Required RJob properties:
#   --privileged=true --store-host-nvme=true, plus either:
#   (a) --host-network=false + RJOB_DIND_NETWORK_MODE=pod-isolated, or
#   (b) --host-network=true  + RJOB_DIND_NETWORK_MODE=slirp-netns
#
# Usage inside the RJob:
#   LIGHTRL_DIND_ISOLATED_NETWORK=1 \
#   RUN_ID=<run-id> \
#   bash deploy/workers/start_rjob_dind_worker.sh doctor
#
#   source /run/lightrl-dind/<instance>/worker.env
#   # Use `serve` instead of `doctor` to also start the pool server.
set -euo pipefail

MODE="${1:-serve}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

log() { printf '[rjob-dind] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

[[ "${MODE}" == "doctor" || "${MODE}" == "serve" ]] \
  || die "mode must be doctor or serve"
[[ "$(id -u)" -eq 0 ]] || die "privileged RJob entrypoint must run as root"
[[ -d /nvme && -w /nvme ]] || die "/nvme is missing or not writable; use --store-host-nvme=true"

RAW_INSTANCE="${RJOB_DIND_INSTANCE:-${RUN_ID:-${HOSTNAME:-rjob}}}"
INSTANCE="${RAW_INSTANCE//[^a-zA-Z0-9_.-]/_}"
INSTANCE="${INSTANCE:0:32}"
[[ -n "${INSTANCE}" ]] || die "empty DinD instance name"

STATE_ROOT="${RJOB_DIND_STATE_ROOT:-/run/lightrl-dind/${INSTANCE}}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/nvme/lightrl-dind/${INSTANCE}/docker}"
DOCKER_OFFLINE_BUNDLE_DIR="${DOCKER_OFFLINE_BUNDLE_DIR:-}"
ENABLE_INNER_GPU="${RJOB_DIND_ENABLE_INNER_GPU:-0}"
NETWORK_MODE="${RJOB_DIND_NETWORK_MODE:-pod-isolated}"
DOCKER_START_TIMEOUT="${DOCKER_START_TIMEOUT:-180}"
# Docker Hub's anonymous quota is shared by the cluster egress. Prefer the
# same pull-through mirrors already used by the standalone SETA worker, while
# retaining the canonical image names in every task Dockerfile.
DOCKER_REGISTRY_MIRRORS="${DOCKER_REGISTRY_MIRRORS:-https://docker.m.daocloud.io,https://docker.1ms.run}"
ENV_SERVER_PORT="${ENV_SERVER_PORT:-18081}"
DOCKER_SOCKET="${STATE_ROOT}/docker.sock"
DOCKER_HOST="unix://${DOCKER_SOCKET}"
DAEMON_CONFIG="${STATE_ROOT}/daemon.json"
DOCKER_LOG="${STATE_ROOT}/dockerd.log"
DOCKER_LAUNCHER_PID="${STATE_ROOT}/dockerd-launcher.pid"
POOL_LOG="${STATE_ROOT}/pool_server.log"
POOL_PID="${STATE_ROOT}/pool_server.pid"
ENV_FILE="${STATE_ROOT}/worker.env"
SLIRP_LOG="${STATE_ROOT}/slirp4netns.log"
SLIRP_PID="${STATE_ROOT}/slirp4netns.pid"
SLIRP_READY_FIFO="${STATE_ROOT}/slirp4netns.ready"
NETNS_PID="${STATE_ROOT}/netns.pid"
NETNS_GATE="${STATE_ROOT}/netns.start-dockerd"

case "${HOST_NETWORK:-false}:${NETWORK_MODE}" in
  1:slirp-netns|true:slirp-netns|TRUE:slirp-netns|yes:slirp-netns|YES:slirp-netns)
    log "hostNetwork Pod accepted only because dockerd is isolated in slirp-netns"
    ;;
  1:*|true:*|TRUE:*|yes:*|YES:*)
    die "refusing hostNetwork RJob unless RJOB_DIND_NETWORK_MODE=slirp-netns"
    ;;
  *:pod-isolated)
    [[ "${LIGHTRL_DIND_ISOLATED_NETWORK:-0}" == "1" ]] \
      || die "set LIGHTRL_DIND_ISOLATED_NETWORK=1 only when RJob uses --host-network=false"
    ;;
  *:slirp-netns) ;;
  *) die "RJOB_DIND_NETWORK_MODE must be pod-isolated or slirp-netns" ;;
esac

case "${DOCKER_DATA_ROOT}/" in
  /nvme/*) ;;
  *) die "DOCKER_DATA_ROOT must be under /nvme, got ${DOCKER_DATA_ROOT}" ;;
esac

# Offline APT installation stores its temporary source/list metadata here, so
# the state directory must exist before package installation begins.
mkdir -p "${STATE_ROOT}"

install_offline_packages_if_needed() {
  if command -v docker >/dev/null 2>&1 \
    && command -v dockerd >/dev/null 2>&1 \
    && docker compose version >/dev/null 2>&1 \
    && { [[ "${NETWORK_MODE}" != "slirp-netns" ]] || command -v slirp4netns >/dev/null 2>&1; }; then
    return 0
  fi
  [[ -n "${DOCKER_OFFLINE_BUNDLE_DIR}" && -d "${DOCKER_OFFLINE_BUNDLE_DIR}" ]] \
    || die "Docker Engine/Compose missing; set DOCKER_OFFLINE_BUNDLE_DIR to an offline .deb bundle"

  local packages=()
  mapfile -d '' packages < <(
    find "${DOCKER_OFFLINE_BUNDLE_DIR}" -maxdepth 1 -type f -name '*.deb' -print0 | sort -z
  )
  (( ${#packages[@]} > 0 )) || die "no .deb files in ${DOCKER_OFFLINE_BUNDLE_DIR}"
  if [[ -f "${DOCKER_OFFLINE_BUNDLE_DIR}/Packages" \
        && -f "${DOCKER_OFFLINE_BUNDLE_DIR}/ROOT_PACKAGES" \
        && -x "$(command -v apt-get 2>/dev/null || true)" ]]; then
    local source_list="${STATE_ROOT}/docker-offline.list"
    local apt_lists="${STATE_ROOT}/apt-lists"
    local roots=()
    printf 'deb [trusted=yes] file:%s ./\n' "${DOCKER_OFFLINE_BUNDLE_DIR}" >"${source_list}"
    mkdir -p "${apt_lists}/partial"
    mapfile -t roots < <(sed '/^[[:space:]]*$/d' "${DOCKER_OFFLINE_BUNDLE_DIR}/ROOT_PACKAGES")
    (( ${#roots[@]} > 0 )) || die "ROOT_PACKAGES is empty in ${DOCKER_OFFLINE_BUNDLE_DIR}"
    log "installing offline Docker roots through local APT repo: ${roots[*]}"
    apt-get -o APT::Sandbox::User=root \
      -o Dir::Etc::sourcelist="${source_list}" -o Dir::Etc::sourceparts=- \
      -o Dir::State::lists="${apt_lists}" update >/dev/null \
      || die "could not index offline Docker APT bundle"
    # The only configured source is the local file: repository.  Do not use
    # --no-download here: APT still needs its file method to copy selected
    # packages into the cache before dpkg installation.
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends -y \
      -o APT::Sandbox::User=root \
      -o Dir::Etc::sourcelist="${source_list}" -o Dir::Etc::sourceparts=- \
      -o Dir::State::lists="${apt_lists}" install "${roots[@]}" \
      || die "offline APT installation failed; bundle is incomplete"
  else
    log "installing ${#packages[@]} offline Docker package(s) with dpkg fallback"
    dpkg -i "${packages[@]}" \
      || die "offline package installation failed; bundle must include all dependencies"
  fi
}

install_offline_packages_if_needed
for command_name in docker dockerd python3 curl timeout findmnt; do
  require_command "${command_name}"
done
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 plugin is unavailable"

if [[ "${NETWORK_MODE}" == "slirp-netns" ]]; then
  for command_name in ip slirp4netns unshare; do
    require_command "${command_name}"
  done
fi

if [[ "${ENABLE_INNER_GPU}" == "1" ]]; then
  require_command nvidia-container-runtime
  require_command nvidia-smi
fi

mkdir -p "${DOCKER_DATA_ROOT}"
BACKING_FS="$(findmnt -n -T "${DOCKER_DATA_ROOT}" -o FSTYPE 2>/dev/null || true)"
DOCKER_STORAGE_DRIVER="${DOCKER_STORAGE_DRIVER:-}"
if [[ -z "${DOCKER_STORAGE_DRIVER}" ]]; then
  case "${BACKING_FS}" in
    ext2|ext3|ext4|xfs) DOCKER_STORAGE_DRIVER="overlay2" ;;
    *)
      [[ -e /dev/fuse ]] || die "unsupported backing fs=${BACKING_FS:-unknown} and /dev/fuse is absent"
      require_command fuse-overlayfs
      DOCKER_STORAGE_DRIVER="fuse-overlayfs"
      ;;
  esac
fi
[[ "${DOCKER_STORAGE_DRIVER}" != "vfs" ]] || die "vfs is prohibited for SETA long runs"

python3 - "${DAEMON_CONFIG}" "${DOCKER_DATA_ROOT}" "${DOCKER_STORAGE_DRIVER}" "${ENABLE_INNER_GPU}" "${DOCKER_REGISTRY_MIRRORS}" <<'PY'
import json
import sys

path, data_root, storage_driver, enable_inner_gpu, raw_mirrors = sys.argv[1:]
registry_mirrors = [item.strip().rstrip("/") for item in raw_mirrors.split(",") if item.strip()]
if any(not item.startswith("https://") for item in registry_mirrors):
    raise SystemExit("DOCKER_REGISTRY_MIRRORS accepts only HTTPS endpoints")
config = {
    "data-root": data_root,
    "storage-driver": storage_driver,
    "registry-mirrors": registry_mirrors,
    "default-address-pools": [{"base": "172.24.0.0/13", "size": 24}],
    "max-concurrent-downloads": 8,
    "max-concurrent-uploads": 4,
    # Keep already-started SETA containers alive across a recoverable dockerd
    # restart. The outer trainer/worker supervisor still detects lost leases.
    "live-restore": True,
    "log-driver": "json-file",
    "log-opts": {"max-size": "50m", "max-file": "3"},
}
if enable_inner_gpu == "1":
    config["runtimes"] = {
        "nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}
    }
with open(path, "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
PY

export DOCKER_HOST
if ! timeout 5 docker info >/dev/null 2>&1; then
  rm -f "${DOCKER_SOCKET}" "${STATE_ROOT}/docker.pid"
  log "starting private dockerd: data_root=${DOCKER_DATA_ROOT} driver=${DOCKER_STORAGE_DRIVER}"
  docker_args=(
    --config-file "${DAEMON_CONFIG}"
    --host "unix://${DOCKER_SOCKET}"
    --exec-root "${STATE_ROOT}/exec"
    --pidfile "${STATE_ROOT}/docker.pid"
  )
  if [[ "${NETWORK_MODE}" == "slirp-netns" ]]; then
    # Create dockerd inside the namespace from process birth. Persisted
    # `ip netns` mounts cannot be re-entered reliably under the RJob shim
    # (setns returns EINVAL), whereas slirp's standard PID mode does not need a
    # later setns operation.
    rm -f "${NETNS_PID}" "${NETNS_GATE}" "${SLIRP_READY_FIFO}"
    nohup unshare --net -- bash -c '
      netns_pid=$1
      start_gate=$2
      shift 2
      ip link set lo up
      printf "%s\n" "$$" >"${netns_pid}"
      while [[ ! -e "${start_gate}" ]]; do sleep 0.1; done
      exec dockerd "$@"
    ' bash "${NETNS_PID}" "${NETNS_GATE}" "${docker_args[@]}" \
      >"${DOCKER_LOG}" 2>&1 &
    printf '%s\n' "$!" >"${DOCKER_LAUNCHER_PID}"

    elapsed=0
    while [[ ! -s "${NETNS_PID}" ]]; do
      if ! kill -0 "$(<"${DOCKER_LAUNCHER_PID}")" 2>/dev/null; then
        tail -100 "${DOCKER_LOG}" >&2 || true
        die "dockerd network namespace process exited during startup"
      fi
      (( elapsed < 30 )) || die "dockerd network namespace was not ready in 30s"
      sleep 1
      elapsed=$((elapsed + 1))
    done

    mkfifo "${SLIRP_READY_FIFO}"
    exec 9<>"${SLIRP_READY_FIFO}"
    nohup slirp4netns \
      --configure --mtu=65520 --enable-sandbox --ready-fd=9 \
      "$(<"${NETNS_PID}")" tap0 9>&9 >"${SLIRP_LOG}" 2>&1 &
    printf '%s\n' "$!" >"${SLIRP_PID}"
    if ! IFS= read -r -n 1 -t 30 -u 9; then
      exec 9>&-
      tail -100 "${SLIRP_LOG}" >&2 || true
      die "slirp4netns did not configure the dockerd namespace in 30s"
    fi
    exec 9>&-
    rm -f "${SLIRP_READY_FIFO}"
    touch "${NETNS_GATE}"
    log "dockerd child network namespace ready via slirp4netns"
  else
    nohup dockerd "${docker_args[@]}" >"${DOCKER_LOG}" 2>&1 &
    printf '%s\n' "$!" >"${DOCKER_LAUNCHER_PID}"
  fi
fi

elapsed=0
while ! timeout 10 docker info >/dev/null 2>&1; do
  if (( elapsed >= DOCKER_START_TIMEOUT )); then
    tail -100 "${DOCKER_LOG}" >&2 || true
    die "dockerd did not become ready in ${DOCKER_START_TIMEOUT}s"
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

# CAMEL 0.2.90's TerminalToolkit constructs one low-level Docker API client
# with a hard-coded /var/run/docker.sock even when DOCKER_HOST is set.  Expose
# only this Pod's private socket at that compatibility path.  Fail closed if
# the image already provides a different daemon socket; never replace it.
DEFAULT_DOCKER_SOCKET="/var/run/docker.sock"
if [[ -e "${DEFAULT_DOCKER_SOCKET}" || -L "${DEFAULT_DOCKER_SOCKET}" ]]; then
  if [[ -L "${DEFAULT_DOCKER_SOCKET}" \
        && "$(readlink "${DEFAULT_DOCKER_SOCKET}")" == "${DOCKER_SOCKET}" ]]; then
    :
  elif [[ -L "${DEFAULT_DOCKER_SOCKET}" ]]; then
    PREVIOUS_PRIVATE_SOCKET="$(readlink "${DEFAULT_DOCKER_SOCKET}")"
    case "${PREVIOUS_PRIVATE_SOCKET}" in
      /run/lightrl-dind/*/docker.sock)
        # A sleep-infinity debug box intentionally runs several experiments in
        # sequence. Replace only a stale socket created by this launcher; never
        # overwrite a live daemon or an arbitrary host/container socket.
        if [[ ! -S "${PREVIOUS_PRIVATE_SOCKET}" ]] \
          || ! timeout 5 docker --host "unix://${PREVIOUS_PRIVATE_SOCKET}" info >/dev/null 2>&1; then
          rm -f "${DEFAULT_DOCKER_SOCKET}"
          ln -s "${DOCKER_SOCKET}" "${DEFAULT_DOCKER_SOCKET}"
        else
          die "${DEFAULT_DOCKER_SOCKET} points to another live private RJob daemon"
        fi
        ;;
      *) die "${DEFAULT_DOCKER_SOCKET} points outside the private RJob state root" ;;
    esac
  else
    die "${DEFAULT_DOCKER_SOCKET} already exists and is not the private RJob socket"
  fi
else
  ln -s "${DOCKER_SOCKET}" "${DEFAULT_DOCKER_SOCKET}"
fi
timeout 10 docker --host "unix://${DEFAULT_DOCKER_SOCKET}" info >/dev/null \
  || die "CAMEL Docker socket compatibility path is unusable"

ACTUAL_ROOT="$(docker info --format '{{.DockerRootDir}}')"
ACTUAL_DRIVER="$(docker info --format '{{.Driver}}')"
[[ "${ACTUAL_ROOT}" == "${DOCKER_DATA_ROOT}" ]] \
  || die "DockerRootDir mismatch: expected=${DOCKER_DATA_ROOT} actual=${ACTUAL_ROOT}"
[[ "${ACTUAL_DRIVER}" == "${DOCKER_STORAGE_DRIVER}" ]] \
  || die "storage driver mismatch: expected=${DOCKER_STORAGE_DRIVER} actual=${ACTUAL_DRIVER}"

PROBE_NETWORK="lightrl-dind-probe-${INSTANCE}"
docker network rm "${PROBE_NETWORK}" >/dev/null 2>&1 || true
docker network create "${PROBE_NETWORK}" >/dev/null
docker network inspect "${PROBE_NETWORK}" >/dev/null
docker network rm "${PROBE_NETWORK}" >/dev/null

{
  printf 'export DOCKER_HOST=%q\n' "${DOCKER_HOST}"
  printf 'export DOCKER_DATA_ROOT=%q\n' "${DOCKER_DATA_ROOT}"
  printf 'export WORKER_URLS=%q\n' "http://127.0.0.1:${ENV_SERVER_PORT}"
  printf 'export ENV_SERVER_HOST=%q\n' "127.0.0.1"
  printf 'export ENV_SERVER_PORT=%q\n' "${ENV_SERVER_PORT}"
  printf 'export RJOB_DIND_NETWORK_MODE=%q\n' "${NETWORK_MODE}"
} >"${ENV_FILE}"

log "Docker ready: root=${ACTUAL_ROOT} driver=${ACTUAL_DRIVER} socket=${DOCKER_SOCKET} mirrors=${DOCKER_REGISTRY_MIRRORS}"
if [[ "${MODE}" == "doctor" ]]; then
  log "doctor passed; source ${ENV_FILE}"
  exit 0
fi

if curl --noproxy '*' --fail --silent --max-time 5 \
  "http://127.0.0.1:${ENV_SERVER_PORT}/healthz" >/dev/null 2>&1; then
  log "pool server already healthy"
  exit 0
fi

export ENV_SERVER_HOST="127.0.0.1"
export ENV_SERVER_PORT
export WORKER_MAX_TOTAL_RUNS="${WORKER_MAX_TOTAL_RUNS:-40}"
export WORKER_MAX_TASKS="${WORKER_MAX_TASKS:-32}"
export WORKER_MAX_RUNS_PER_TASK="${WORKER_MAX_RUNS_PER_TASK:-8}"
VISIBLE_CPUS="$(nproc 2>/dev/null || echo 1)"
if (( VISIBLE_CPUS >= 128 )); then
  DEFAULT_CONCURRENT_BUILDS=8
else
  DEFAULT_CONCURRENT_BUILDS=4
fi
export WORKER_MAX_CONCURRENT_BUILDS="${WORKER_MAX_CONCURRENT_BUILDS:-${DEFAULT_CONCURRENT_BUILDS}}"
export WORKER_MAX_CONCURRENT_RESETS="${WORKER_MAX_CONCURRENT_RESETS:-32}"
export WORKER_MAX_CONCURRENT_CLOSES="${WORKER_MAX_CONCURRENT_CLOSES:-32}"
export WORKER_DOCKER_BUILD_QUEUE_TIMEOUT="${WORKER_DOCKER_BUILD_QUEUE_TIMEOUT:-1200}"
export SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-1}"
export FINAL_DOCKER_CLEANUP="${FINAL_DOCKER_CLEANUP:-0}"
# Legacy SETA compose projects require the default namespace.  This daemon is
# already isolated per RJob/run, so default cannot collide with another pool.
export TERMINAL_RL_POOL_NAMESPACE="${TERMINAL_RL_POOL_NAMESPACE:-default}"

nohup bash "${REPO_ROOT}/deploy/workers/run_pool_server_pu_v2.sh" \
  >"${POOL_LOG}" 2>&1 &
printf '%s\n' "$!" >"${POOL_PID}"

elapsed=0
while ! curl --noproxy '*' --fail --silent --max-time 5 \
  "http://127.0.0.1:${ENV_SERVER_PORT}/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$(<"${POOL_PID}")" 2>/dev/null; then
    tail -100 "${POOL_LOG}" >&2 || true
    die "pool server exited during startup"
  fi
  if (( elapsed >= 120 )); then
    tail -100 "${POOL_LOG}" >&2 || true
    die "pool server did not become healthy in 120s"
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

log "SETA pool ready: WORKER_URLS=http://127.0.0.1:${ENV_SERVER_PORT}"
log "source ${ENV_FILE} before launching training"
