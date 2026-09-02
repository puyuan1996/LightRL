#!/usr/bin/env bash
# Submit a privileged, host-NVMe-backed RJob and keep its private dockerd/SETA
# pool alive for manual `brainctl exec` evaluation. Set KEEP_ALIVE=0 for a
# one-shot doctor smoke. The default topology matches the 4-GPU evaluation
# recipe; override RJOB_GPU/RJOB_CPU/RJOB_MEMORY for a smaller smoke.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
PERSIST_ROOT="${LIGHTRL_PERSIST_ROOT:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl}"
BUNDLE="${DOCKER_OFFLINE_BUNDLE_DIR:-${PERSIST_ROOT}/runtime/docker-rft-20260408-ubuntu22.04}"
RUN_ID="${RUN_ID:-lightrl-private-dind-smoke-$(date +%Y%m%d-%H%M%S)}"
RJOB_NAME="${RJOB_NAME:-${RUN_ID}}"
CHARGED_GROUP="${CHARGED_GROUP:-narmodel_gpu}"
GPU_COUNT="${RJOB_GPU:-4}"
CPU_COUNT="${RJOB_CPU:-50}"
MEMORY_MIB="${RJOB_MEMORY:-800000}"
KEEP_ALIVE="${KEEP_ALIVE:-1}"
DIND_MODE="${DIND_MODE:-serve}"
[[ "${DIND_MODE}" == "doctor" || "${DIND_MODE}" == "serve" ]] || {
  echo "DIND_MODE must be doctor or serve" >&2
  exit 2
}

[[ -d "${BUNDLE}" ]] || {
  echo "missing Docker offline bundle: ${BUNDLE}" >&2
  exit 1
}

read -r -d '' COMMAND <<'EOS' || true
set -euo pipefail
cd /mnt/shared-storage-user/puyuan/code/LightRL
export HOST_NETWORK=false
export LIGHTRL_DIND_ISOLATED_NETWORK=1
export RJOB_DIND_NETWORK_MODE=pod-isolated
export RJOB_DIND_INSTANCE="${RUN_ID}"
export DOCKER_DATA_ROOT="/nvme/lightrl-dind/${RUN_ID}/docker"
export DOCKER_OFFLINE_BUNDLE_DIR="${DOCKER_OFFLINE_BUNDLE_DIR}"

bash deploy/workers/start_rjob_dind_worker.sh "${DIND_MODE}"
STATE="/run/lightrl-dind/${RUN_ID:0:32}"
source "${STATE}/worker.env"
docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}}'
test "$(docker info --format '{{.DockerRootDir}}')" = "${DOCKER_DATA_ROOT}"
test -S "${STATE}/docker.sock"
test -L /var/run/docker.sock
if [[ "${DIND_MODE}" == "serve" ]]; then
  curl --noproxy '*' --fail --silent --show-error --max-time 10 \
    "${WORKER_URLS}/healthz" >/dev/null
  echo "private_dind_ready=PASS socket=${STATE}/docker.sock worker=${WORKER_URLS}"
else
  echo "private_dind_doctor=PASS socket=${STATE}/docker.sock"
fi
if [[ "${KEEP_ALIVE}" == "1" ]]; then
  echo "container_ready=1; exec into this replica and run:"
  echo "  source ${STATE}/worker.env"
  echo "  cd /mnt/shared-storage-user/puyuan/code/LightRL"
  echo "  CONFIRM_LOCAL_CLEANUP=1 BACKGROUND=0 bash examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh"
  exec sleep infinity
fi
EOS

submit_args=()
[[ "${RJOB_DRY_RUN:-0}" == "1" ]] && submit_args+=(--dry-run)
if python3 "${REPO_ROOT}/deploy/rjob/submit_host_nvme_rjob.py" --help 2>&1 \
    | grep -q -- '--private-machine'; then
  submit_args+=(--private-machine="${PRIVATE_MACHINE:-no}")
fi

exec python3 "${REPO_ROOT}/deploy/rjob/submit_host_nvme_rjob.py" \
  "${submit_args[@]}" \
  --name="${RJOB_NAME}" \
  --image="${IMAGE}" \
  --gpu="${GPU_COUNT}" --cpu="${CPU_COUNT}" --memory="${MEMORY_MIB}" \
  --charged-group="${CHARGED_GROUP}" \
  --priority="${RJOB_PRIORITY:-5}" \
  --env="INSIDE_RJOB=1" \
  --env="RUN_ID=${RUN_ID}" \
  --env="DOCKER_OFFLINE_BUNDLE_DIR=${BUNDLE}" \
  --env="HOST_NETWORK=false" \
  --env="LIGHTRL_DIND_ISOLATED_NETWORK=1" \
  --env="RJOB_DIND_NETWORK_MODE=pod-isolated" \
  --env="DIND_MODE=${DIND_MODE}" \
  --env="KEEP_ALIVE=${KEEP_ALIVE}" \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel \
  --mount=gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata \
  -- bash -lc "${COMMAND}"
