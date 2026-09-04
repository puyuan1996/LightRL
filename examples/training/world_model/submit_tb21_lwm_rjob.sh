#!/usr/bin/env bash
# Submit the selected LWM phases as independent rjobs.  Site-specific images
# and mounts are overrideable; no private cluster path is required by the
# trainer itself.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RJOB_NAMESPACE="${RJOB_NAMESPACE:-ailab-narmodel}"
RJOB_IMAGE="${RJOB_IMAGE:-${LIGHTRL_RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}}"
RJOB_GPU="${RJOB_GPU:-1}"
RJOB_CPU="${RJOB_CPU:-16}"
RJOB_MEMORY_MB="${RJOB_MEMORY_MB:-256000}"
RJOB_PRIORITY="${RJOB_PRIORITY:-5}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_MOUNT_PUYUAN="${RJOB_MOUNT_PUYUAN:-gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan}"
RJOB_MOUNT_NARMODEL="${RJOB_MOUNT_NARMODEL:-gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel}"
RJOB_MOUNT_TRUSTCYBER="${RJOB_MOUNT_TRUSTCYBER:-gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata}"
PHASES="${WM_PHASES:-baseline,replay,value_mpc}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${WM_DRY_RUN:-0}" == "1" ]]; then
  echo "[lwm-rjob] dry-run namespace=${RJOB_NAMESPACE} phases=${PHASES} image=${RJOB_IMAGE} resources=${RJOB_GPU}GPU/${RJOB_CPU}CPU/${RJOB_MEMORY_MB}MiB private-machine=${RJOB_PRIVATE_MACHINE}"
  exit 0
fi
command -v rjob >/dev/null || { echo "[lwm-rjob] rjob CLI not found" >&2; exit 1; }

IFS=',' read -r -a phase_list <<< "${PHASES}"
for phase in "${phase_list[@]}"; do
  phase="${phase//[[:space:]]/}"
  [[ -n "${phase}" ]] || continue
  case "${phase}" in baseline|replay|value_mpc) ;; *) echo "unknown phase: ${phase}" >&2; exit 2 ;; esac
  stamp="$(date +%Y%m%d-%H%M%S)"
  name="${RJOB_NAME_PREFIX:-lwm-tb21}-${phase}-${stamp}"
  command_string="cd ${REPO_ROOT} && RUNS_ROOT=${RUNS_ROOT} WM_INPUT=${WM_INPUT:-${RUNS_ROOT}/evaluation} WM_SUPPLEMENT_INPUT=${WM_SUPPLEMENT_INPUT:-${RUNS_ROOT}/training} WM_PHASE=${phase} WM_STAMP=${name} WM_ENCODER=${WM_ENCODER:-hf-policy} WM_HF_MODEL=${WM_HF_MODEL:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B} WM_MAX_TRAJECTORIES=${WM_MAX_TRAJECTORIES:-256} WM_MAX_TRANSITIONS=${WM_MAX_TRANSITIONS:-0} WM_EPOCHS=${WM_EPOCHS:-5} WM_BATCH_SIZE=${WM_BATCH_SIZE:-32} WM_LATENT_DIM=${WM_LATENT_DIM:-128} WM_SEED=${WM_SEED:-42} WM_REPLAY_BUFFER_SIZE=${WM_REPLAY_BUFFER_SIZE:-4096} WM_REPLAY_RATIO=${WM_REPLAY_RATIO:-1.0} PYTHON_BIN=${PYTHON_BIN} bash examples/training/world_model/run_tb21_lwm_phase.sh"
  echo "[lwm-rjob] submitting ${name} (${phase})"
  rjob submit \
    --namespace="${RJOB_NAMESPACE}" \
    --name="${name}" \
    --gpu="${RJOB_GPU}" --cpu="${RJOB_CPU}" --memory="${RJOB_MEMORY_MB}" \
    --charged-group="${RJOB_CHARGED_GROUP:-narmodel_gpu}" \
    --priority="${RJOB_PRIORITY}" --image="${RJOB_IMAGE}" \
    --private-machine="${RJOB_PRIVATE_MACHINE}" \
    --mount="${RJOB_MOUNT_PUYUAN}" --mount="${RJOB_MOUNT_NARMODEL}" \
    --mount="${RJOB_MOUNT_TRUSTCYBER}" --privileged=false --host-network=false \
    --share-host-shm=true --termination-grace-period-seconds=30 \
    -- bash -lc "${command_string}"
done
