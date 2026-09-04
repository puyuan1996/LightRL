#!/usr/bin/env bash
# Submit the selected LWM phases as independent rjobs.  Site-specific images
# and mounts are overrideable; no private cluster path is required by the
# trainer itself.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RJOB_IMAGE="${RJOB_IMAGE:-${LIGHTRL_RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}}"
RJOB_GPU="${RJOB_GPU:-1}"
RJOB_CPU="${RJOB_CPU:-16}"
RJOB_MEMORY_MB="${RJOB_MEMORY_MB:-128000}"
RJOB_PRIORITY="${RJOB_PRIORITY:-5}"
RJOB_MOUNT="${RJOB_MOUNT:-gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan}"
PHASES="${WM_PHASES:-baseline,replay,value_mpc}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${WM_DRY_RUN:-0}" == "1" ]]; then
  echo "[lwm-rjob] dry-run phases=${PHASES} image=${RJOB_IMAGE} resources=${RJOB_GPU}GPU/${RJOB_CPU}CPU/${RJOB_MEMORY_MB}MiB"
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
  command_string="cd ${REPO_ROOT} && RUNS_ROOT=${RUNS_ROOT} WM_INPUT=${WM_INPUT:-${RUNS_ROOT}/evaluation} WM_SUPPLEMENT_INPUT=${WM_SUPPLEMENT_INPUT:-${RUNS_ROOT}/training} WM_PHASE=${phase} WM_ENCODER=${WM_ENCODER:-hf-policy} WM_HF_MODEL=${WM_HF_MODEL:-${REPO_ROOT}/models/Qwen3-8B} PYTHON_BIN=${PYTHON_BIN} bash examples/training/world_model/run_tb21_lwm_phase.sh"
  echo "[lwm-rjob] submitting ${name} (${phase})"
  rjob submit \
    --name="${name}" \
    --gpu="${RJOB_GPU}" --cpu="${RJOB_CPU}" --memory="${RJOB_MEMORY_MB}" \
    --charged-group="${RJOB_CHARGED_GROUP:-narmodel_gpu}" \
    --priority="${RJOB_PRIORITY}" --image="${RJOB_IMAGE}" \
    --mount="${RJOB_MOUNT}" --privileged=false --host-network=false \
    --share-host-shm=true --termination-grace-period-seconds=30 \
    -- bash -lc "${command_string}"
done
