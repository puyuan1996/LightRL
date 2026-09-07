#!/usr/bin/env bash
# Submit the streaming LWM A/B (both arms in one job, shared hidden cache) as
# a single rjob.  Mirrors submit_tb21_lwm_rjob.sh conventions; site-specific
# images and mounts are overrideable.
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
RJOB_PREEMPTIBLE="${RJOB_PREEMPTIBLE:-no}"
RJOB_GPU_AFFINITY_TYPE="${RJOB_GPU_AFFINITY_TYPE:-guaranteed}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_MOUNT_PUYUAN="${RJOB_MOUNT_PUYUAN:-gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan}"
RJOB_MOUNT_NARMODEL="${RJOB_MOUNT_NARMODEL:-gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel}"
RJOB_MOUNT_TRUSTCYBER="${RJOB_MOUNT_TRUSTCYBER:-gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${WM_DRY_RUN:-0}" == "1" ]]; then
  echo "[lwm-stream-rjob] dry-run namespace=${RJOB_NAMESPACE} image=${RJOB_IMAGE} resources=${RJOB_GPU}GPU/${RJOB_CPU}CPU/${RJOB_MEMORY_MB}MiB private-machine=${RJOB_PRIVATE_MACHINE}"
  exit 0
fi
command -v rjob >/dev/null || { echo "[lwm-stream-rjob] rjob CLI not found" >&2; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
name="${RJOB_NAME_PREFIX:-lwm-tb21-stream}-${stamp}"
command_string="cd ${REPO_ROOT} && RUNS_ROOT=${RUNS_ROOT} WM_INPUT=${WM_INPUT:-${RUNS_ROOT}/evaluation} WM_SUPPLEMENT_INPUT=${WM_SUPPLEMENT_INPUT:-${RUNS_ROOT}/training} WM_STAMP=${name} WM_ENCODER=${WM_ENCODER:-hf-policy} WM_HF_MODEL=${WM_HF_MODEL:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B} WM_MAX_TRAJECTORIES=${WM_MAX_TRAJECTORIES:-256} WM_MAX_TRANSITIONS=${WM_MAX_TRANSITIONS:-0} WM_MAX_CONTEXT_TOKENS=${WM_MAX_CONTEXT_TOKENS:-1536} WM_MAX_ACTION_TOKENS=${WM_MAX_ACTION_TOKENS:-512} WM_MAX_FEEDBACK_TOKENS=${WM_MAX_FEEDBACK_TOKENS:-512} WM_ENCODE_BATCH_SIZE=${WM_ENCODE_BATCH_SIZE:-8} WM_BACKPROP_TO_LLM=${WM_BACKPROP_TO_LLM:-0} WM_STREAM_CHUNKS=${WM_STREAM_CHUNKS:-8} WM_STREAM_ARMS=${WM_STREAM_ARMS:-noreplay,replay} WM_STREAM_EPOCHS_PER_CHUNK=${WM_STREAM_EPOCHS_PER_CHUNK:-1} WM_BATCH_SIZE=${WM_BATCH_SIZE:-32} WM_LATENT_DIM=${WM_LATENT_DIM:-128} WM_SEED=${WM_SEED:-42} WM_REPLAY_BUFFER_SIZE=${WM_REPLAY_BUFFER_SIZE:-4096} WM_REPLAY_RATIO=${WM_REPLAY_RATIO:-0.5} WM_REPLAY_WARMUP_CHUNKS=${WM_REPLAY_WARMUP_CHUNKS:-0} PYTHON_BIN=${PYTHON_BIN} bash examples/training/world_model/run_tb21_lwm_stream.sh"
echo "[lwm-stream-rjob] submitting ${name}"
rjob submit \
  --namespace="${RJOB_NAMESPACE}" \
  --name="${name}" \
  --gpu="${RJOB_GPU}" --cpu="${RJOB_CPU}" --memory="${RJOB_MEMORY_MB}" \
  --charged-group="${RJOB_CHARGED_GROUP:-narmodel_gpu}" \
  --priority="${RJOB_PRIORITY}" --image="${RJOB_IMAGE}" \
  --gpu-affinity-type="${RJOB_GPU_AFFINITY_TYPE}" \
  --private-machine="${RJOB_PRIVATE_MACHINE}" --preemptible="${RJOB_PREEMPTIBLE}" \
  --mount="${RJOB_MOUNT_PUYUAN}" --mount="${RJOB_MOUNT_NARMODEL}" \
  --mount="${RJOB_MOUNT_TRUSTCYBER}" --privileged=false --host-network=false \
  --share-host-shm=true --termination-grace-period-seconds=30 \
  -- bash -lc "${command_string}"
