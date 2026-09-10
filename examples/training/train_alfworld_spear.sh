#!/usr/bin/env bash
set -euo pipefail

# Official SPEAR ALFWorld settings. The current LightRL terminal launcher is
# intentionally not used: ALFWorld requires an environment-worker adapter.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ALFWORLD_DATA_DIR="${ALFWORLD_DATA_DIR:-${ROOT}/datasets/alfworld}"
export ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ALFWORLD_DATA_DIR}/base_config.yaml}"
export MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B}"
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-32}" GROUP_SIZE="${GROUP_SIZE:-8}"
export MAX_STEPS="${MAX_STEPS:-50}" REPLAY_LOSS_COEF="${REPLAY_LOSS_COEF:-1}"
[[ -d "${ALFWORLD_DATA_DIR}" ]] || bash "${ROOT}/examples/training/download_alfworld.sh"
OFFICIAL_ROOT="${SPEAR_VERL_AGENT_ROOT:-/mnt/shared-storage-user/puyuan/code/archive/SPEAR/verl-agent}"
[[ -d "${OFFICIAL_ROOT}/verl" ]] || { echo "missing verl-agent runtime: ${OFFICIAL_ROOT}" >&2; exit 2; }
export PYTHONPATH="${OFFICIAL_ROOT}:${ROOT}:${PYTHONPATH:-}"
cd "${OFFICIAL_ROOT}"
exec python3 -m verl.trainer.main_ppo \
  data.train_batch_size="${TRAIN_DATA_SIZE}" actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
  env.env_name=alfworld/AlfredTWEnv env.max_steps="${MAX_STEPS}" \
  actor_rollout_ref.actor.enable_trajectory_replay=true \
  actor_rollout_ref.actor.trajectory_buffer_size=2048 \
  actor_rollout_ref.actor.trajectory_score_threshold=1 \
  actor_rollout_ref.actor.trajectory_tolerate_steps=5 \
  actor_rollout_ref.actor.replay_loss_coef="${REPLAY_LOSS_COEF}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  env.config="${ALFWORLD_CONFIG}" "$@"
