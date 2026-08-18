#!/usr/bin/env bash
# Qwen policy-hidden JEPA next-belief training with grouped heldout evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

: "${WM_TRAJECTORIES:?set WM_TRAJECTORIES to a SETA directory, records JSONL, or verified replay .pt}"
: "${WM_HF_MODEL:?set WM_HF_MODEL to the local policy checkpoint}"

export WM_ENCODER="${WM_ENCODER:-hf-policy}"
export WM_STATE_VIEW="${WM_STATE_VIEW:-belief_view_v1}"
export WM_PREDICTION_TARGET="${WM_PREDICTION_TARGET:-next_state}"
export WM_OBJECTIVE_POPULATION="${WM_OBJECTIVE_POPULATION:-has_next}"
export WM_PRED_LOSS_TYPE="${WM_PRED_LOSS_TYPE:-scaled_mse}"
export WM_SPLIT_GROUP_KEY="${WM_SPLIT_GROUP_KEY:-task_id}"
export WM_CHECKPOINT_SELECTION="${WM_CHECKPOINT_SELECTION:-best_validation}"

exec bash examples/training/world_model/train_seta_latent.sh "$@"
