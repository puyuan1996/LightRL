#!/usr/bin/env bash
# Single-seed policy-level A/B for DAPO versus DAPO + latent-WM auxiliary loss.
# The latent-WM arm needs a frozen target provider that returns one target latent
# per rollout sample (see --world-model-target-provider-path).
set -euo pipefail

ARM="${ARM:-baseline}"
case "${ARM}" in
  baseline)
    _wm_args=(
      --world-model-loss-coef 0
    )
    ;;
  latent_wm)
    : "${WORLD_MODEL_TARGET_PROVIDER_PATH:?set WORLD_MODEL_TARGET_PROVIDER_PATH=package.module:function for ARM=latent_wm}"
    _wm_args=(
      --world-model-enable
      --world-model-backprop-to-llm
      --world-model-loss-coef "${WORLD_MODEL_LOSS_COEF:-0.01}"
      --world-model-target-provider-path "${WORLD_MODEL_TARGET_PROVIDER_PATH}"
      --world-model-latent-dim "${WORLD_MODEL_LATENT_DIM:-128}"
      --world-model-policy-projection "${WORLD_MODEL_POLICY_PROJECTION:-hash}"
    )
    ;;
  *)
    echo "ARM must be baseline or latent_wm" >&2
    exit 2
    ;;
esac

export SEED="${SEED:-1234}"
export RUN_ID="${RUN_ID:-seta-policy-${ARM}-seed${SEED}-$(date +%Y%m%d-%H%M%S)}"
_joined="${EXTRA_ALGO_ARGS:-} --seed ${SEED}"
for _arg in "${_wm_args[@]}"; do _joined+=" ${_arg}"; done
export EXTRA_ALGO_ARGS="${_joined}"

exec bash examples/training/train_qwen3_8b_seta_dapo.sh "$@"
