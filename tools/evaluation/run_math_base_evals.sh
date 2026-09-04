#!/usr/bin/env bash
# Run the four Math RLVR baseline suites against one explicitly served model.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL:?set MODEL to the served model name (no implicit checkpoint)}"
DATA_ROOT="${MATH_DATA_ROOT:-${ROOT}/benchmarks/math}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/eval_results}"
PYTHON="${PYTHON:-python3}"
N="${N:-16}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
CONCURRENCY="${CONCURRENCY:-32}"
REWARD_TYPE="${REWARD_TYPE:-math}"
if awk -v p="${TOP_P}" 'BEGIN { exit !(p == 1.0) }'; then
  TAG="${TAG:-T${TEMPERATURE}}"
else
  TAG="${TAG:-T${TEMPERATURE}_p${TOP_P}}"
fi
DATASETS="${DATASETS:-aime-2025 aime-2024 amc23 math-500}"

mkdir -p "${OUTPUT_DIR}"
for dataset in ${DATASETS}; do
  data_path="${dataset}"
  if [[ -f "${dataset}" ]]; then data_path="${dataset}"; fi
  echo "[math-eval] dataset=${dataset} model=${MODEL} n=${N} cap=${MAX_TOKENS} reward=${REWARD_TYPE}"
  "${PYTHON}" "${ROOT}/tools/evaluation/eval_math.py" \
    --data "${data_path}" --data-root "${DATA_ROOT}" --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" --tag "${TAG}" --n "${N}" \
    --temperature "${TEMPERATURE}" --top-p "${TOP_P}" \
    --max-tokens "${MAX_TOKENS}" --concurrency "${CONCURRENCY}" \
    --reward-type "${REWARD_TYPE}"
done
