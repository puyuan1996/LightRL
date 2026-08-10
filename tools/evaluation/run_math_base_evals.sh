#!/usr/bin/env bash
# Math eval sweep against an already-running sglang server (see launch_sglang_math.sh).
#
# Sequential on purpose: one dataset at a time keeps the server's scheduler
# saturated without queueing tens of thousands of requests at once.
#
# Required:
#   MODEL           the served model name, passed straight to eval_math.py
#
# Optional:
#   TAG=T1.0        label written into the output filenames
#   PORT=30000      sglang port
#   CONCURRENCY=128
#   MAX_TOKENS=32768   see docs/evaluation/math_rlvr.md on why 8192 is too small
#   MATH_DATA_ROOT  default <repo>/benchmarks/math
#   PYTHON          interpreter to use (default: python)
#   DATASETS        space-separated subset of: aime-2025 aime-2024 amc23 math-500
set -uo pipefail

: "${MODEL:?MODEL is required (served model name)}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export MATH_DATA_ROOT="${MATH_DATA_ROOT:-${REPO}/benchmarks/math}"

PYTHON="${PYTHON:-python}"
PORT="${PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-128}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TAG="${TAG:-T1.0}"
DATASETS="${DATASETS:-aime-2025 aime-2024 amc23 math-500}"

mkdir -p "${MATH_DATA_ROOT}/logs" "${MATH_DATA_ROOT}/eval_results"

run() {  # name, data, n, temperature, top_p, tag
  local name="$1" data="$2" n="$3" temp="$4" topp="$5" tag="$6"
  echo "=== [$(date +%H:%M:%S)] START ${name} n=${n} T=${temp} top_p=${topp} tag=${tag} ==="
  timeout 14400 "${PYTHON}" "${SCRIPT_DIR}/eval_math.py" \
    --data "${data}" --n "${n}" --model "${MODEL}" --port "${PORT}" \
    --temperature "${temp}" --top-p "${topp}" --max-tokens "${MAX_TOKENS}" \
    --tag "${tag}" --concurrency "${CONCURRENCY}" 2>&1 | tail -45
  echo "=== [$(date +%H:%M:%S)] DONE ${name} (exit=$?) ==="
}

for ds in ${DATASETS}; do
  case "${ds}" in
    # k=16 on the small competition sets. T=1.0/top_p=1.0 matches slime's
    # in-training eval defaults, so these are directly comparable to the eval
    # curve a training run emits.
    aime-2025) run "AIME2025" aime-2025/aime-2025.jsonl 16 1.0 1.0 "${TAG}" ;;
    aime-2024) run "AIME2024" aime-2024/aime-2024.jsonl 16 1.0 1.0 "${TAG}" ;;
    amc23)     run "AMC23"    amc23/amc23.jsonl         16 1.0 1.0 "${TAG}" ;;
    # k=4 on MATH-500: 500 problems, so k=16 would cost more than it informs.
    # Note rm_type=dapo cannot score ~30% of this set (non-integer answers);
    # eval_math.py reports that as strict_scoring_error_rate rather than crashing.
    math-500)  run "MATH-500" math-500/math-500.jsonl    4 1.0 1.0 "${TAG}" ;;
    *) echo "[WARN] unknown dataset '${ds}', skipping" >&2 ;;
  esac
done
echo "=== [$(date +%H:%M:%S)] ALL EVALS COMPLETE ==="
