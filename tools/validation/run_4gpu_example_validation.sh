#!/usr/bin/env bash
# Validate either examples/train_qwen3_8b_seta_dive_po.sh or mixed DAPO.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/LightRL}"
EXPERIMENT="${EXPERIMENT:-dive_po}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-lightrl-4gpu-${EXPERIMENT}-validation-${RUN_TIMESTAMP}}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
WORKER_URLS="${WORKER_URLS:-http://100.96.26.133:18081}"
LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"

log() { printf '[4gpu-%s] %s\n' "${EXPERIMENT}" "$*"; }
die() { log "ERROR: $*"; exit 1; }

cd "${REPO_ROOT}"
[[ -x "${LIGHTRFT_PY312_BIN}/python3" ]] || die "missing Python environment: ${LIGHTRFT_PY312_BIN}"
export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/slime:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RUN_ID RUN_TIMESTAMP WORKER_URLS EXPERIMENT
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/validation_driver.log") 2>&1

gpu_count="$(nvidia-smi -L | sed -n '/^GPU /p' | wc -l)"
[[ "${gpu_count}" == "${EXPECTED_GPUS}" ]] || die "expected ${EXPECTED_GPUS} GPUs, found ${gpu_count}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
free -h
df -h / "${REPO_ROOT}"

first_worker="${WORKER_URLS%%,*}"
curl --noproxy '*' --silent --show-error --fail --max-time 10 "${first_worker%/}/healthz"
printf '\n'

python3 -m compileall -q agentic_rl tools
git diff --check

case "${EXPERIMENT}" in
  dive_po)
    bash examples/train_qwen3_8b_seta_dive_po.sh --dry-run >/dev/null
    ;;
  mixed)
    bash examples/train_qwen3_8b_mixed_dapo.sh --dry-run >/dev/null
    ;;
  *)
    die "EXPERIMENT must be dive_po or mixed"
    ;;
esac

bash tools/rjob/run_4gpu_example_smoke.sh

python3 - "${RUN_DIR}" "${EXPERIMENT}" <<'PY'
import ast
import json
import math
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
experiment = sys.argv[2]
metrics_path = run_dir / "logs/metrics.jsonl"
train_log_path = run_dir / "logs/train.log"
if not metrics_path.is_file() or not train_log_path.is_file():
    raise SystemExit("missing metrics.jsonl or train.log")
records = [
    json.loads(line)
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(records) < 2:
    raise SystemExit(f"expected multiple rollout metric records, found {len(records)}")
train_records = []
for match in re.finditer(
    r"train-step\s+\d+:\s+(\{[^\n]+\})",
    train_log_path.read_text(encoding="utf-8", errors="replace"),
):
    try:
        train_records.append(ast.literal_eval(match.group(1)))
    except (SyntaxError, ValueError):
        pass
nonzero = [
    record
    for record in train_records
    if math.isfinite(float(record.get("train/loss", float("nan"))))
    and math.isfinite(float(record.get("train/grad_norm", float("nan"))))
    and abs(float(record["train/loss"])) > 0
    and abs(float(record["train/grad_norm"])) > 0
]
if not nonzero:
    raise SystemExit("no finite non-zero actor update")

if experiment == "dive_po":
    intrinsic = [
        record.get("intrinsic/fused")
        for record in records
        if record.get("intrinsic/fused") is not None
    ]
    if not intrinsic:
        raise SystemExit("DIVE-PO emitted no intrinsic/fused metrics")
else:
    datasets = {str(record.get("dataset")) for record in records}
    expected = {"seta", "agent_safetybench", "agentharm"}
    missing = expected - datasets
    if missing:
        raise SystemExit(f"mixed smoke missing dataset metrics: {sorted(missing)}")

environment_outputs = run_dir / "environment_outputs"
if not environment_outputs.is_dir():
    raise SystemExit(f"missing structured environment outputs: {environment_outputs}")
print(
    "EXAMPLE_VALIDATION_OK",
    f"experiment={experiment}",
    f"metric_records={len(records)}",
    f"actor_train_steps={len(train_records)}",
    f"nonzero_updates={len(nonzero)}",
)
print("LAST_METRICS_RECORD", json.dumps(records[-1], sort_keys=True))
PY

log "validation passed; artifacts=${RUN_DIR}"
