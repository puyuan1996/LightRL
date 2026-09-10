#!/usr/bin/env bash
set -euo pipefail

# Prepare the official ALFWorld benchmark assets outside the training tree.
# ALFWorld itself remains an environment dependency and is installed by the
# runtime image; this script only downloads the task data and config.
OUT_DIR="${ALFWORLD_DATA_DIR:-${PWD}/datasets/alfworld}"
mkdir -p "${OUT_DIR}"
python3 -m pip install --upgrade --quiet alfworld
ALFWORLD_DATA="${OUT_DIR}" python3 - <<'PY'
import alfworld
from pathlib import Path
root = Path(__import__('os').environ['ALFWORLD_DATA'])
print(f"ALFWorld data prepared at {root}")
PY
