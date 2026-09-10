#!/usr/bin/env bash
set -euo pipefail

# Prepare the official ALFWorld benchmark assets outside the training tree.
# ALFWorld itself remains an environment dependency and is installed by the
# runtime image; this script only downloads the task data and config.
OUT_DIR="${ALFWORLD_DATA_DIR:-${PWD}/datasets/alfworld}"
mkdir -p "${OUT_DIR}"
python3 -c 'import alfworld' 2>/dev/null || python3 -m pip install --upgrade --quiet alfworld
export ALFWORLD_DATA="${OUT_DIR}"
if command -v alfworld-download >/dev/null 2>&1; then
  alfworld-download --data-dir "${OUT_DIR}"
else
  python3 - <<'PY'
import alfworld
from pathlib import Path
root = Path(__import__('os').environ['ALFWORLD_DATA'])
root.mkdir(parents=True, exist_ok=True)
print(f"ALFWorld package is installed; run `alfworld-download --data-dir {root}` in an image that provides the downloader")
PY
fi
