#!/usr/bin/env bash
set -euo pipefail

# Prepare the official ALFWorld benchmark assets outside the training tree.
# Keep the downloader lightweight: the training image supplies Torch and the
# environment dependencies, while this script only installs the ALFWorld
# package metadata and invokes its official downloader.
OUT_DIR="${ALFWORLD_DATA_DIR:-${PWD}/datasets/alfworld}"
VENV="${ALFWORLD_VENV:-/mnt/shared-storage-user/puyuan/runtime/alfworld-venv}"
PYTHON="${ALFWORLD_PYTHON:-${VENV}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${PYTHON_BIN:-python3}"
fi
export TMPDIR="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/shared-storage-user/puyuan/runtime/pip-cache}"
export HOME="${ALFWORLD_HOME:-/mnt/shared-storage-user/puyuan/runtime/alfworld-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "${OUT_DIR}" "${PIP_CACHE_DIR}" "${HOME}" "${XDG_CACHE_HOME}"
export ALFWORLD_DATA="${OUT_DIR}"

if ! "${PYTHON}" -c 'import alfworld' 2>/dev/null; then
  "${PYTHON}" -m pip install --disable-pip-version-check --no-deps --quiet alfworld
fi

DOWNLOAD_BIN="$(dirname "${PYTHON}")/alfworld-download"
if [[ -x "${DOWNLOAD_BIN}" ]]; then
  "${DOWNLOAD_BIN}" -f
elif command -v alfworld-download >/dev/null 2>&1; then
  alfworld-download -f
else
  echo "alfworld-download is unavailable in ${PYTHON}" >&2
  exit 2
fi
