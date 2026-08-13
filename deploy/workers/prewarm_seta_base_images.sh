#!/usr/bin/env bash
# Resolve every base image used by SETA before formal training. Docker Hub
# official images use an explicit mirror; named registries retain their source.
# This avoids turning a shared-egress 429 or a late cold pull into missing
# rollout samples halfway through a baseline.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/benchmarks/seta_env}"
DOCKER_BASE_MIRROR_PREFIX="${DOCKER_BASE_MIRROR_PREFIX:-docker.m.daocloud.io/library}"
DOCKER_BASE_PREFETCH_TIMEOUT="${DOCKER_BASE_PREFETCH_TIMEOUT:-600}"

log() { printf '[seta-base-prewarm] %s\n' "$*"; }
[[ -d "${DATASET_DIR}" ]] || { log "ERROR: dataset not found: ${DATASET_DIR}"; exit 1; }
[[ "${DOCKER_BASE_MIRROR_PREFIX}" != *://* ]] \
  || { log "ERROR: mirror prefix must be a Docker reference without URL scheme"; exit 1; }
[[ "${DOCKER_BASE_PREFETCH_TIMEOUT}" =~ ^[0-9]+$ ]] \
  || { log "ERROR: DOCKER_BASE_PREFETCH_TIMEOUT must be an integer"; exit 1; }

mapfile -t base_images < <(
  find "${DATASET_DIR}" -mindepth 2 -maxdepth 2 -type f -name Dockerfile -print0 \
    | xargs -0 awk '
        # SETA Dockerfiles use uppercase Docker instructions. Matching the
        # literal token avoids treating Python heredoc lines such as
        # `from pathlib import Path` as base images.
        $1 == "FROM" {
          for (i = 2; i <= NF; i++) {
            if ($i ~ /^--/) continue
            print $i
            break
          }
        }
      ' \
    | sort -u
)

prefetched=0
cached=0
for image in "${base_images[@]}"; do
  [[ -n "${image}" && "${image}" != "scratch" ]] || continue
  canonical="${image}"
  case "${canonical}" in
    docker.io/library/*) canonical="${canonical#docker.io/library/}" ;;
    library/*) canonical="${canonical#library/}" ;;
  esac
  if docker image inspect "${canonical}" >/dev/null 2>&1; then
    cached=$((cached + 1))
    log "cached ${canonical}"
    continue
  fi
  if [[ "${canonical}" == */* ]]; then
    # Explicit registries/namespaces (currently GHCR) retain their source.
    mirrored="${canonical}"
  else
    mirrored="${DOCKER_BASE_MIRROR_PREFIX%/}/${canonical}"
  fi
  log "pull ${mirrored} -> ${canonical}"
  timeout "${DOCKER_BASE_PREFETCH_TIMEOUT}" docker pull "${mirrored}"
  if [[ "${mirrored}" != "${canonical}" ]]; then
    docker tag "${mirrored}" "${canonical}"
  fi
  docker image inspect "${canonical}" >/dev/null
  prefetched=$((prefetched + 1))
done

log "complete cached=${cached} prefetched=${prefetched}"
