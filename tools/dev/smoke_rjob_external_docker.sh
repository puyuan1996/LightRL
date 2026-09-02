#!/usr/bin/env bash
# Allocate one short-lived GPU RJob and test a Docker daemon exposed elsewhere.
# This is deliberately a diagnostic: it does not run/remove containers.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
DEV_IP="${POD_IP:-100.98.75.44}"
DOCKER_ENDPOINT="${DOCKER_ENDPOINT:-tcp://${DEV_IP}:2375}"
RJOB_NAME="${RJOB_NAME:-lightrl-docker-external-smoke-$(date +%Y%m%d-%H%M%S)}"
CHARGED_GROUP="${CHARGED_GROUP:-narmodel_gpu}"

cat >&2 <<EOF
[external-docker-smoke] RJob=${RJOB_NAME}
[external-docker-smoke] endpoint=${DOCKER_ENDPOINT}
[external-docker-smoke] resources=1GPU/4CPU/16000MiB
EOF

# DOCKER_HOST only configures the client.  The command exits non-zero when the
# endpoint is absent, which makes a stale address immediately visible.
read -r -d '' DIAGNOSTIC <<'EOS' || true
set +e
echo "worker=$(hostname) ip=$(hostname -I 2>/dev/null) uid=$(id -u)"
awk '/CapEff|CapBnd|Seccomp/ {printf "%s=%s ", $1, $2} END {print ""}' /proc/self/status
for x in docker dockerd docker-compose; do
  printf '%s=' "$x"
  command -v "$x" || true
done
ls -l /var/run/docker.sock /run/docker.sock 2>&1
echo "DOCKER_HOST=${DOCKER_HOST:-unset}"

rc=127
if command -v docker >/dev/null 2>&1; then
  timeout 10 docker info \
    --format 'server={{.ServerVersion}} root={{.DockerRootDir}} driver={{.Driver}}' \
    2>&1
  rc=$?
fi
echo "docker_info_rc=${rc}"
exit "${rc}"
EOS

exec /kubebrain/rlaunch \
  --request-timeout=120s \
  --image-check=false \
  --max-wait-duration="${MAX_WAIT_DURATION:-10m}" \
  --worker-garbage-collection-time="${GC_TIME:-15m}" \
  --memory=16000 --cpu=4 --gpu=1 \
  --charged-group="${CHARGED_GROUP}" \
  --private-machine=yes \
  --custom-resources=brainpp.cn/fuse=1 \
  --image="${IMAGE}" \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel \
  --env="DOCKER_HOST=${DOCKER_ENDPOINT}" \
  --env="DOCKER_TLS_VERIFY=${DOCKER_TLS_VERIFY:-0}" \
  ${RJOB_PREDICT_ONLY:+--predict-only} \
  -- bash -lc "${DIAGNOSTIC}"
