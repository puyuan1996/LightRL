#!/usr/bin/env bash
# Convert a Megatron torch_dist training checkpoint to HF format on narmodel RJob.
set -euo pipefail

ROOT="${LIGHTRL_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RJOB_CONFIG_FILE="${RJOB_CONFIG_FILE:-${ROOT}/local/rjob/rjob.env}"
if [[ -f "${RJOB_CONFIG_FILE}" ]]; then source "${RJOB_CONFIG_FILE}"; fi
: "${INPUT_DIR:?set INPUT_DIR to a torch_dist checkpoint directory}"
: "${ORIGIN_HF_DIR:?set ORIGIN_HF_DIR to the base HF model directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the converted HF output directory}"
: "${RJOB_NAME:?set RJOB_NAME before submitting}"

: "${RJOB_NAMESPACE:?set RJOB_NAMESPACE (or RJOB_CONFIG_FILE) before submitting}"
: "${RJOB_GROUP:?set RJOB_GROUP (or RJOB_CONFIG_FILE) before submitting}"
RJOB_GPU="${RJOB_GPU:-4}"
RJOB_CPU="${RJOB_CPU:-50}"
RJOB_MEMORY="${RJOB_MEMORY:-560000}"
RJOB_PRIORITY="${RJOB_PRIORITY:-9}"
: "${RJOB_IMAGE:?set RJOB_IMAGE (or RJOB_CONFIG_FILE) before submitting}"
: "${RJOB_MOUNTS:?set RJOB_MOUNTS (or RJOB_CONFIG_FILE) before submitting}"
RJOB_AUTO_DELETE="${RJOB_AUTO_DELETE:-720h}"
VOCAB_SIZE="${VOCAB_SIZE:-151936}"

POD_COMMAND=$(cat <<'EOS'
set -euo pipefail
cd "${LIGHTRL_ROOT}"
export PYTHONPATH="${LIGHTRL_ROOT}/Megatron-LM:${LIGHTRL_ROOT}/slime:${LIGHTRL_ROOT}:${PYTHONPATH:-}"
mkdir -p "$(dirname -- "${OUTPUT_DIR}")"
echo "[math-rlvr-convert] input=${INPUT_DIR} origin=${ORIGIN_HF_DIR} output=${OUTPUT_DIR} vocab=${VOCAB_SIZE}"
exec python3 slime/tools/convert_torch_dist_to_hf.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --origin-hf-dir "${ORIGIN_HF_DIR}" \
  --vocab-size "${VOCAB_SIZE}"
EOS
)

export SUBMIT_NAME="${RJOB_NAME}" SUBMIT_NAMESPACE="${RJOB_NAMESPACE}"
export SUBMIT_GROUP="${RJOB_GROUP}" SUBMIT_GPU="${RJOB_GPU}" SUBMIT_CPU="${RJOB_CPU}"
export SUBMIT_MEMORY="${RJOB_MEMORY}" SUBMIT_PRIORITY="${RJOB_PRIORITY}"
export SUBMIT_IMAGE="${RJOB_IMAGE}" SUBMIT_AUTO_DELETE="${RJOB_AUTO_DELETE}"
export SUBMIT_MOUNTS="${RJOB_MOUNTS}" SUBMIT_COMMAND="${POD_COMMAND}"
export POD_LIGHTRL_ROOT="${LIGHTRL_ROOT}" POD_INPUT_DIR="${INPUT_DIR}" POD_ORIGIN_HF_DIR="${ORIGIN_HF_DIR}"
export POD_OUTPUT_DIR="${OUTPUT_DIR}" POD_VOCAB_SIZE="${VOCAB_SIZE}"

python3 - <<'PY'
import json
import os
import sys

from brainpp.rjob import Affinity, Container, Job, Metadata, PrivateMachine, Resources, RestartPolicy, RJobClient, Spec, Task, Template

env = os.environ
pod_env = {
    "RJOB_TASK_INDEX": "0", "LIGHTRL_ROOT": env["POD_LIGHTRL_ROOT"],
    "RJOB_NAME": env["SUBMIT_NAME"],
    "INPUT_DIR": env["POD_INPUT_DIR"],
    "ORIGIN_HF_DIR": env["POD_ORIGIN_HF_DIR"],
    "OUTPUT_DIR": env["POD_OUTPUT_DIR"],
    "VOCAB_SIZE": env["POD_VOCAB_SIZE"],
}
resources = Resources(
    cpu=int(env["SUBMIT_CPU"]),
    gpu=int(env["SUBMIT_GPU"]),
    memory_in_mb=int(env["SUBMIT_MEMORY"]),
)
task = Task(
    replicas=1,
    restart_policy=RestartPolicy.Never,
    private_machine=PrivateMachine.Group,
    host_network=False,
    share_host_shm=True,
    share_host_nvme=False,
    mount_config=env["SUBMIT_MOUNTS"].split(),
    template=Template(
        affinity=Affinity(),
        containers=[Container(
            name="generated-task-0",
            image=env["SUBMIT_IMAGE"],
            command=["bash", "-lc", env["SUBMIT_COMMAND"]],
            resources=resources,
            requests=resources,
            environments=pod_env,
        )],
    ),
)
job = Job(
    metadata=Metadata(
        name=env["SUBMIT_NAME"],
        charged_group=env["SUBMIT_GROUP"],
        annotations={
            "rjob.brainpp.cn/job-command": "bash local/rjob/submit_math_rlvr_convert.sh",
            "volcano.brainpp.cn/priority": env["SUBMIT_PRIORITY"],
            "lightrl.puyuan.cn/topology": "math-rlvr-checkpoint-convert",
        },
    ),
    spec=Spec(
        preemptible="no",
        backoff_limit=1,
        host_network=False,
        auto_delete_duration=env["SUBMIT_AUTO_DELETE"],
        tasks={"generated-task-0": task},
    ),
)
if os.environ.get("RJOB_DRY_RUN", "0") == "1":
    print(json.dumps(job.to_k8s_template(env["SUBMIT_NAMESPACE"]), indent=2))
    sys.exit(0)
print(RJobClient().submit(job, no_packaging=True) or env["SUBMIT_NAME"])
PY
