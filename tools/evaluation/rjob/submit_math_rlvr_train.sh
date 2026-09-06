#!/usr/bin/env bash
# Submit the math DAPO training payload to a private narmodel RJob.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${HF_CKPT:?set HF_CKPT before submitting}"
: "${REF_LOAD:?set REF_LOAD before submitting}"
: "${RJOB_NAME:?set RJOB_NAME before submitting}"

RJOB_NAMESPACE="${RJOB_NAMESPACE:-ailab-narmodel}"
RJOB_GROUP="${RJOB_GROUP:-narmodel_gpu}"
RJOB_GPU="${RJOB_GPU:-4}"
RJOB_CPU="${RJOB_CPU:-50}"
RJOB_MEMORY="${RJOB_MEMORY:-560000}"
RJOB_PRIORITY="${RJOB_PRIORITY:-9}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
RJOB_MOUNTS="${RJOB_MOUNTS:-gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan gpfs://gpfs2/trustcyberdata:/mnt/shared-storage-gpfs2/trustcyberdata}"
RJOB_AUTO_DELETE="${RJOB_AUTO_DELETE:-720h}"

MATH_DATA_ROOT="${MATH_DATA_ROOT:-/mnt/shared-storage-user/puyuan/math_rlvr_data}"
TRAIN_DATASET="${TRAIN_DATASET:-aime-2025}"
REWARD_TYPE="${REWARD_TYPE:-math}"
RESPONSE_CAP="${RESPONSE_CAP:-32768}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-120}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES))}"
EVAL_DATASETS="${EVAL_DATASETS:-aime-2025,aime-2024}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
SEED="${SEED:-1}"
ACTOR_GPUS="${ACTOR_GPUS:-$((RJOB_GPU / 2))}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$((RJOB_GPU - ACTOR_GPUS))}"
NUM_GPUS="${NUM_GPUS:-${ACTOR_GPUS}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
COLOCATE="${COLOCATE:-0}"
TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
RUN_ID="${RUN_ID:-${RJOB_NAME}}"
PERSIST_ROOT="${PERSIST_ROOT:-/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/puyuan/lightrl}"
RUN_DIR="${RUN_DIR:-${PERSIST_ROOT}/runs/training/${RUN_ID}}"

POD_COMMAND=$(cat <<'EOS'
set -euo pipefail
cd /mnt/shared-storage-user/puyuan/code/LightRL
export PYTHONPATH="/mnt/shared-storage-user/puyuan/code/LightRL/Megatron-LM:/mnt/shared-storage-user/puyuan/code/LightRL:/mnt/shared-storage-user/puyuan/code/LightRL/slime:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/config"
echo "[math-dapo-rjob] job=${RJOB_NAME} run=${RUN_ID} train=${TRAIN_DATASET} reward=${REWARD_TYPE} cap=${RESPONSE_CAP} seed=${SEED}"
echo "[math-dapo-rjob] data_root=${MATH_DATA_ROOT} eval=${EVAL_DATASETS} actor=${ACTOR_GPUS} rollout=${ROLLOUT_GPUS} n=${N_SAMPLES} rollouts=${NUM_ROLLOUT}"
exec bash examples/training/train_qwen3_8b_dapo_math.sh
EOS
)

export SUBMIT_NAME="${RJOB_NAME}" SUBMIT_NAMESPACE="${RJOB_NAMESPACE}"
export SUBMIT_GROUP="${RJOB_GROUP}" SUBMIT_GPU="${RJOB_GPU}" SUBMIT_CPU="${RJOB_CPU}"
export SUBMIT_MEMORY="${RJOB_MEMORY}" SUBMIT_PRIORITY="${RJOB_PRIORITY}"
export SUBMIT_IMAGE="${RJOB_IMAGE}" SUBMIT_AUTO_DELETE="${RJOB_AUTO_DELETE}"
export SUBMIT_MOUNTS="${RJOB_MOUNTS}" SUBMIT_COMMAND="${POD_COMMAND}"
export POD_HF_CKPT="${HF_CKPT}" POD_REF_LOAD="${REF_LOAD}"
export POD_MATH_DATA_ROOT="${MATH_DATA_ROOT}" POD_TRAIN_DATASET="${TRAIN_DATASET}"
export POD_REWARD_TYPE="${REWARD_TYPE}" POD_RESPONSE_CAP="${RESPONSE_CAP}"
export POD_ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" POD_N_SAMPLES="${N_SAMPLES}"
export POD_NUM_ROLLOUT="${NUM_ROLLOUT}" POD_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export POD_EVAL_DATASETS="${EVAL_DATASETS}" POD_EVAL_N_SAMPLES="${EVAL_N_SAMPLES}"
export POD_EVAL_INTERVAL="${EVAL_INTERVAL}" POD_EVAL_TOP_P="${EVAL_TOP_P}"
export POD_SAVE_INTERVAL="${SAVE_INTERVAL}" POD_SEED="${SEED}"
export POD_NUM_GPUS="${NUM_GPUS}" POD_ACTOR_GPUS="${ACTOR_GPUS}"
export POD_ROLLOUT_GPUS="${ROLLOUT_GPUS}" POD_ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE}"
export POD_COLOCATE="${COLOCATE}" POD_TRAIN_BACKEND="${TRAIN_BACKEND}"
export POD_RUN_ID="${RUN_ID}" POD_RUN_DIR="${RUN_DIR}"

python3 - <<'PY'
import json
import os
import sys

from brainpp.rjob import Affinity, Container, Job, Metadata, PrivateMachine, Resources, RestartPolicy, RJobClient, Spec, Task, Template

env = os.environ
pod_env = {
    "RJOB_TASK_INDEX": "0", "RJOB_NAME": env["SUBMIT_NAME"],
    "HF_CKPT": env["POD_HF_CKPT"], "REF_LOAD": env["POD_REF_LOAD"],
    "MATH_DATA_ROOT": env["POD_MATH_DATA_ROOT"], "TRAIN_DATASET": env["POD_TRAIN_DATASET"],
    "REWARD_TYPE": env["POD_REWARD_TYPE"], "RESPONSE_CAP": env["POD_RESPONSE_CAP"],
    "ROLLOUT_BATCH_SIZE": env["POD_ROLLOUT_BATCH_SIZE"], "N_SAMPLES": env["POD_N_SAMPLES"],
    "NUM_ROLLOUT": env["POD_NUM_ROLLOUT"], "GLOBAL_BATCH_SIZE": env["POD_GLOBAL_BATCH_SIZE"],
    "EVAL_DATASETS": env["POD_EVAL_DATASETS"], "EVAL_N_SAMPLES": env["POD_EVAL_N_SAMPLES"],
    "EVAL_INTERVAL": env["POD_EVAL_INTERVAL"], "EVAL_TOP_P": env["POD_EVAL_TOP_P"],
    "SAVE_INTERVAL": env["POD_SAVE_INTERVAL"], "SEED": env["POD_SEED"],
    "NUM_GPUS": env["POD_NUM_GPUS"], "ACTOR_GPUS": env["POD_ACTOR_GPUS"],
    "ROLLOUT_GPUS": env["POD_ROLLOUT_GPUS"],
    "ROLLOUT_NUM_GPUS_PER_ENGINE": env["POD_ROLLOUT_NUM_GPUS_PER_ENGINE"],
    "COLOCATE": env["POD_COLOCATE"], "TRAIN_BACKEND": env["POD_TRAIN_BACKEND"],
    "RUN_ID": env["POD_RUN_ID"], "RUN_DIR": env["POD_RUN_DIR"],
    "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"), "WANDB_DIR": env["POD_RUN_DIR"],
}
resources = Resources(cpu=int(env["SUBMIT_CPU"]), gpu=int(env["SUBMIT_GPU"]), memory_in_mb=int(env["SUBMIT_MEMORY"]))
task = Task(
    replicas=1, restart_policy=RestartPolicy.Never, private_machine=PrivateMachine.Group,
    host_network=False, share_host_shm=True, share_host_nvme=False,
    mount_config=env["SUBMIT_MOUNTS"].split(),
    template=Template(affinity=Affinity(), containers=[Container(
        name="generated-task-0", image=env["SUBMIT_IMAGE"],
        command=["bash", "-lc", env["SUBMIT_COMMAND"]], resources=resources, requests=resources,
        environments=pod_env,
    )]),
)
job = Job(
    metadata=Metadata(name=env["SUBMIT_NAME"], charged_group=env["SUBMIT_GROUP"], annotations={
        "rjob.brainpp.cn/job-command": "bash tools/evaluation/rjob/submit_math_rlvr_train.sh",
        "volcano.brainpp.cn/priority": env["SUBMIT_PRIORITY"], "lightrl.puyuan.cn/topology": "math-dapo",
    }),
    spec=Spec(preemptible="no", backoff_limit=1, host_network=False,
              auto_delete_duration=env["SUBMIT_AUTO_DELETE"], tasks={"generated-task-0": task}),
)
if os.environ.get("RJOB_DRY_RUN", "0") == "1":
    print(json.dumps(job.to_k8s_template(env["SUBMIT_NAMESPACE"]), indent=2))
    sys.exit(0)
print(RJobClient().submit(job, no_packaging=True) or env["SUBMIT_NAME"])
PY
