# LightRL 重构后人工验证步骤

本文所有命令均供人工执行。本轮自动流程没有运行 Docker、rjob、训练、推理或
pytest。

## 共同前置检查

```bash
export LIGHTRL_REPO=/mnt/shared-storage-user/puyuan/code/LightRL
cd "$LIGHTRL_REPO"

git status --short --branch
git log -6 --oneline
```

预期输出中包含以下三个连续的结构提交：

```text
a71ba399 refactor: flatten the private trajectory store
4f1b58b3 refactor: move worker URL template out of package
20b8a0bf refactor: merge trajectory policy into its store
```

仓库当前已有若干与本轮无关的 untracked 文件；不要用
`git clean`、`git reset --hard` 或其他批量清理命令。

## 一键入口

开发机启动 Docker daemon 和 LightRL CPU pool server：

```bash
cd "$LIGHTRL_REPO"
bash tools/validation/start_local_docker_server.sh
```

脚本以前台方式运行；保持这个终端开启。它会打印供 rjob 使用的
`WORKER_URLS=http://<CPU_WORKER_IP>:18081`。

进入已经 Running 的 4-GPU rjob 后，一键执行完整验证：

```bash
cd "$LIGHTRL_REPO"

WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
  bash tools/validation/run_4gpu_seta_dapo_validation.sh
```

该脚本依次检查 4-GPU 拓扑、CPU/内存/磁盘、Python 静态编译、imports、相关
pytest、CLI dry-run、worker `/healthz`，然后真实运行 task 307 的 3 个
SETA+DAPO rollout，并检查 fatal error signature 和 `metrics.jsonl`。增加验证
长度时只需覆盖：

```bash
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
NUM_ROLLOUT=10 \
  bash tools/validation/run_4gpu_seta_dapo_validation.sh
```

仅重跑训练、跳过已经通过的静态和 pytest 阶段：

```bash
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
RUN_STATIC_CHECKS=0 \
RUN_IMPORT_SMOKE=0 \
RUN_RELEVANT_TESTS=0 \
RUN_CLI_DRY_RUN=0 \
  bash tools/validation/run_4gpu_seta_dapo_validation.sh
```

以下章节保留等价的分步命令和排障方法。

## 方案一：本地开发机 + Docker server

### 1. 检查并启动宿主机 Docker daemon

```bash
timeout 10 docker info >/dev/null 2>&1 || sudo systemctl start docker
timeout 10 docker info
docker compose version
```

预期 `docker info` 返回 Server 信息，Compose 为 V2。若 daemon/socket/代理
异常，按仓库维护的恢复入口执行：

```bash
cd "$LIGHTRL_REPO"
sudo env \
  DOCKER_DATA_ROOT=/data \
  PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
  bash deploy/workers/fix_dockerd_and_proxy.sh
```

若本机 Docker data root 不是 `/data`，把它替换为
`docker info --format '{{.DockerRootDir}}'` 的结果。详细排障见
`docs/operations/cpu_workers.md`。

### 2. 启动开发容器

仓库没有维护第二套 LightRL service Dockerfile；现有约定是使用 rjob 同款
`rft` 镜像，并把宿主 Docker socket 和共享目录以相同绝对路径挂入。这样
pool server 创建 SETA compose 容器时，宿主路径保持一致。

```bash
export LIGHTRL_IMAGE=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408

docker pull "$LIGHTRL_IMAGE"
docker run --rm -it \
  --name lightrl-dev \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /mnt/shared-storage-user/puyuan:/mnt/shared-storage-user/puyuan \
  -w "$LIGHTRL_REPO" \
  -e ENV_SERVER_PORT=18081 \
  -e WORKER_MAX_TASKS=2 \
  -e WORKER_MAX_RUNS_PER_TASK=2 \
  -e WORKER_MAX_CONCURRENT_BUILDS=1 \
  -e WORKER_MAX_CONCURRENT_RESETS=2 \
  "$LIGHTRL_IMAGE" bash
```

使用 `--network host`，所以不再同时使用 `-p 18081:18081`。若本机安全策略
禁止挂 Docker socket，不要改用 privileged DinD；改为直接在宿主机运行下一节
的 pool server。

### 3. 容器内做 import/配置冒烟

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL

python3 -m compileall -q agentic_rl tests/agentic_rl tools

python3 - <<'PY'
from agentic_rl import REGISTRY, load_config
from agentic_rl.rollout import trajectory_store

assert callable(trajectory_store._trajectory_save_decision)
assert callable(trajectory_store._save_rollout_artifacts)
assert "dapo" in REGISTRY.names("algorithms")
cfg = load_config("configs/experiment/qwen3_8b_seta_dapo.yaml")
assert cfg["algorithm"]["base"]["name"] == "dapo"
assert cfg["environment"]["name"] == "seta"
print("IMPORT_SMOKE_OK")
PY

NUM_GPUS=4 \
ACTOR_GPUS=2 \
ROLLOUT_GPUS=2 \
TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/train_qwen3_8b_seta_dapo.sh --dry-run
```

预期出现 `IMPORT_SMOKE_OK`；dry-run 输出 JSON，`command` 指向
`agentic_rl/backends/slime/runtime/train.sh`，且不启动 Ray/训练。

### 4. 启动最小 CPU Docker worker

在开发容器的第一个 shell 中执行：

```bash
cd "$LIGHTRL_REPO"

ENV_SERVER_PORT=18081 \
WORKER_MAX_TASKS=2 \
WORKER_MAX_RUNS_PER_TASK=2 \
WORKER_MAX_CONCURRENT_BUILDS=1 \
WORKER_MAX_CONCURRENT_RESETS=2 \
SKIP_PREFLIGHT_CLEANUP=1 \
bash deploy/workers/run_pool_server_pu_v2.sh
```

另开宿主 shell：

```bash
curl --noproxy '*' --fail --max-time 5 http://127.0.0.1:18081/healthz
curl --noproxy '*' --fail --max-time 5 http://127.0.0.1:18081/status \
  | python3 -m json.tool
```

预期 `/healthz` 为成功状态，`/status` 能看到 pool capacity。若失败：

- `docker info` timeout：先修复 daemon，不要重复启动 pool server。
- `permission denied /var/run/docker.sock`：确认容器以 root 运行且 socket 已挂载。
- 端口占用：用 `ss -ltnp | grep ':18081 '` 找已有服务；不要同时启动两个实例。
- base image/apt timeout：检查 `/etc/seta_build_proxy.env`，再运行
  `deploy/workers/fix_dockerd_and_proxy.sh`。
- 磁盘 guard 拒绝启动：检查 Docker data root 的空间/inode，不要简单关闭 guard。

### 5. 人工运行相关测试

这一步会执行测试，因此只在人工确认后运行：

```bash
cd "$LIGHTRL_REPO"

python3 -m pytest -q \
  tests/agentic_rl/test_rollout_log_metrics.py \
  tests/agentic_rl/test_harness_option_routing.py \
  tests/agentic_rl/test_agent_runner_harness_option.py
```

判定通过：pytest 全部通过，且日志中没有 legacy trajectory import 的
`ModuleNotFoundError`。

## 方案二：已申请的 4-GPU SSH rjob

### 1. 等待并进入现有任务

任务 metadata name：

```text
lightrl-manual-4g-20260731-74537606
```

在开发机人工查询（不要重新提交同名任务）：

```bash
rjob get lightrl-manual-4g-20260731-74537606
```

只有状态为 `Running` 后才 SSH。当前已知副本名为：

```bash
ssh -CAXY \
  lightrl-manual-4g-20260731-74537606-d8cm2.puyuan+root.ailab-narmodel.pod@h.pjlab.org.cn
```

若调度后副本名变化，以 `rjob get` 输出替换
`lightrl-manual-4g-20260731-74537606-d8cm2`。该任务已按 4 GPU、SSH、
`sleep infinity` 申请，不需要再次提交。

### 2. 确认共享代码版本

rjob 挂载同一共享目录，因此通常不需要复制或 `git pull`：

```bash
export LIGHTRL_REPO=/mnt/shared-storage-user/puyuan/code/LightRL
cd "$LIGHTRL_REPO"

git status --short --branch
git log -3 --oneline
nvidia-smi
```

应看到四张 GPU 和上述三个结构提交。如果 rjob 使用的是另一个 clone，先在
开发机把目标分支推到 `<REMOTE>`，再在 rjob 的干净 clone 中执行：

```bash
git fetch <REMOTE> refactor/lightrl
git switch refactor/lightrl
git merge --ff-only <REMOTE>/refactor/lightrl
```

有未提交改动时不要执行 merge；先人工确认归属。

### 3. 静态和 import 冒烟

```bash
cd "$LIGHTRL_REPO"

python3 -m compileall -q agentic_rl tests/agentic_rl tools

python3 - <<'PY'
from agentic_rl import REGISTRY, load_config
from agentic_rl.rollout import trajectory_store

assert callable(trajectory_store._trajectory_save_decision)
assert "dapo" in REGISTRY.names("algorithms")
cfg = load_config("configs/experiment/qwen3_8b_seta_dapo.yaml")
assert cfg["cluster"]["num_gpus"] > 0
print("RJOB_IMPORT_SMOKE_OK")
PY

NUM_GPUS=4 \
ACTOR_GPUS=2 \
ROLLOUT_GPUS=2 \
TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/train_qwen3_8b_seta_dapo.sh --dry-run
```

预期 `RJOB_IMPORT_SMOKE_OK`，dry-run 不启动训练且解析到 4-GPU override。

### 4. 运行 4-GPU SETA+DAPO 最小验证

先把 `<CPU_WORKER_IP>` 替换为可达、`/healthz` 正常的 SETA worker：

```bash
curl --noproxy '*' --fail --max-time 5 \
  http://<CPU_WORKER_IP>:18081/healthz
```

然后执行仓库内固定的小规模脚本：

```bash
cd "$LIGHTRL_REPO"

RUN_ID="manual-refactor-seta-dapo-$(date +%Y%m%d-%H%M%S)" \
WORKER_URLS="http://<CPU_WORKER_IP>:18081" \
NUM_GPUS=4 \
ACTOR_GPUS=2 \
ROLLOUT_GPUS=2 \
TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
NUM_ROLLOUT=3 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES=2 \
WANDB_MODE=disabled \
MAX_CKPT_KEEP=0 \
bash tools/rjob/run_seta_dapo_refactor_smoke.sh
```

trainer rjob 本身不要求本地 Docker daemon；SETA task containers 由
`WORKER_URLS` 指向的 CPU worker 管理。

通过标准：

- 四张 GPU 被识别，actor/rollout 各使用两张。
- task 307 成功 reset，连续完成脚本配置的 3 个 rollout。
- 日志中没有 `ModuleNotFoundError`、旧 trajectory 路径错误、
  `_get_terminal_save_dir NameError`、Ray actor crash 或 CUDA OOM。
- 至少产生一个可训练 batch；若进入 optimizer step，loss/reward/grad norm 为
  有限数。
- 日志位于
  `runs/<RUN_ID>/rjob_outer.log` 和 `runs/<RUN_ID>/logs/`。

快速检查：

```bash
RUN_DIR="$LIGHTRL_REPO/runs/<RUN_ID>"

rg -n 'Traceback|ModuleNotFoundError|NameError|CUDA out of memory|RayActorError' \
  "$RUN_DIR" || true
rg -n 'reward|loss|grad.norm|trainable|rollout' "$RUN_DIR/logs" | tail -80
```

常见问题：

- `/healthz` 不通：先修 CPU worker/网络，不要归因于 Python 重构。
- `/reset` 500、Docker build exit 17：在 CPU worker 执行
  `deploy/workers/docker_worker_doctor.sh diagnose`。
- `No module named agentic_rl...`：确认 rjob 进入的是共享目录当前 commit，并检查
  `PYTHONPATH` 没有指向旧 clone。
- GPU 数量/拓扑不符：检查 `nvidia-smi` 和 dry-run 输出中的
  `NUM_GPUS/ACTOR_GPUS/ROLLOUT_GPUS/TP_SIZE`。
- OOM：保持 smoke 的 batch/sample/token 设置；不要直接切换完整 8-GPU recipe。
- task 307 自身 worker cache 失败：更换一个已知可构建 task，并通过
  `SMOKE_TASK_NAME=<id>` 覆盖，保留其余参数不变。
