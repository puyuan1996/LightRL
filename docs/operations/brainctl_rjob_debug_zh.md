# 使用 brainctl exec 调试 LightRL rjob

本文记录已经实际验证可用的 `brainctl exec` 操作，以及 SETA + DAPO 远程
调试中最常用的排障方法。

## 1. 工作原理

RJob 包含 Job 和实际运行的 Replica/Pod 两层。`brainctl exec` 通过集群控制面
在指定 Replica 的容器里创建进程，原理类似 `kubectl exec`：

- 不依赖 Pod 开放 SSH 端口；
- 不需要容器内配置 SSH key；
- 必须使用实际 Replica 名，而不是只写 RJob 名。

本文示例：

```text
namespace: ailab-narmodel
rjob:     lightrl-manual-4g-20260731-74537606
replica:  lightrl-manual-4g-20260731-74537606-d8cm2
```

Replica 在任务重建后可能变化，应以网页或 `brainctl/rjob get` 的当前结果为准。

## 2. 常用命令

进入交互式 shell：

```bash
brainctl -n ailab-narmodel \
  --request-timeout=120s \
  exec \
  replica/lightrl-manual-4g-20260731-74537606-d8cm2 \
  -- bash
```

执行单条命令：

```bash
brainctl -n ailab-narmodel \
  --request-timeout=120s \
  exec \
  replica/lightrl-manual-4g-20260731-74537606-d8cm2 \
  -- nvidia-smi
```

执行多条命令时使用 `bash -lc`：

```bash
brainctl -n ailab-narmodel \
  --request-timeout=120s \
  exec \
  replica/lightrl-manual-4g-20260731-74537606-d8cm2 \
  -- bash -lc '
    cd /mnt/shared-storage-user/puyuan/code/LightRL
    git status --short --branch
    nvidia-smi
  '
```

`/root/.profile` 中缺少 `/root/.cargo/env` 的提示是非致命登录环境警告，不影响
后续命令。

## 3. 一键运行 SETA + DAPO 验证

在 Replica 内同步执行：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL

WORKER_URLS=http://100.96.26.133:18081 \
NUM_ROLLOUT=3 \
  bash tools/validation/run_4gpu_seta_dapo_validation.sh
```

验证成功应同时出现：

```text
TRAINING_METRICS_OK ... actor_train_steps=... nonzero_actor_updates=...
[4gpu-seta-dapo] All enabled validation stages passed
```

脚本会检查 GPU 和资源、静态编译、imports、相关 pytest、CLI dry-run、worker
健康状态、rollout 指标，以及是否真正产生有限非零的 loss/grad 更新。

## 4. 推荐的后台启动方式

长训练不要依赖持续的 exec 连接。使用 `nohup` 并将日志写入共享目录：

```bash
brainctl -n ailab-narmodel \
  --request-timeout=120s \
  exec \
  replica/lightrl-manual-4g-20260731-74537606-d8cm2 \
  -- bash -lc '
    cd /mnt/shared-storage-user/puyuan/code/LightRL
    RUN_ID=lightrl-seta-dapo-$(date +%Y%m%d-%H%M%S)
    RUN_DIR="$PWD/runs/$RUN_ID"
    mkdir -p "$RUN_DIR"

    if mkdir "$RUN_DIR/.launch-once" 2>/dev/null; then
      nohup env \
        RUN_ID="$RUN_ID" \
        WORKER_URLS=http://100.96.26.133:18081 \
        NUM_ROLLOUT=3 \
        bash tools/validation/run_4gpu_seta_dapo_validation.sh \
        >"$RUN_DIR/launcher.log" 2>&1 </dev/null &
      echo $! >"$RUN_DIR/launcher.pid"
      echo "RUN_ID=$RUN_ID PID=$! LOG=$RUN_DIR/launcher.log"
    else
      echo "RUN_ID 已启动过，不重复启动"
    fi
  '
```

`.launch-once` 是原子标记：控制面超时后重试命令时，可防止意外启动两份训练。

查看日志：

```bash
brainctl -n ailab-narmodel \
  --request-timeout=120s \
  exec \
  replica/lightrl-manual-4g-20260731-74537606-d8cm2 \
  -- bash -lc '
    tail -n 200 \
      /mnt/shared-storage-user/puyuan/code/LightRL/runs/<RUN_ID>/launcher.log
  '
```

## 5. 已验证的排障经验

### TLS handshake timeout 或 EOF

这是开发机连接集群控制面的失败，不代表 rjob 已停止，也不会终止已经用
`nohup` 启动的进程。

重试启动前先检查共享目录中的 `.launch-once`、PID 和日志；确认未启动后再重试。
查询、查看日志等只读命令可以直接重试。

### `rg: command not found`

rjob 镜像不一定包含 ripgrep。验证脚本已经自动回退到 `grep`，不要把安装 `rg`
作为训练前置条件。

### `ModuleNotFoundError: slime.utils`

使用训练环境并补齐仓库内 slime 路径：

```bash
export PATH=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin:$PATH
export PYTHONPATH="$PWD/slime:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### worker 地址被旧文件覆盖

每次运行必须使用当前 run 独立的
`runs/<RUN_ID>/config/worker_urls.txt`。当前 smoke 脚本已实现隔离，默认 worker
为：

```text
http://100.96.26.133:18081
```

### `/reset` 返回 HTTP 500

先手工检查：

```bash
curl --noproxy '*' --fail --max-time 10 \
  http://100.96.26.133:18081/healthz

curl --noproxy '*' --fail --max-time 10 \
  http://100.96.26.133:18081/status | python3 -m json.tool
```

本次故障来自 SETA task 307 Compose 的 build context 和 namespace 标签不兼容。
相关 Compose 已修复。若再次出现 500，应先读取响应正文，而不是反复提交 RJob。

### rollout 有 reward，但 actor train 被跳过

依次检查：

1. 日志是否有 `NameError`、`ModuleNotFoundError` 或 traceback；
2. `logs/metrics.jsonl` 中 `trainable_count` 是否大于 0；
3. 是否出现 `all loss_masks are zero`；
4. `train-step` 是否包含有限的 `train/loss` 和 `train/grad_norm`。

同一 GRPO group 内 reward 完全相同时，零 advantage、零梯度是正常现象；验证应
至少运行多个 rollout，并确认其中至少一个 group 产生非零更新。

### 停止失败的 Ray 验证任务

只停止已确认失效的具体 Ray job：

```bash
export PATH=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin:$PATH

ray job stop \
  --address=http://127.0.0.1:8265 \
  <RAY_JOB_ID>
```

停止后检查 `nvidia-smi` 和 worker `/status`。如果训练被强制停止，可能留下
worker lease；应使用日志或 `/status` 中的准确 lease ID 调用 `/close`：

```bash
curl --noproxy '*' --fail \
  -H 'Content-Type: application/json' \
  -d '{"lease_id":"<RUN_LEASE_ID>"}' \
  http://100.96.26.133:18081/close
```

不要猜测 lease ID，也不要批量清理不属于当前实验的容器。

## 6. 资源检查

每轮训练前后执行：

```bash
free -h
df -h / /mnt/shared-storage-user/puyuan
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
curl --noproxy '*' --fail --max-time 10 \
  http://100.96.26.133:18081/status | python3 -m json.tool
```

重点关注 GPU 显存、内存 available、共享盘和 Docker data root 空间、worker
active runs/pending closes。资源接近上限时优先减小 `NUM_ROLLOUT`、batch 或
并发，不要在同一 4-GPU Replica 上重复启动训练。

