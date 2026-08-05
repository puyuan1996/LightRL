# LightRL

LightRL 是面向智能体环境的强化学习后训练框架。仓库以显式组合的方式组织实验：

```text
Harness × Model × Algorithm × Environment
```

- Harness：Camel-Agent、Claude Code CLI。
- Model：Qwen3-8B、Qwen3-30B-A3B、GLM-5.1。
- Algorithm：GRPO、DAPO、DIVE-PO；LWM 尚在开发中。
- Environment：SETA、Agent-SafetyBench、AgentHarm、SWE-smith 等。

当前主要训练后端是仓库内维护的 Slime 与 Megatron-LM。SETA 等终端环境由独立
Docker worker 执行，GPU 训练进程通过 `WORKER_URLS` 调用 worker；不要在 GPU
训练节点上重复启动同一个环境服务。

## 仓库结构

```text
LightRL/
├── agentic_rl/
│   ├── algorithms/             # GRPO、DAPO、DIVE-PO、LWM 扩展点
│   ├── environments/           # 终端及 benchmark 环境
│   ├── harnesses/              # Camel-Agent、Claude Code、PRM 适配
│   ├── misc/                   # 奖励、可观测性与第三方集成
│   ├── platform/               # 扁平的配置、CLI、后端与服务基础设施
│   ├── rollout/                # rollout 编排、采样与轨迹管理
│   └── data/                   # 数据转换与下载
├── configs/                    # 可组合配置及实验配方
├── examples/                   # 用户工作流：训练配方与端到端验证
├── benchmarks/                 # benchmark 数据与任务定义
├── deploy/workers/             # Docker worker 运维脚本
├── tools/                      # 分析、评测、诊断等辅助工具
├── tests/                      # 单元及静态冒烟测试
├── slime/                      # 训练后端
├── Megatron-LM/                # 模型训练后端
├── runs/                       # 所有运行期输出（不应写到仓库根目录）
└── docs/                       # 架构、算法和运维文档
```

架构和配置扩展方式参见
[架构说明](docs/architecture.md)与[配置说明](docs/configuration.md)。

## 运行前提

- Python 3.10 及以上；实际训练应使用已准备好 CUDA、Slime、Megatron-LM 和模型
  checkpoint 的集群环境。
- SETA/terminal 训练需要一台 Docker 可用的 CPU worker。
- GPU 节点必须能访问 worker 的服务端口，默认是 `18081`。
- worker URL、凭据和站点专用调度配置应放在被 Git 忽略的
  `local/cluster/` 或环境变量中，不要提交到仓库。

只检查 Python 包时可安装：

```bash
python3 -m pip install -e '.[rollout,worker,train]'
python3 -c 'import agentic_rl'
```

这不会自动准备模型权重、CUDA、Slime 或 Megatron-LM 运行环境。

## 典型执行流程：Docker worker + GPU 训练

以下流程使用两个终端。终端 A 位于 Docker/CPU worker 主机；终端 B 位于具有
GPU 和训练环境的节点或 rjob。

### 1. 在终端 A 启动 Docker worker

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL

# 脚本会先检查 Docker daemon；未运行时尝试启动，然后以前台方式启动 pool server。
bash examples/validation/start_docker_worker.sh
```

保持该终端运行。脚本会打印类似：

```text
WORKER_URLS=http://<CPU_WORKER_IP>:18081
```

另开一个 shell 检查服务：

```bash
curl --noproxy '*' --fail http://<CPU_WORKER_IP>:18081/healthz
curl --noproxy '*' --fail http://<CPU_WORKER_IP>:18081/status \
  | python3 -m json.tool
```

Docker daemon、代理、磁盘或并发参数的说明见
[CPU worker 运维](docs/operations/cpu_workers.md)和
[Docker 环境稳定性](docs/operations/docker_env_server_stability.md)。

### 2. 在终端 B 检查实验配置

`--dry-run` 只解析配置，不启动 Ray、rollout 或训练：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL

bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_seta_dive_po.sh --dry-run
bash examples/training/train_qwen3_8b_mixed_dapo.sh --dry-run
```

各示例所需模型路径和环境变量参见 [examples/README.md](examples/README.md)。

### 3. 启动真实训练

下面是 4-GPU Qwen3-8B + SETA + DAPO 的典型命令。将 worker 地址替换为终端 A
实际打印的值：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL

WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
NUM_GPUS=4 \
ACTOR_GPUS=2 \
ROLLOUT_GPUS=2 \
TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/training/train_qwen3_8b_seta_dapo.sh
```

其他已维护入口：

```bash
# SETA + DIVE-PO
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/training/train_qwen3_8b_seta_dive_po.sh

# SETA + Agent-SafetyBench + AgentHarm + DAPO
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/training/train_qwen3_8b_mixed_dapo.sh
```

默认配方面向正式训练，运行时间较长。首次部署建议先执行下一节的有界验证。

## 4-GPU 有界端到端验证

在已经运行的 4-GPU rjob/GPU 节点内执行：

```bash
cd /mnt/shared-storage-user/puyuan/code/LightRL

# SETA + DAPO
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
NUM_ROLLOUT=3 \
bash examples/validation/validate_4gpu_seta_dapo.sh

# DIVE-PO：2 个 rollout
EXPERIMENT=dive_po \
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
NUM_ROLLOUT=2 \
bash examples/validation/validate_4gpu_dive_po_or_mixed.sh

# Mixed DAPO：2 个 rollout
EXPERIMENT=mixed \
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
NUM_ROLLOUT=2 \
bash examples/validation/validate_4gpu_dive_po_or_mixed.sh
```

脚本会检查 GPU、内存、磁盘、Python 编译、worker 健康状态、rollout metrics
和 actor train step；成功时输出 `EXAMPLE_VALIDATION_OK` 或对应的通过标记。

rjob 的 SSH、`brainctl exec` 和日志排查见
[rjob 调试指南](docs/operations/brainctl_rjob_debug_zh.md)。更完整的人工验证与
常见报错处理见[验证手册](docs/manual_validation_20260731.md)。

## 输出目录

所有运行期文件应位于：

```text
runs/<RUN_ID>/
├── config/                   # 配置快照和本次运行生成的数据清单
├── environment_outputs/     # SETA/terminal 环境及 AgentRunner 输出
├── logs/                    # 训练与指标日志
└── manifest.json
```

`runs/latest` 指向最近一次运行。训练不应在仓库根目录生成
`build_outputs`、`tmp_doc_latest` 或 benchmark 临时文件。

## 当前验证状态

截至 2026-07-31，当前重构分支已完成以下真实 4×H200 有界验证：

- SETA + DIVE-PO：连续完成 2 个 rollout、4 个有效 trajectory 和 4 个 actor
  train step；DIVE-PO intrinsic 指标与环境输出正常生成。
- Mixed DAPO：连续完成 2 个 rollout、12 个 trajectory 和 4 个非零 actor
  train step；SETA、Agent-SafetyBench、AgentHarm 三类指标齐全。
- Mixed DAPO 的 4 个 train step 均为有限值，观测到的 grad norm 为
  `1.44～3.11`；Ray job 正常退出。
- 运行结束后 worker 无残留 active task，4 张 GPU 显存均已释放。

以上是短程正确性验证，不代表模型已经收敛或取得正式 benchmark 成绩。复现步骤
和排障说明见 [2026-07-31 人工验证手册](docs/manual_validation_20260731.md)。

## 开发状态

- GRPO 与 DAPO 直接使用 Slime 实现；LightRL 只维护新增的 DIVE-PO 扩展。
- DIVE-PO 配方位于
  `examples/training/train_qwen3_8b_seta_dive_po.sh`，算法说明见
  [DIVE-PO centered gate](docs/algorithms/dive_po_centered_gate.md)。
- LWM 仍为 Slime 内的 WIP；使用前请阅读
  [LWM 指南](docs/algorithms/lwm_guide_zh.md)。
- 重构范围与保守保留项见
  [重构审查记录](docs/refactor_review_20260731.md)。
