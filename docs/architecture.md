# LightRL 框架架构与训练数据流

本文面向需要阅读、运行或扩展 LightRL 的用户，说明仓库目录职责、核心模块边界，以及 SETA + Qwen3-8B + DAPO 训练配方的完整调用链。

## 1. 一句话理解

LightRL 将“模型训练”和“智能体环境交互”连接起来：Slime/Megatron 负责模型更新，`agentic_rl` 负责数据集、环境、工具调用、奖励和 rollout，Docker worker 负责实际运行终端任务。

```text
数据集任务
  ↓
环境分配 / reset
  ↓
模型生成回答与工具调用
  ↓
Docker 或本地环境执行工具
  ↓
任务评估与奖励计算
  ↓
构造训练 Sample
  ↓
Slime/Megatron 执行 DAPO/GRPO 更新
  ↓
记录指标、轨迹和 checkpoint
```

## 2. 仓库顶层结构

```text
examples/training/       可执行训练配方
configs/rollout/         rollout 模型与交互配置模板
agentic_rl/              LightRL 自有逻辑
slime/                   Slime 训练后端
Megatron-LM/             Megatron 模型训练后端
deploy/                  worker、Docker 和 rjob 部署脚本
benchmarks/              任务数据、Docker 环境和评测资源
tools/                   分析、验证和运维工具
docs/                    架构、运行和实验文档
runs/                    运行日志、轨迹和分析结果
```

训练入口保持三层结构：

```text
examples/training/<recipe>.sh
  → agentic_rl/platform/slime_train.sh
  → slime/train_async.py
```

## 3. `agentic_rl` 目录职责

### `algorithms/`：算法扩展

- `dive_po/`：DIVE-PO 默认配置、探索奖励和探索状态管理。
  - `defaults.sh`：DAPO、动态采样、探索 profile、Agent57、LP-RND、CDE 等开关。
  - `exploration/`：内在奖励、轨迹新颖性、探索 bonus 及其调度。
  - `rewards/`：奖励组件合并、alias 同步和奖励后处理。
- `prm/`：过程奖励模型支持。
  - `agent.py`：调用 PRM 对中间 turn 进行评分。

GRPO/DAPO 的核心优化器由 Slime 提供；LightRL 主要负责其环境交互、样本构造和额外奖励接入。

### `data/`：数据准备

- `load_tasks.py`：读取并规范化任务。
- `convert_*_to_dataset.py`：转换 Agent-SafetyBench、AgentHarm、Tau2 等数据。
- `convert_sweverified.py`、`convert_swesmith.py`：转换 SWE 任务及其环境元数据。
- `mix_jsonl_datasets.py`：按比例混合多个数据源。
- `download.py` 及相关 shell 脚本：下载或准备外部数据。

### `environments/`：任务环境抽象

- `registry.py`：环境注册表。统一定义数据源、runtime、远程/本地模式、直接得分模式和轨迹别名。
- `client.py`：访问远程 worker 的 HTTP 客户端，提供 allocate、reset、heartbeat、exec、close。
- `protocol.py`：环境服务端与客户端的数据协议。
- `reward_rules.py`：基于任务结果和工具行为的规则奖励。
- `terminal/`：Docker/Terminal-Bench 环境及 Compose 生命周期管理。
- `agent_safetybench/runtime.py`：Agent-SafetyBench 本地 runtime 和规则奖励。
- `agentharm/runtime.py`：AgentHarm 本地 runtime 和工具任务奖励。
- `tau2/runtime.py`：Tau2 对话环境 runtime。

### `harnesses/`：智能体执行框架适配

- `factory.py`：根据 `HARNESS_OPTION` 创建 agent。
- `camel/`：CAMEL agent、prompt 和工具调用适配。
- `claude_code/`：Claude Code agent、MCP server、prompt 和 Qwen gateway 适配。
- `_developer_prompt.py`：公共 developer prompt 生成逻辑。

### `rollout/`：核心交互与样本生成

- `entrypoint.py`：注册给 Slime 的自定义生成入口。
- `generate_steps.py`：一次 rollout 的分步实现，包括环境会话、模型 turn、工具执行、评估和奖励后处理。
- `runner.py`：agent runner 工厂和统一调用接口。
- `environment_factory.py`：创建本地或远程环境客户端。
- `admission.py`：环境并发准入、熔断、租约释放和失败恢复。
- `backends/`：推理引擎实现与工厂；当前包含 SGLang 生成客户端、重试、超时和结果处理。
- `sample_builder.py`：把交互 turn 转换成训练 `Sample`，处理 outcome、PRM、DAPO overlong 和探索奖励。
- `trajectory_store.py`：保存 `traj.json`、任务元数据、reward breakdown 和轨迹索引。

### `platform/`：训练平台与 worker 基础设施

- `env.py`：统一环境变量解析和变量说明表。
- `paths.py`：统一 run、log、metric、trajectory 和 checkpoint 路径。
- `types.py`：任务、turn、运行上下文等公共类型。
- `router.py`、`router_app.py`、`router_cli.py`：将训练侧环境请求转发到 worker。
- `worker_app.py`、`worker_cli.py`、`worker_pool.py`：管理环境租约、并发、Docker 生命周期和资源压力。
- `worker_admission.py`：worker 的容量和并发准入控制。
- `slime_train.sh`：统一训练编排入口。
- `slime_train/`：按执行阶段拆分的启动脚本：
  - `lib_bootstrap.sh`：基础环境、GPU 拆分和进程准备。
  - `lib_run_dir.sh`：run 目录、checkpoint 和命名。
  - `lib_rollout_cfg.sh`：rollout 配置、模型配置和日志初始化。
  - `lib_dataset.sh`：数据集转换、检查和混合。
  - `lib_worker.sh`：worker URL、router、健康检查和 stale lease 修复。
  - `lib_args.sh`：拼装最终训练参数。
  - `lib_launch.sh`：启动 router、Ray、训练任务并监控收尾。

### `evaluation/`：评测结果处理

- `swebench/report.py`：整理 SWE-bench 标准结果。

### `misc/`：日志与观测

- `rollout_log.py`：记录 reward、成功率、长度、失败率、KL、entropy 等训练指标。
- `rollout_formatting.py`：统一 rollout 数据格式。
- `jsonl_sink.py`：将结构化指标写入 JSONL。

## 4. SETA DAPO 训练脚本

示例入口：[`examples/training/train_qwen3_8b_seta_dapo.sh`](../examples/training/train_qwen3_8b_seta_dapo.sh)

默认配方为：

```text
模型：Qwen3-8B
数据：SETA
算法：DAPO
Harness：camel-agent
GPU：4（默认 2 actor + 2 rollout）
Tensor Parallel：2
探索：关闭
环境：远程 Docker worker
```

运行前必须设置：

```bash
export WORKER_URLS=http://<worker-host>:18081
bash examples/training/train_qwen3_8b_seta_dapo.sh
```

脚本会先检查 GPU 数量和 worker 的 `/healthz`。`--dry-run` 只打印最终训练命令，不启动 Ray、router 或训练任务；`BACKGROUND=1` 时使用 `nohup` 后台运行。

## 5. 启动调用链

### 5.1 Recipe 层

训练脚本设置模型、数据集、算法、GPU 拆分、run ID 和配置路径，然后执行：

```bash
bash agentic_rl/platform/slime_train.sh
```

### 5.2 编排层

`slime_train.sh` 按顺序加载：

```text
lib_bootstrap
  → lib_run_dir
  → lib_rollout_cfg
  → lib_dataset
  → lib_worker
  → lib_args
  → lib_launch
```

各阶段完成基础环境、路径、rollout 配置、数据准备、worker 检查和参数拼装，最后准备 `slime/train_async.py` 的完整命令。

### 5.3 Ray/Slime 层

`lib_launch.sh` 启动 Ray head 和 placement group，然后提交：

```bash
python3 -u slime/train_async.py ...
```

关键自定义入口为：

```text
agentic_rl.rollout.entrypoint.generate
agentic_rl.misc.rollout_log.rollout_log
agentic_rl.misc.rollout_log.eval_rollout_log
```

前者负责生成训练样本，后两者负责训练和评测指标。

## 6. 单个 rollout 的数据流

```text
Slime train_async
  ↓
rollout.entrypoint.generate
  ↓
准备 RunPlan
  ↓
申请环境 lease、reset 环境
  ↓
创建 SGLang client 和 agent harness
  ↓
循环执行模型 turn
  ├─ 模型生成文本
  ├─ 解析 tool call
  ├─ 通过 worker 执行 Docker 工具
  ├─ heartbeat 保活
  └─ 继续下一轮或结束
  ↓
环境 evaluate 得到任务得分
  ↓
sample_builder 构造训练 Sample
  ├─ outcome reward
  ├─ PRM turn reward（可选）
  ├─ DAPO overlong penalty
  └─ DIVE-PO exploration reward（可选）
  ↓
保存轨迹并释放环境 lease
  ↓
返回 Samples 给 Slime
```

每个 rollout 通常包含多个 prompt sample；模型交互中的每个 turn 可以被转换为一个训练样本，失败或不可训练样本会被标记并过滤。

## 7. 训练更新与资源协作

```text
GPU 训练主机                         CPU/Docker worker
──────────────                       ────────────────
Ray                                  pool_server
Megatron actor                       Docker daemon
SGLang rollout engine                Compose task containers
agentic_rl.generate                  reset / exec / close
DAPO optimizer
```

环境请求路径为：

```text
训练进程 → 本地 router（可选） → 远程 pool worker → Docker task container
```

单 worker 直连时可以不启动本地 router；多个 worker 或显式启用 router 时，由 router 负责转发和 worker URL 热更新。训练通过 heartbeat 维持 lease，任务完成后由 worker 回收容器和网络。

## 8. 主要运行产物

一次 run 通常包含：

```text
runs/<RUN_ID>/
├── config/                  最终配置快照
├── logs/train.log           训练主日志
├── logs/metrics.jsonl       结构化 rollout/训练指标
├── trajectories/            轨迹和 reward breakdown
├── reproducibility/         源码、环境和 worker 快照
└── meta.json                run 路径与元数据
```

checkpoint 通常写入独立的 checkpoint 根目录，由 `--save` 和 `--save-interval` 控制保存策略；W&B 可以配置为 offline 模式，将本地记录写入对应 run 目录。

## 9. 阅读和排障入口

建议按以下顺序定位问题：

1. 查看 `runs/<RUN_ID>/logs/train.log`，确认参数、Ray 和训练状态。
2. 查看 `runs/<RUN_ID>/config/run_config.json`，确认最终生效配置。
3. 查看 worker 的 `healthz/status`，确认 Docker worker 是否可用。
4. 查看 `remote_logs/` 或 worker pool 日志，定位 reset、exec、close 和 Docker 错误。
5. 查看 `logs/metrics.jsonl` 和 `trajectories/`，区分模型失败、任务失败和基础设施失败。
6. 修改训练行为时，优先检查 `rollout/entrypoint.py`、`rollout/generate_steps.py` 和 `rollout/sample_builder.py`；修改启动行为时，检查 `platform/slime_train/` 对应阶段脚本。
