# LightRL Architecture

LightRL 将训练配方、项目新增逻辑和第三方训练后端明确分开：

```text
examples/training/       # 可直接执行、可审阅的训练 recipe
configs/rollout/         # rollout 模型模板（唯一保留的组合配置）
agentic_rl/
├── algorithms/dive_po/  # LightRL 新增的 DIVE-PO 实现与 defaults
├── algorithms/prm/      # PRM(process reward)奖励 agent
├── data/                 # 数据转换与下载
├── environments/        # 环境运行时、注册表(registry.py)与协议(protocol.py)
├── evaluation/          # 评测适配
├── harnesses/           # Camel / Claude Code agent harness
├── inference/           # 推理客户端
├── rollout/             # rollout 编排(entrypoint)+ 步骤实现(generate_steps)+ trajectory
├── platform/            # Slime runtime、worker 与 router
└── misc/                # 日志(rollout_log/jsonl_sink)和第三方集成(clawsentry)
slime/                   # 第三方训练后端
Megatron-LM/             # 第三方模型训练后端
```

公开训练链路只有三层：

```text
examples/training/<recipe>.sh
  -> agentic_rl/platform/slime_train.sh   # 68 行编排器,按原序 source slime_train/lib_*.sh
  -> slime/train_async.py
```

启动器已拆分为 7 个阶段库(`agentic_rl/platform/slime_train/`):lib_bootstrap、
lib_run_dir、lib_rollout_cfg、lib_dataset、lib_worker、lib_args、lib_launch;
调用方路径与行为完全不变(四个 recipe 的 dry-run 命令逐 token 对拍一致)。

GRPO、DAPO 直接由 Slime 提供，`agentic_rl/algorithms/` 不为它们维护占位包。
DIVE-PO 是 LightRL 新增能力，因此其 exploration 与 reward 实现集中在
`agentic_rl/algorithms/dive_po/`。Harness 的名称映射集中在
`agentic_rl/harnesses/factory.py`，并通过惰性 import 隔离可选依赖。
环境（数据源）的判定集中在 `agentic_rl/environments/registry.py` 的 `EnvSpec`
表：本地/远程选择、打分模式、安全奖励模式与轨迹落盘别名都查这张表，新增
环境只需注册一行。环境变量解析集中在 `agentic_rl/platform/env.py`。

单个 `WORKER_URLS` 默认由训练进程直接访问；多个 worker 或显式设置
`START_ENV_POOL_SERVER=1` 时才需要本地 router。站点地址、凭据和调度容量通过环境变量
或被 Git 忽略的 `local/cluster/` 提供。
