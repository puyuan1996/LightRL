# LightRL

**LightRL 是面向智能体环境的轻量、高效、可扩展强化学习后训练框架。**

English README: [README.md](README.md)。

实验按四个维度显式组合：

```text
Harness × Model × Algorithm × Environment
```

- **Harness**:Camel-Agent、Claude Code CLI(`agentic_rl/harnesses/`)
- **Model**:Qwen3-8B、Qwen3-30B-A3B、GLM-5.1(rollout 模板见 `configs/rollout/`)
- **Algorithm**:GRPO / DAPO(由仓库内置 Slime 后端直接提供）、
  DIVE-PO(LightRL 自研探索扩展）
- **Environment**:SETA 终端任务、Agent-SafetyBench、AgentHarm、tau2、
  SWE-smith / SWE-verified(`agentic_rl/environments/`)

训练后端为仓库内的 Slime 与 Megatron-LM。终端类环境在独立 Docker worker
中执行,GPU 训练进程通过 `WORKER_URLS` 经 HTTP 调用;多 worker 时可在本地
启动 router 做租约路由。不要在 GPU 训练节点上重复启动同一个环境服务。

## 特性

- **开箱即用的训练配方**——每个实验就是一个可审阅的 shell 脚本；先用
  `--dry-run` 打印完整后端命令，确认无误再启动。
- **多智能体环境支持**——SETA 终端任务、Agent-SafetyBench、AgentHarm、
  tau2、SWE-smith / SWE-verified；终端类环境在独立 Docker worker 中
  隔离执行，训练侧经 HTTP 调用。
- **低成本扩展**——新增一个 benchmark 环境，只需在环境注册表中加一行
  声明并实现一个 runtime 类；harness 与奖励后处理同样有明确的注册入口。
- **自研 DIVE-PO 探索算法**——episodic / lifelong 新颖度估计 +
  dual-stream advantage 注入，按配方一键开关；GRPO / DAPO 由内置
  Slime 后端直接提供。
- **完整的可观测性**——逐轮对话轨迹、结构化指标（JSONL）、wandb 曲线，
  默认开启、无需额外配置。
- **有界验证脚本**——4-GPU 小样本端到端自检（rollout → 奖励 → 梯度更新），
  数分钟即可确认部署正确性。

## 架构

```text
examples/training/<recipe>.sh
  → agentic_rl/platform/slime_train.sh          # 启动器:数据准备、worker 发现、命令组装
  → slime/train_async.py                        # 训练后端(GRPO/DAPO)
      → agentic_rl/rollout/entrypoint.generate  # 自定义 rollout 钩子
          ├─ environments/registry.py           #   本地 runtime vs 远程 Docker worker
          ├─ harnesses/factory.py               #   camel-agent / claude-code
          ├─ inference/sglang.py                #   经 sglang 的 token 级生成
          ├─ rollout/generate_steps.py          #   交互主循环、评分、探索奖励
          └─ rollout/sample_builder.py          #   奖励成形 → Sample.reward["score"]
      → algorithms/dive_po/rewards/dual_stream  # 自定义奖励后处理(组内归一化)
```

## 仓库结构

```text
LightRL/
├── agentic_rl/
│   ├── algorithms/             # DIVE-PO(exploration/rewards/defaults)+ PRM 奖励 agent
│   ├── data/                   # 数据转换与下载(可独立运行的脚本)
│   ├── environments/           # registry.py、protocol.py、环境 runtime、奖励规则、HTTP client
│   ├── evaluation/             # SWE-bench 官方格式导出
│   ├── harnesses/              # Camel-Agent / Claude Code harness + 工厂
│   ├── inference/              # sglang 轮次客户端库(rollout 与 harness 共用)
│   ├── misc/                   # rollout 日志、JSONL sink、ClawSentry 集成
│   ├── platform/               # slime 启动器、worker/router 服务、路径、env 解析、http client
│   └── rollout/                # entrypoint 钩子、generate_steps、runner、准入、sglang 装配、轨迹存储
├── configs/rollout/            # rollout 模型模板(唯一保留的配置层)
├── examples/                   # 训练配方 + 有界端到端验证
├── benchmarks/                 # benchmark 数据与任务定义
├── deploy/workers/             # Docker worker 运维脚本
├── tools/                      # 分析、评测、诊断工具
├── tests/                      # 单元测试(pytest)
├── slime/                      # 训练后端(第三方)
├── Megatron-LM/                # 模型训练后端(第三方)
├── runs/                       # 所有运行期输出(已被 Git 忽略)
└── docs/                       # 架构、算法与运维文档
```

## 运行前提

- Python ≥ 3.10;真实训练使用已准备好 CUDA、Slime、Megatron-LM 与模型
  checkpoint 的集群镜像。
- SETA/terminal 任务需要一台 Docker 可用的 CPU worker。
- GPU 节点必须能访问 worker 服务端口(默认 `18081`)。
- 站点地址、凭据与调度参数放在环境变量或被 Git 忽略的 `local/cluster/`,
  不要提交到仓库。

仅做源码级安装:

```bash
python3 -m pip install -e '.[rollout,worker,train]'
python3 -c 'import agentic_rl'
```

该命令不会准备模型权重、CUDA 或训练后端运行环境。

## 快速开始

### 1. 启动 Docker worker(CPU 主机)

```bash
cd LightRL
bash examples/validation/start_docker_worker.sh
# 输出 WORKER_URLS=http://<WORKER_IP>:18081

curl --noproxy '*' --fail http://<WORKER_IP>:18081/healthz
```

daemon、代理、磁盘与并发参数见
[CPU worker 运维](docs/operations/cpu_workers.md)与
[Docker 环境稳定性](docs/operations/docker_env_server_stability.md)。

### 2. dry-run 检查配方(GPU 主机)

```bash
bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_seta_dive_po.sh --dry-run
bash examples/training/train_qwen3_8b_mixed_dapo.sh --dry-run
```

### 3. 启动真实训练

```bash
WORKER_URLS=http://<WORKER_IP>:18081 \
NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 TP_SIZE=2 ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/training/train_qwen3_8b_seta_dapo.sh
```

其余已维护入口(DIVE-PO、mixed、GLM-5.1)见
[examples/README.md](examples/README.md)。

### 4. 有界端到端验证(首次部署推荐)

```bash
WORKER_URLS=http://<WORKER_IP>:18081 NUM_ROLLOUT=3 \
  bash examples/validation/validate_4gpu_seta_dapo.sh

EXPERIMENT=dive_po WORKER_URLS=http://<WORKER_IP>:18081 \
  bash examples/validation/validate_4gpu_dive_po_or_mixed.sh

EXPERIMENT=mixed WORKER_URLS=http://<WORKER_IP>:18081 \
  bash examples/validation/validate_4gpu_dive_po_or_mixed.sh
```

脚本依次检查 GPU/资源、静态编译、imports、单元测试、CLI dry-run、worker
健康、rollout 指标,并验证 actor 产生有限且非零的更新
(`TRAINING_METRICS_OK` / `EXAMPLE_VALIDATION_OK`)。

## 配置说明

训练配置直接写在 recipe 脚本中,环境变量可覆盖默认值。Python 侧的环境
变量解析集中在 `agentic_rl/platform/env.py`,其中 `ENV_VARS` 表文档化了
rollout 域的全部变量。详见 [配置说明](docs/configuration.md)。

## 输出目录

```text
runs/<RUN_ID>/
├── config/                # 配置快照与本次运行生成的数据清单
├── environment_outputs/   # 环境侧 AgentRunner 输出
├── logs/                  # train.log、metrics.jsonl
├── trajectories/          # 逐样本 traj.json + index.jsonl(旁路观测)
└── metrics/               # wandb / 分析产物
```

训练不应在仓库根目录生成临时文件;`runs/latest` 指向最近一次运行。

## 当前验证状态

最近一次有界验证(2026-08-07,4×H200,P0–P2 重构轮完成后;更早记录见
[2026-07-31 人工验证手册](docs/manual_validation_20260731.md)):

- SETA + DAPO:3 个 rollout、6 个 actor train step、非零有限更新
  (`TRAINING_METRICS_OK`)。
- SETA + DIVE-PO:3 个 rollout、7 个 actor train step、4 个非零更新
  (`EXAMPLE_VALIDATION_OK`);轨迹产物完整导出。
- Mixed(SETA + Agent-SafetyBench + AgentHarm)+ DAPO:8 条指标记录、
  4 个 actor train step、4 个非零更新(`EXAMPLE_VALIDATION_OK`)。

以上是短程正确性验证,不代表模型已收敛或取得正式 benchmark 成绩。

## 开发

```bash
python3 -m pytest tests/ -q        # 219 个测试
python3 -m compileall -q agentic_rl
```

扩展点:

- 新环境 → `agentic_rl/environments/registry.py` 注册一行 `EnvSpec` +
  实现 `environments/protocol.py:EnvClient` 协议。
- 新 harness → `agentic_rl/harnesses/factory.py` 注册
  (`_HARNESS_ALIASES` + `_HARNESS_TARGETS`),并实现
  `rollout/runner.py:RolloutAgent` 协议。
- 新奖励后处理 → 暴露 `post_process_rewards(args, samples)`,用
  `CUSTOM_REWARD_POST_PROCESS_PATH` 指过去。

## 文档导航

- [架构说明](docs/architecture.md)——包边界与分层
- [配置说明](docs/configuration.md)——recipe 与环境变量
- [DIVE-PO 奖励数学](docs/algorithms/dive_po_dual_stream.md)
- [Harness 选择](docs/harnesses/README.md)
- [评测工具](docs/evaluation/README.md)
- [运维手册](docs/operations/)——站点相关(brainctl/rjob、CPU worker、
  Docker 稳定性),移植时请将其中地址替换为你方站点

## 致谢

LightRL 基于并内置两个训练后端：[Slime](https://github.com/THUDM/slime)
(rollout/训练运行时）与 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
（模型训练）。智能体 RL 技术栈——终端环境、harness、奖励成形与 rollout
编排——最初在 **OpenClaw-RL** 中研发，后经抽取与重构成为本轻量框架。
感谢上述项目的作者们。

## 引用

如果 LightRL 对您的工作有帮助，请引用：

```bibtex
@misc{lightrl,
  title={LightRL: A Lightweight, Efficient, Scalable RL Post-training Framework for Agentic Environments},
  author={Pu, Yuan and Zhang, Shaoang and Zhang, Chenhao and Li, Xueyan and Lu, Yudong and Tang, Jia and Wang, Guanchu and Niu, Yazhe},
  publisher={GitHub},
  howpublished={\url{https://github.com/opendilab/LightRL}},
  year={2026},
}
```

## License

[MIT](LICENSE)
