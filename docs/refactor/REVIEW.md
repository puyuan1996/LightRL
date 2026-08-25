# LightRL 结构重构审查

基线为 `64e2fb91`，调研日期为 2026-08-24。已递归通读 `docs/` 内 38 个文档，共 4,360 行。

## Step 0：项目意图摘要（24 行）

```text
01 LightRL 的目标是轻量、可扩展的 agentic RL post-training 框架。
02 Slime/Megatron 负责优化与参数更新，LightRL 负责任务、交互、奖励和样本。
03 训练入口已规定为 recipe → platform/slime_train.sh → slime/train_async.py 三层。
04 agentic_rl 按领域组织，共享层应小而平，禁止万能 common 包。
05 docs 已使用 data、environments、evaluation 三个术语，重构必须沿用。
06 data 负责下载、加载、转换、混合与固定 split。
07 environments 负责 reset/step/evaluate 、工具执行与生命周期。
08 evaluation 负责对已有轨迹/结果打分、汇总或导出官方格式。
09 environment registry 是任务到 runtime/data mode/reward mode 的单一发现点。
10 rollout 是环境租约、多轮生成、工具调用、奖励和 Sample 构造的主编排层。
11 harness 是 agent 行为适配，应只依赖 turn client 契约。
12 SGLang 是可替换推理 backend，不是与 rollout 并列的业务域。
13 platform 负责 router、worker pool、远程 Docker 资源和训练编排。
14 训练主机与 CPU/Docker worker 通过 HTTP lease/heartbeat/close 协作。
15 轨迹、reward breakdown、metrics 和配置快照是一等运行产物。
16 DAPO/GRPO 优化语义在 Slime，DIVE-PO 只扩展奖励与探索状态。
17 安全任务、SWE 任务、Tau2 都应复用同一交互主链。
18 examples 是用户工作流，tools 是分析、评测、开发和运维辅助。
19 配置文件应只表达数据，不作为隐式 Python 注册表。
20 已规划但未完成：benchmark 资产的 data/env/eval 角色分离。
21 已规划但未完成：engine 工厂与 rollout 边界归一。
22 不一致：docs 声称领域分层，代码却把底层 types/env/http 放在 platform。
23 不一致：docs 区分 data/env/eval，顶层 benchmarks 却混放原数据、JSONL 和 Docker 任务。
24 不一致：tools 声称是辅助工具，却按历史项目形成 6 个分类。
```

## Step 1：基线目录与行数

下表是重构前深度 3 的重点源码树。按要求忽略 Git、cache、egg-info、数据和 checkpoint；`benchmarks` 大量任务资产以数量汇总，不将 JSON/Docker fixture 误当源码逐件展开。

```text
agentic_rl/inference/
├── __init__.py                                      1
└── sglang.py                                       354
agentic_rl/rollout/
├── __init__.py                                      1
├── admission.py                                   230
├── entrypoint.py                                  265
├── environment_factory.py                         144
├── generate_steps.py                             1390
├── runner.py                                       202
├── sample_builder.py                               290
├── sglang_factory.py                               114
└── trajectory_store.py                            1091
agentic_rl/environments/
├── __init__.py                                      1
├── client.py                                       183
├── protocol.py                                      37
├── registry.py                                     208
├── reward_rules.py                                 212
├── agent_safetybench/{__init__.py:1,runtime.py:500}
├── agentharm/{__init__.py:1,runtime.py:606}
├── tau2/{__init__.py:1,runtime.py:527}
└── terminal/{__init__.py:1,docker_compose.py:853,runtime.py:2898,swe_utils.py:24}
agentic_rl/data/
├── convert_agent_safetybench_to_dataset.py          98
├── convert_agentharm_to_dataset.py                 171
├── convert_swesmith.py                            1670
├── convert_sweverified.py                          429
├── convert_task_to_dataset.py                      175
├── convert_tau2_to_dataset.py                      239
├── download.py                                     139
├── download_swesmith.sh                            388
├── download_sweverified.sh                          86
├── load_tasks.py                                   119
└── mix_jsonl_datasets.py                           180
benchmarks/  (assets)
├── agent_safetybench 1; agent_safetybench_convert 4
├── agentharm 6; agentharm_convert 11; mcpsafety 250
└── seta_env 10,048; seta_env_convert 12; seta_env_retry 121 symlinks/3 dirs
tools/
├── analysis/      14 files: 30–1689 lines/file
├── dev/            5 files: 30‑406 lines/file
├── evaluation/    13 files: 72‑488 lines/file
├── network/        1 file: tcp_relay.py 92
├── reproducibility/1 file: capture_formal_run_source.py 168
└── world_model/     4 files: 73‑416 lines/file
```

## 职责、公开面与 import 方

| 基线范围 | 一句话职责 | 关键类/函数 | 主要 import 方 |
|---|---|---|---|
| `inference` | SGLang 单 turn 生成与工具调用解析 | `SGLangTurnClient`, `process_tool_calls` | rollout engine factory、Camel/Claude harness |
| `rollout` | 编排环境租约、多 turn 采样、奖励和训练样本 | `generate`, `AgentRunner`, `create_agent_runner`, `RunPlan`, sample/trajectory helpers | Slime custom hook、tests、analysis/eval tools |
| `environments` | 统一本地/远程任务的 reset/exec/evaluate/close | `EnvSpec`, `get_env_spec`, `EnvClient`, `TerminalEnvClient`, runtimes, reward rules | rollout、platform worker、tests、worker tools |
| `data` | 下载、加载、转换和混合任务 | `load_terminal_bench_tasks`, converter `main`, `mix_datasets` | training shell libraries、tests；terminal runtime 读 SWE metadata |
| `benchmarks` | 保存静态数据与可执行 Docker 任务资产 | 无 Python API | configs、data converters、worker/prebuild/eval scripts |
| `evaluation` | 官方结果格式与 coverage 导出 | `write_official_artifacts`, `build_prediction_coverage` | rollout logging、SWE tests |
| `tools` | 运行后分析、评测、开发冒烟和基础设施辅助 | 各文件的 CLI `main` | shell recipes/tests/人工调用；不是 library API |

## 依赖审计

基线主图（`A → B` 表示 A import B）：

```text
slime/scripts → rollout → inference
                         → harnesses → inference       # 与 rollout 共用具体实现
                         → environments → platform     # 底层反依赖
platform → environments                                    # 与上行构成包级双向
environments → data                                       # Tau2/SWE 任务定义
misc → algorithms/rewards ↔ rewards/dual_stream       # 包初始化循环
```

显式问题与证据：

- 循环 import：`algorithms/dive_po/rewards/__init__.py:3` 导入 `dual_stream`，而 `dual_stream.py:30` 经包名导入 `rewards.postprocess`。
- 反向依赖：环境 runtime/client 从 `platform.types/env/http_client` 取底层原语，同时 platform worker 又 import environments。
- 重复 Tau2 实现：`data/convert_tau2_to_dataset.py:115,149` 与 `environments/tau2/runtime.py:116,152` 分别复制 `ensure_tau2_importable` / `task_instruction`。
- 重复 Dockerfile 检查：`tools/dev/dockerfile_precheck.py:17-64` 声明复制 `environments/terminal/docker_compose.py:400-446`。
- 分散 backend 构造：实现在 `inference/sglang.py`，工厂在 `rollout/sglang_factory.py`，harness 直接 import 具体 client。

## `tools/` 审计

“最后修改”按该目录内文件最新 Git author date 统计，比 move 后的文件系统 mtime 更可复现。

| 原子目录 | 文件数 | `__init__.py` | 一次性脚本 | 最后修改 | 结论 |
|---|---:|---|---|---|---|
| analysis | 14 | 否 | 否，均有参数/CLI 或配套资产 | 2026-08-17 | 保留 |
| dev | 5 | 否 | 否，可重复的 prebuild/smoke/precheck | 2026-08-13 | 保留 |
| evaluation | 13 | 否 | 否，官方评测/结果汇总 | 2026-08-24 | 保留 |
| network | 1 | 否 | 否，通用 TCP relay CLI | 2026-08-12 | **1 文件目录**，并入 infra |
| reproducibility | 1 tracked | 否 | 否，正式运行快照 CLI | 2026-08-13 | **1 文件目录**，并入 infra |
| world_model | 4 | 否 | 否，可重复的 probe/eval | 2026-08-07 | 并入 evaluation |

死代码判定同时要求“全库无引用 + 非 CLI + 非文档引用”。无脚本满足三项：无调用方的文件均有 `__main__` 或 shell CLI 入口。工作树中未跟踪的 `tools/analysis/swe_report.py` 与被删的 `agentic_rl/evaluation/swebench/report.py` 字节相同，已按 docs 意图恢复到 evaluation，而不当作新工具。

## Step 2：三个裁决

### Q1：合并 `inference/` 与 `rollout/`

**结论：合并为 `rollout/backends/{factory,sglang}.py`。**

证据：基线只有一个 354 行 SGLang 实现；其工厂原本已在 rollout；样本/turn 类型由共享 types 持有，没有独立 inference request protocol 或非 rollout 消费者。抽出 `harnesses.protocol.TurnClient` 后，依赖变为 `rollout backend → harness protocol`，harness 不再反向 import rollout。

反方意见：当存在多个可独立部署的引擎、独立 batching/request API 或被非 rollout 服务使用时，`inference` 独立包更清晰。本库尚无这些证据，保留两个单文件包只增加跳转和双向依赖。

### Q2：选择 (c) data / environment / evaluation 三分离

| 关注点 | 代码 | 资产 |
|---|---|---|
| 数据供给 | `agentic_rl/data` | `benchmarks/datasets` 原数据、JSONL、split、manifest |
| 交互动力学 | `agentic_rl/environments` | `benchmarks/environments` Dockerfile、task.yaml、fixtures |
| 评测打分 | `agentic_rl/evaluation` | `tools/evaluation` 可执行评测/汇总 CLI |

方案 (a) 会让静态输入与 runtime 紧耦合；(b) 会让 Docker 生命周期落入 loader；(d) 保留了 benchmark-first 混合和路径特例。(c) 与 docs 术语一致，且新 benchmark 只在其真正需要的角色中增文件，不必创建同名全套目录。

反方意见：benchmark-first 可把一个任务所有东西放在一处，独立发布时更方便。但本库的数据转换、runtime 和官方 evaluator 已有不同依赖与生命周期，物理混放反而让每次新增都需要理解全链。

### Q3：`tools/` 按生命周期收敛为 4 类

**规则：** `analysis/` 读运行产物；`evaluation/` 发起或汇总评测；`dev/` 是开发期冒烟/预构建；`infra/` 处理网络、环境包和可复现快照。不再为一个研究主题创建顶层工具目录。

反方意见：单层扁平化能彻底消除小目录。但基线已有 38 个可执行文件，单层会丢失“何时使用”的发现性。四类正好满足子目录数 `<=4`，且每类至少 4 个文件。

## 行为影响

- `[BEHAVIOR CHANGE]` 仅有文件系统路径变化：`DATASET_DIR` 默认环境资产根改为 `benchmarks/environments`，数据产物改为 `benchmarks/datasets`；仓库内配置、脚本、测试和文档已一次性更新。
- 算法、奖励、采样参数、任务 ID、JSONL 内容与数值行为未改变。
