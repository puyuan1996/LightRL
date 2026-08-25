# LightRL 结构重构计划与落地映射

本文记录已落地的目标架构；实际提交顺序与本计划一致。

## 目标目录树

```text
agentic_rl/
├── __init__.py                  # 只导出核心数据类型
├── types.py                     # 任务、交互、turn/run 类型
├── env.py                       # 环境变量与超时配置
├── http_client.py               # 通用 HTTP client
├── http_server.py               # 通用服务端 JSON helper
├── data/                         # 数据下载/转换/加载/混合
│   ├── __init__.py
│   ├── tau2_support.py
│   └── convert_*/download_*/load_tasks.py/...
├── environments/                 # 交互动力学与环境生命周期
│   ├── registry.py client.py protocol.py reward_rules.py
│   ├── agent_safetybench/ agentharm/ tau2/
│   └── terminal/
│       ├── runtime.py docker_compose.py swe_utils.py
│       └── validation.py
├── evaluation/                   # 评测结果导出/打分
│   └── swebench/report.py
├── harnesses/                     # agent 行为与 tool-call 适配
│   ├── protocol.py tool_calls.py factory.py
│   ├── camel/
│   └── claude_code/
├── rollout/                       # 采样编排、样本和轨迹
│   ├── backends/
│   │   ├── factory.py
│   │   └── sglang.py
│   └── admission.py entrypoint.py environment_factory.py
│       generate_steps.py runner.py sample_builder.py trajectory_store.py
├── algorithms/                    # 奖励/算法扩展
├── misc/                          # 运行日志与观测
└── platform/                      # router/worker/训练编排
benchmarks/
├── README.md
├── datasets/                      # 原数据、JSONL、split、manifest
└── environments/                  # Docker 任务、task.yaml、tests/fixtures
tools/
├── README.md
├── analysis/                      # 运行产物分析
├── evaluation/                    # 评测、probe、汇总
├── dev/                           # 开发期 smoke/prebuild/precheck
└── infra/                         # 网络、快照、离线运行资产
```

## 包职责与允许依赖

| 包/层 | 职责 | 允许向下依赖 |
|---|---|---|
| root primitives | 不带业务政策的 types/env/http | 标准库和必需第三方库，不依赖业务包 |
| `data` | 数据供给和任务文本规范化 | root primitives |
| `harnesses` | agent/tool-call 适配及 `TurnClient` 契约 | root primitives |
| `algorithms` | 奖励与算法插件 | root primitives |
| `environments` | reset/exec/evaluate/close | root primitives；单向复用 `data.tau2_support` |
| `evaluation` | 读取已有输出并打分/导出 | root primitives；不被 rollout 调用 |
| `rollout` | 组合 backend、harness、environment 并产生 Sample | root/data/environment/harness/algorithm |
| `misc` | 消费运行结果并记录指标 | evaluation/algorithm/root |
| `platform` | 对外服务、worker 和训练编排 | environments/root，下层不得回指 platform |
| `examples/configs/tools` | 用户入口和运维 | 可调用上述包，不得被 library 反向 import |

结构约束：基础原语放在根部而非 `platform`；backend 位于 rollout 内；`TurnClient` 契约位于 harness 侧；三类 benchmark 职责使用不同目录。这四点让反向 import 在物理路径上显眼，再由 AST DFS 作自动检查。

## 文件级迁移映射

对数千个 benchmark 资产使用 glob 表示；所有移动均以 `git mv` 执行。

| 旧路径 | 新路径 | 操作 | 理由 |
|---|---|---|---|
| `agentic_rl/inference/sglang.py` | `agentic_rl/rollout/backends/sglang.py` | move | 引擎只服务 rollout，与工厂共居 |
| `agentic_rl/rollout/sglang_factory.py` | `agentic_rl/rollout/backends/factory.py` | move | backend 构造细节封装到同一子包 |
| `agentic_rl/inference/__init__.py` | — | delete | 空包壳；不保留 re-export shim |
| SGLang client 内的 turn 契约 | `agentic_rl/harnesses/protocol.py` | split | harness 拥有最小 `TurnClient` protocol |
| harness 内重复 tool-call helper | `agentic_rl/harnesses/tool_calls.py` | merge | Camel/Claude 共用，避免依赖 backend |
| `agentic_rl/platform/types.py` | `agentic_rl/types.py` | move | 业务无关类型不属于上层 platform |
| `agentic_rl/platform/env.py` | `agentic_rl/env.py` | move | 消除 environments → platform |
| `agentic_rl/platform/http_client.py` | `agentic_rl/http_client.py` | move | 消除 client/environments → platform |
| `agentic_rl/platform/http.py` | `agentic_rl/http_server.py` | move/rename | 下沉 helper，同时避免遮蔽 stdlib `http` |
| Tau2 converter/runtime helpers | `agentic_rl/data/tau2_support.py` | merge | 删除两套相同的 import/task text 逻辑 |
| Dockerfile precheck 两份实现 | `agentic_rl/environments/terminal/validation.py` | merge | runtime 与 dev CLI 共用无重依赖实现 |
| `agentic_rl/evaluation/swebench/report.py` 的未跟踪副本 | 原 canonical 路径 | merge/restore | docs 已声明 evaluation，副本字节相同 |
| `benchmarks/{agent_safetybench,agentharm,mcpsafety}*` | `benchmarks/datasets/<same>` | move | 原始/转换数据与运行环境分离 |
| `benchmarks/seta_env_convert` | `benchmarks/datasets/seta_env_convert` | move | JSONL/split/manifest 属数据供给 |
| `benchmarks/seta_env` | `benchmarks/environments/seta_env` | move | Docker 任务属交互环境资产 |
| `benchmarks/seta_env_retry` | — | delete | 121 个软链接是两次运行生成的 retry view，无代码/配置引用 |
| `tools/world_model/*` | `tools/evaluation/*` | move | 均为 probe/eval，不需要研究主题顶层目录 |
| `tools/network/*` | `tools/infra/*` | move | 网络是基础设施生命周期 |
| `tools/reproducibility/*` | `tools/infra/*` | move | 源码/运行快照属基础设施 |
| `tools/analysis/swe_report.py` | `agentic_rl/evaluation/swebench/report.py` | merge | 收回未跟踪重复副本，不保留别名 |
| 全库 Python/docs/YAML/shell/TOML/JSON | 新 import 与路径 | update | 一次性更新，无兼容层 |

## 提交列表

| Commit | 动机 |
|---|---|
| `f0bf7e46 refactor(tools): group utilities by operational lifecycle` | 6 个工具子目录收敛为 4 个生命周期目录 |
| `8dc50c1f refactor(rollout): merge inference engine into rollout` | 合并 engine/factory，抽出 harness protocol |
| `6e5f9cb6 refactor(bench): split data environment and evaluation concerns` | 分离 benchmark 资产，恢复 evaluation，合并重复 helper |
| `57e485cc refactor(core): extract dependency-neutral primitives` | 将 types/env/http 移出 platform，打破反向依赖和 rewards 循环 |
| `a80bd448 fix(refactor): preserve import and test invariants` | 修复 stdlib 遮蔽、惰性可选导入、剩余测试路径与包内导入 |
| `docs(refactor): document architecture decisions and verification` | 写入 REVIEW/PLAN/VERIFICATION 与最终架构文档 |

未创建虚假的 `chore: remove dead code` 提交：严格三条件审计后没有可删的一次性/死脚本。被移除的只有合并后的空 `inference/__init__.py`，已随对应 rollout commit 提交，这比为提交名而删除可用 CLI 更符合约束。
