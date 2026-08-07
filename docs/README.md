# LightRL 文档导航

## 架构与配置

- [architecture.md](architecture.md)——包边界、主链路与扩展点（环境注册表、
  env 解析、训练三层入口)。
- [configuration.md](configuration.md)——recipe 即配置；`platform/env.py`
  的 `ENV_VARS` 声明表；`environments/registry.py` 的 `EnvSpec` 表。
- [refactor_review_20260731.md](refactor_review_20260731.md)——2026-07-31
  包结构审查记录（分层边界的历史快照）。

## 算法

- [algorithms/dive_po_dual_stream.md](algorithms/dive_po_dual_stream.md)——
  DIVE-PO dual-stream advantage 注入的公式与正确性分析；对应实现
  `agentic_rl/algorithms/dive_po/rewards/dual_stream.py`（生产默认）。
- [algorithms/dive_po_iclr2027_draft.md](algorithms/dive_po_iclr2027_draft.md)——
  DIVE-PO 论文草稿。
- [algorithms/lwm_guide_zh.md](algorithms/lwm_guide_zh.md)——LWM(WIP，
  实现位于 Slime 侧)。

## 使用

- [harnesses/README.md](harnesses/README.md)——harness 选择与注册。
- [evaluation/README.md](evaluation/README.md)——评测工具与 SWE-bench 产物导出。
- [../examples/README.md](../examples/README.md)——训练与验证入口清单。

## 验证记录

- [manual_validation_20260731.md](manual_validation_20260731.md)——2026-07-31
  人工验证手册与常见报错处理。最新一轮（2026-08-07,P0–P2 重构后）结论见
  根目录 README 的"当前验证状态"。

## 运维（站点相关）

`operations/` 是本团队集群（brainctl/rjob + 内网 worker）的运维手册，
开源移植时请将其中地址、命名空间与路径替换为你方站点：

- [operations/brainctl_rjob_debug_zh.md](operations/brainctl_rjob_debug_zh.md)
- [operations/cpu_workers.md](operations/cpu_workers.md)
- [operations/docker_env_server_stability.md](operations/docker_env_server_stability.md)
