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
- [algorithms/jepa_world_model_progress_zh.md](algorithms/jepa_world_model_progress_zh.md)——
  JEPA latent world model 的 PR 历史、数据审计、实验结果、结论边界与后续计划。
- [algorithms/jepa_world_model_project_status_zh.md](algorithms/jepa_world_model_project_status_zh.md)——
  JEPA latent world model 的导师问答、项目结构、Goal 进度与迁移验收建议。

## 使用

- [harnesses/README.md](harnesses/README.md)——harness 选择与注册。
- [evaluation/README.md](evaluation/README.md)——评测工具、SETA fixed12 协议与
  SWE-bench 产物导出。
- [performance/seta_training_efficiency_zh.md](performance/seta_training_efficiency_zh.md)——
  SETA 训练耗时、同 Pod 私有 worker 的长窗口提速结果、GPU 等待瓶颈、
  slime trace 与高吞吐执行 profile。
- [../examples/README.md](../examples/README.md)——训练与验证入口清单。

## 运维

- [operations/checkpoint_wandb_storage_zh.md](operations/checkpoint_wandb_storage_zh.md)——
  tracker 感知的 checkpoint 清理、磁盘满非致命策略与 W&B offline 同步。
