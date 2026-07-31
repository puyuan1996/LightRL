# LightRL 辅助工具

`tools/` 不提供训练、worker 启动或端到端验证入口。这些用户工作流统一位于
[`examples/`](../examples/README.md)。

```text
tools/
├── analysis/       # 轨迹、训练指标、case study 和 DIVE-PO 对比分析
├── evaluation/     # 安全评测、SWE-bench 官方 harness 和结果汇总
├── dev/            # worker 等开发期冒烟与诊断
└── world_model/    # LWM 数据、probe、候选集和离线评估工具
```

这里的脚本通常处理已有运行结果或协助开发，不是正式训练配方。典型用法：

```bash
python3 tools/analysis/analyze_trajectories.py --run-dir runs/<RUN_ID>
python3 tools/analysis/plot_training_metrics.py --run-dir runs/<RUN_ID>
bash tools/analysis/run_case_study.sh runs/<RUN_ID>
```

安全评测入口及参数见 [`docs/evaluation/README.md`](../docs/evaluation/README.md)；
LWM 工具仍处于开发阶段，使用前阅读
[`docs/algorithms/lwm_guide_zh.md`](../docs/algorithms/lwm_guide_zh.md)。
