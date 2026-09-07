# Evaluation utilities

Evaluation utilities are grouped under `tools/evaluation/`.

推荐使用模块入口：

```bash
python3 -m tools.evaluation run --config <config.yaml> --dry-run
python3 -m tools.evaluation smoke --config <config.yaml> --task <task>
```

`examples/evaluation/` 只保留面向用户的完整配方；例如 Qwen3-8B + SETA
fixed12 + Camel-Agent 的 4-GPU 配方为
`examples/evaluation/run_qwen3_8b_seta_fixed12_camel_4gpu.sh`。通用 CLI 和
benchmark 脚本不负责提交 RJob，Math RLVR 的可审阅提交器位于
`local/rjob/`，站点参数通过未跟踪配置注入。

Math RLVR 的完整协议、入口、指标和实验状态见
[`math_rlvr_zh.md`](math_rlvr_zh.md)。

- Safety: input preparation, official evaluation orchestration, summaries,
  backend validation, and reward-scale validation.
- SWE: official SWE-bench Verified harness execution.
- Development checks: rule-reward and environment-backend validators.

These utilities consume existing checkpoints or run artifacts; they are not
training entrypoints. Run each command with `--help` (or read its shell usage)
before launching an external benchmark.

## SETA

- [seta_fixed12_protocol_zh.md](seta_fixed12_protocol_zh.md)：当前 DAPO / DIVE-PO
  成对长训使用的 fixed12 样本、系统抽样方法、冻结解码协议与统计解释边界。

The official-format export logic lives in
`tools/evaluation/benchmarks/swebench/report.py`（负责 SWE-bench 官方
predictions/coverage/summary 产物）；eval rollout 结束时由
`misc/rollout_log.py` 在设置了 `SWEBENCH_RESULTS_DIR` 时调用。
