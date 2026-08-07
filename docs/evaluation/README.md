# Evaluation utilities

Evaluation utilities are grouped under `tools/evaluation/`.

- Safety: input preparation, official evaluation orchestration, summaries,
  backend validation, and reward-scale validation.
- SWE: official SWE-bench Verified harness execution.
- Development checks: rule-reward and environment-backend validators.

These utilities consume existing checkpoints or run artifacts; they are not
training entrypoints. Run each command with `--help` (or read its shell usage)
before launching an external benchmark.

The package-side export logic lives in `agentic_rl/evaluation/`（当前为
`swebench/report.py`，负责 SWE-bench 官方 predictions/coverage/summary
产物）；eval rollout 结束时由 `misc/rollout_log.py` 在设置了
`SWEBENCH_RESULTS_DIR` 时调用。
