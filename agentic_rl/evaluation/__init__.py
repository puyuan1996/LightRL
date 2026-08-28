"""Official-format result exporters consumed by the rollout pipeline.

边界说明(避免与同名目录混淆):

- 本包是**库代码**:被 `agentic_rl/misc/rollout_log.py` 在 eval hook 中
  import,把 rollout samples 导出成官方格式(目前只有 SWE-bench)。
- 离线评估的 harness 适配层在 `agentic_rl.harnesses.eval`;
  离线评估的命令行工具集在 `tools/evaluation/`。三者的分工见
  `docs/harnesses/README.md` 与 `tools/evaluation/README.md`。
"""

from .swebench import build_prediction_coverage, write_official_artifacts

__all__ = ["build_prediction_coverage", "write_official_artifacts"]
