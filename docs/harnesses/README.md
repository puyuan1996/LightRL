# Harnesses

Harness 实现位于 `agentic_rl/harnesses/`，训练 recipe 通过 `HARNESS_OPTION` 选择：

- `camel-agent`：默认 terminal-agent harness；
- `claude-code`：Claude Code CLI 与 LightRL MCP bridge。

例如：

```bash
HARNESS_OPTION=claude-code \
  bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
```

映射集中在 `agentic_rl/harnesses/factory.py`，使用惰性 import 保持可选依赖隔离。
