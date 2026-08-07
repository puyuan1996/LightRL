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
展示名（canonical → 连字符形式）由同一文件的 `display_harness_name()` 提供。

注：PRM(process reward model）奖励 agent 不是 harness，位于
`agentic_rl/algorithms/prm/`，通过 recipe 的 PRM 相关参数启用。
