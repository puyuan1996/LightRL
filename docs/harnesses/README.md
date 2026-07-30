# Harnesses

Harness implementations live in `agentic_rl/harnesses/` and are selected by
`harness.name`.

Available integrations:

- `camel_agent`: the default terminal-agent harness.
- `claude_code_cli`: Claude Code CLI with the LightRL MCP bridge.

Inspect a Claude Code composition without starting training:

```bash
python3 -m agentic_rl.cli train --dry-run \
  --config configs/experiment/dive_po_qwen3_8b_seta.yaml \
  harness.name=claude_code_cli
```

New harnesses should implement the core rollout interface, register a stable
name in `agentic_rl/core/registry.py`, and add a config under
`configs/harness/`. Algorithm and model code should not need modification.
