# LightRL package structure review

The package uses domain-first folders plus two flat shared layers. This replaces
the earlier one-file package chains under `cli`, `config`, `core`, `runtime`,
`services`, `backends`, `rewards`, `observability`, and `integrations`.

## Current boundaries

| Path | Responsibility | Main entrypoints |
|---|---|---|
| `agentic_rl/algorithms` | LightRL-owned DIVE-PO algorithm extension | `dive_po/` |
| `agentic_rl/environments` | terminal and benchmark runtimes | environment registry targets |
| `agentic_rl/harnesses` | Camel and Claude Code harnesses | harness registry targets |
| `agentic_rl/algorithms/prm` | Process reward model bonus agent | `agent.py` |
| `agentic_rl/rollout` | rollout orchestration and trajectory persistence | `entrypoint.py` |
| `agentic_rl/platform` | Slime runtime, router, and worker infrastructure | `slime_train.sh`, `worker_cli.py`, `router_cli.py` |
| `agentic_rl/misc` | flat logging and sink helpers | `rollout_log.py`, `jsonl_sink.py` |

The public training entrypoint is:

```bash
bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
```

The maintained Slime launcher is:

```text
agentic_rl/platform/slime_train.sh
```

Worker and router processes use:

```bash
python3 -m agentic_rl.platform.worker_cli
python3 -m agentic_rl.platform.router_cli
```

No compatibility packages are retained for the removed paths. Repository code,
tests, scripts, YAML recipes, and documentation must import the flat modules
directly.
