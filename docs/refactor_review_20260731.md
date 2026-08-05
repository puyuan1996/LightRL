# LightRL package structure review

The package uses domain-first folders plus two flat shared layers. This replaces
the earlier one-file package chains under `cli`, `config`, `core`, `runtime`,
`services`, `backends`, `rewards`, `observability`, and `integrations`.

## Current boundaries

| Path | Responsibility | Main entrypoints |
|---|---|---|
| `agentic_rl/algorithms` | GRPO, DAPO, DIVE-PO, and LWM algorithm code | algorithm registry targets |
| `agentic_rl/environments` | terminal and benchmark runtimes | environment registry targets |
| `agentic_rl/harnesses` | Camel, Claude Code, and PRM harnesses | harness registry targets |
| `agentic_rl/rollout` | rollout orchestration and trajectory persistence | `entrypoint.py` |
| `agentic_rl/platform` | flat CLI, config, backend, paths, router, and worker infrastructure | `cli.py`, `slime_train.sh`, `worker_cli.py`, `router_cli.py` |
| `agentic_rl/misc` | flat reward, logging, sink, and integration helpers | `rollout_log.py`, `reward_rules.py`, `clawsentry.py` |

The public Python entrypoint is:

```bash
python3 -m agentic_rl.platform.cli --help
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
