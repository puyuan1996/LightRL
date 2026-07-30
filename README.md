# LightRL

LightRL is a lightweight, extensible reinforcement-learning post-training
framework for agentic environments. Its extension model is explicit:

```text
Harness × Model × Algorithm
```

- Harness: Camel-Agent and Claude Code CLI.
- Model: Qwen3-8B, Qwen3-30B-A3B, and GLM-5.1.
- Algorithm: GRPO, DAPO, DIVE-PO, and LWM (work in progress).

## Repository layout

```text
LightRL/
├── agentic_rl/                # installable framework package
│   ├── harnesses/             # Camel / Claude Code / PRM adapters
│   ├── models/                # model profiles and family metadata
│   ├── algorithms/            # GRPO / DAPO / DIVE-PO / LWM extension points
│   ├── rollout/               # orchestration, admission, sampling, trajectories
│   ├── environments/          # terminal and benchmark runtimes
│   ├── inference/             # inference client factories
│   ├── rewards/               # shared reward primitives
│   ├── backends/slime/        # LightRL-to-Slime adapter and launch runtime
│   ├── services/              # router and worker services
│   ├── observability/         # rollout logging and sinks
│   ├── config/                # composition, schema, validation, snapshots
│   ├── core/                  # protocols, types, lazy plugin registry
│   ├── data/                  # dataset conversion utilities
│   ├── evaluation/            # evaluation/report adapters
│   ├── integrations/          # optional external integrations
│   └── cli/                   # agentic-rl command
├── configs/                   # composable axis/default/experiment configs
├── experiments/               # versioned reproducible recipes
├── benchmarks/                # benchmark data and task environments
├── scripts/                   # stable user-facing commands
├── tools/                     # analysis, evaluation, dev, world-model tools
├── deploy/                    # worker/cluster operations
├── tests/                     # package and smoke tests
├── artifacts/baselines/       # metadata for preserved reference runs
├── slime/                     # maintained training backend, including LWM WIP
├── Megatron-LM/               # vendorized model-training backend
└── docs/                      # architecture and operations guides
```

## DIVE-PO

The maintained DIVE-PO recipe is
`configs/experiment/dive_po_qwen3_8b_seta.yaml`. Inspect the fully composed
configuration:

```bash
python3 -m agentic_rl.cli compose \
  --config configs/experiment/dive_po_qwen3_8b_seta.yaml
```

Run it through the stable command:

```bash
CONFIG_PATH=configs/experiment/dive_po_qwen3_8b_seta.yaml scripts/train.sh
```

The preserved historical experiment filename is also executable:

```bash
bash experiments/dive_po/qwen3_8b_seta_v0716/run_terminal-rl_qwen3-8b_seta_dapo_dive_po_v0716_centered_gate.sh
```

Site-specific cluster launchers, worker URL files, credentials, and scheduler
settings belong under `local/cluster/` and are intentionally ignored by Git.

## LWM

LWM is marked WIP. Its algorithm integration boundary is
`agentic_rl/algorithms/lwm/`; the implementation imported from the `lwm`
branch lives in `slime/slime/world_model/`, with operational scripts grouped
under `tools/world_model/`.

See `docs/architecture.md` and `docs/configuration.md` for extension guidance.
