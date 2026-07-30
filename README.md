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
├── examples/                  # runnable model × data × algorithm recipes
├── benchmarks/                # benchmark data and task environments
├── tools/                     # analysis, evaluation, dev, world-model tools
├── deploy/                    # worker/cluster operations
├── tests/                     # package and smoke tests
├── artifacts/baselines/       # metadata for preserved reference runs
├── slime/                     # maintained training backend, including LWM WIP
├── Megatron-LM/               # vendorized model-training backend
└── docs/                      # architecture and operations guides
```

## Examples

Public examples are grouped under `examples/`:

```bash
# Qwen3-8B + SETA + DAPO
bash examples/train_qwen3_8b_seta_dapo.sh --dry-run

# Mixed data + DAPO
bash examples/train_qwen3_8b_mixed_dapo.sh --dry-run

# Qwen3-8B + SETA + DIVE-PO v0716 centered gate
bash examples/train_qwen3_8b_seta_dive_po.sh --dry-run

# GLM-5.1 + SETA + DAPO integration
bash examples/train_glm_5_1_seta_dapo.sh --dry-run
```

The maintained DIVE-PO recipe is
`configs/experiment/dive_po_qwen3_8b_seta.yaml`. Site-specific cluster
launchers, worker URL files, credentials, and scheduler settings belong under
`local/cluster/` and are intentionally ignored by Git.

See `examples/README.md` for required runtime inputs.

## LWM

LWM is marked WIP. Its algorithm integration boundary is
`agentic_rl/algorithms/lwm/`; the implementation imported from the `lwm`
branch lives in `slime/slime/world_model/`, with operational scripts grouped
under `tools/world_model/`.

See `docs/architecture.md` and `docs/configuration.md` for extension guidance.
