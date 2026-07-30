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
├── agentic_rl/
│   ├── harnesses/             # agent/harness integrations
│   ├── models/                # model-family integration points
│   ├── algorithms/
│   │   ├── dive_po/           # DIVE-PO defaults and reward processing
│   │   └── lwm/               # LWM integration boundary (WIP)
│   ├── backends/
│   │   └── slime_runtime/     # maintained Slime launch implementation
│   ├── experiments/
│   │   └── dive_po/           # runnable, versioned experiment entrypoints
│   ├── scripts/
│   │   ├── analysis/
│   │   ├── evaluation/
│   │   ├── world_model/
│   │   └── dev/
│   ├── data_utils/
│   ├── dataset/
│   └── tests/
├── configs/
│   ├── harness/
│   ├── model/
│   ├── algorithm/
│   ├── environment/
│   ├── backend/
│   └── experiment/
├── slime/                     # training backend
├── Megatron-LM/               # model-training backend
├── scripts/                   # stable user-facing commands
└── docs/
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
bash agentic_rl/experiments/dive_po/run_terminal-rl_qwen3-8b_seta_dapo_dive_po_v0716_centered_gate.sh
```

Site-specific cluster launchers, worker URL files, credentials, and scheduler
settings belong under `local/cluster/` and are intentionally ignored by Git.

## LWM

LWM is marked WIP. Its algorithm integration boundary is
`agentic_rl/algorithms/lwm/`; the implementation imported from the `lwm`
branch lives in `slime/slime/world_model/`, with operational scripts grouped
under `agentic_rl/scripts/world_model/`.

See `docs/architecture.md` and `docs/configuration.md` for extension guidance.
