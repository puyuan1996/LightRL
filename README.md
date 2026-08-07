# LightRL

**LightRL is a lightweight, efficient and scalable RL post-training framework for agentic environments.**

中文文档见 [README_zh.md](README_zh.md)。

Experiments are composed explicitly along four axes:

```text
Harness × Model × Algorithm × Environment
```

- **Harness**: Camel-Agent, Claude Code CLI (`agentic_rl/harnesses/`)
- **Model**: Qwen3-8B, Qwen3-30B-A3B, GLM-5.1 (via configs/rollout templates)
- **Algorithm**: GRPO / DAPO (provided by the bundled Slime backend),
  DIVE-PO (LightRL-owned exploration extension)
- **Environment**: SETA terminal tasks, Agent-SafetyBench, AgentHarm, tau2,
  SWE-smith / SWE-verified (`agentic_rl/environments/`)

The training backends are the in-repo Slime and Megatron-LM checkouts.
Terminal environments execute in isolated Docker workers; GPU training
processes reach them over HTTP through `WORKER_URLS` (optionally behind a
local router when several workers are used).

## Features

- **Ready-to-run training recipes** — every experiment is a single reviewable
  shell script; inspect the exact backend command with `--dry-run` before
  anything starts.
- **Agentic environment support** — SETA terminal tasks, Agent-SafetyBench,
  AgentHarm, tau2, and SWE-smith / SWE-verified; terminal environments run
  isolated in dedicated Docker workers and are reached over HTTP.
- **Low-cost extension** — adding a benchmark environment is one declarative
  line in the environment registry plus a runtime class; harnesses and reward
  post-processors have equally explicit registration points.
- **DIVE-PO exploration** — episodic / lifelong novelty estimation with
  dual-stream advantage injection, toggled per recipe; GRPO / DAPO come from
  the bundled Slime backend.
- **Complete observability** — per-turn conversation trajectories, structured
  JSONL metrics, and wandb curves, all on by default.
- **Bounded validation scripts** — 4-GPU end-to-end smoke (rollout → reward →
  gradient updates) that confirms a deployment in minutes.

## Architecture

```text
examples/training/<recipe>.sh
  → agentic_rl/platform/slime_train.sh          # launcher: data prep, worker discovery, command assembly
  → slime/train_async.py                        # training backend (GRPO/DAPO)
      → agentic_rl/rollout/entrypoint.generate  # custom rollout hook
          ├─ environments/registry.py           #   local runtime vs remote Docker worker
          ├─ harnesses/factory.py               #   camel-agent / claude-code
          ├─ inference/sglang.py                #   token-level generation via sglang
          ├─ rollout/generate_steps.py          #   turn loop, scoring, exploration bonuses
          └─ rollout/sample_builder.py          #   reward shaping → Sample.reward["score"]
      → algorithms/dive_po/rewards/dual_stream  # custom reward post-process (group-normalized)
```

## Repository layout

```text
LightRL/
├── agentic_rl/
│   ├── algorithms/             # DIVE-PO (exploration, rewards, defaults) + PRM bonus agent
│   ├── data/                   # dataset converters and download helpers (standalone scripts)
│   ├── environments/           # registry.py, protocol.py, env runtimes, reward rules, HTTP client
│   ├── evaluation/             # SWE-bench official-format export
│   ├── harnesses/              # Camel-Agent / Claude Code harnesses + factory
│   ├── inference/              # sglang turn client library (shared by rollout and harnesses)
│   ├── misc/                   # rollout logging, JSONL sink, ClawSentry integration
│   ├── platform/               # slime launcher, worker/router services, paths, env parsing, http client
│   └── rollout/                # entrypoint hook, generate_steps, runner, admission, sglang factory, trajectory store
├── configs/rollout/            # rollout model templates (only retained config layer)
├── examples/                   # training recipes + bounded end-to-end validation
├── benchmarks/                 # benchmark data and task definitions
├── deploy/workers/             # Docker worker operations scripts
├── tools/                      # analysis, evaluation, diagnostics
├── tests/                      # unit tests (pytest)
├── slime/                      # training backend (third-party)
├── Megatron-LM/                # model training backend (third-party)
├── runs/                       # all runtime outputs (git-ignored)
└── docs/                       # architecture, algorithms, operations
```

## Requirements

- Python ≥ 3.10. Real training expects the prepared cluster image (CUDA,
  Slime, Megatron-LM, model checkpoints).
- SETA/terminal tasks require a separate Docker-capable CPU worker.
- GPU nodes must reach the worker service port (default `18081`).
- Site addresses, credentials and scheduling parameters belong in environment
  variables or the git-ignored `local/cluster/` — never in committed configs.

For a source-only install of the Python package:

```bash
python3 -m pip install -e '.[rollout,worker,train]'
python3 -c 'import agentic_rl'
```

This does not provision model weights, CUDA, or the training backends.

## Quickstart

### 1. Start a Docker worker (CPU host)

```bash
cd LightRL
bash examples/validation/start_docker_worker.sh
# prints WORKER_URLS=http://<WORKER_IP>:18081

curl --noproxy '*' --fail http://<WORKER_IP>:18081/healthz
```

See [docs/operations/cpu_workers.md](docs/operations/cpu_workers.md) for
daemon, proxy, disk and concurrency tuning.

### 2. Dry-run a recipe (GPU host)

```bash
bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_seta_dive_po.sh --dry-run
bash examples/training/train_qwen3_8b_mixed_dapo.sh --dry-run
```

### 3. Launch training

```bash
WORKER_URLS=http://<WORKER_IP>:18081 \
NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 TP_SIZE=2 ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
bash examples/training/train_qwen3_8b_seta_dapo.sh
```

### 4. Bounded end-to-end validation (recommended first)

```bash
WORKER_URLS=http://<WORKER_IP>:18081 NUM_ROLLOUT=3 \
  bash examples/validation/validate_4gpu_seta_dapo.sh

EXPERIMENT=dive_po WORKER_URLS=http://<WORKER_IP>:18081 \
  bash examples/validation/validate_4gpu_dive_po_or_mixed.sh

EXPERIMENT=mixed WORKER_URLS=http://<WORKER_IP>:18081 \
  bash examples/validation/validate_4gpu_dive_po_or_mixed.sh
```

Each script checks GPUs/resources, static compilation, imports, unit tests,
CLI dry-run, worker health, rollout metrics, and verifies finite, non-zero
actor updates (`TRAINING_METRICS_OK` / `EXAMPLE_VALIDATION_OK`).

## Configuration

Training configuration lives directly in the recipe scripts; environment
variables override defaults. Python-side environment parsing is centralized
in `agentic_rl/platform/env.py`, whose `ENV_VARS` table documents the
rollout-domain variables. See [docs/configuration.md](docs/configuration.md).

## Outputs

```text
runs/<RUN_ID>/
├── config/                # resolved config snapshot, generated dataset manifests
├── environment_outputs/   # env-side agent runner outputs
├── logs/                  # train.log, metrics.jsonl
├── trajectories/          # per-sample traj.json + index.jsonl (side-channel)
└── metrics/               # wandb / analysis
```

## Current validation status

Latest bounded validations on 4×H200 (2026-08-07, after the P0–P2 refactor
round; earlier rounds in [docs/manual_validation_20260731.md](docs/manual_validation_20260731.md)):

- SETA + DAPO: 3 rollouts, 6 actor train steps, non-zero finite updates
  (`TRAINING_METRICS_OK`).
- SETA + DIVE-PO: 3 rollouts, 7 actor train steps, 4 non-zero updates
  (`EXAMPLE_VALIDATION_OK`); trajectory artifacts exported.
- Mixed (SETA + Agent-SafetyBench + AgentHarm) + DAPO: 8 metric records,
  4 actor train steps, 4 non-zero updates (`EXAMPLE_VALIDATION_OK`).

These are short-horizon correctness checks, not convergence or benchmark
claims.

## Development

```bash
python3 -m pytest tests/ -q        # 219 tests
python3 -m compileall -q agentic_rl
```

Extension points:

- New environment → one `EnvSpec` in `agentic_rl/environments/registry.py`
  + a runtime implementing `environments/protocol.py:EnvClient`.
- New harness → register in `agentic_rl/harnesses/factory.py`
  (`_HARNESS_ALIASES` + `_HARNESS_TARGETS`) and satisfy the
  `rollout/runner.py:RolloutAgent` protocol.
- New reward post-processing → expose `post_process_rewards(args, samples)`
  and point `CUSTOM_REWARD_POST_PROCESS_PATH` at it.

## Documentation

- [docs/architecture.md](docs/architecture.md) — package boundaries and layering
- [docs/configuration.md](docs/configuration.md) — recipes and env vars
- [docs/algorithms/dive_po_dual_stream.md](docs/algorithms/dive_po_dual_stream.md) — DIVE-PO reward math
- [docs/harnesses/README.md](docs/harnesses/README.md) — harness selection
- [docs/evaluation/README.md](docs/evaluation/README.md) — evaluation tooling
- [docs/operations/](docs/operations/) — site-specific runbooks (our cluster;
  adapt addresses/paths to your site)

## Acknowledgements

LightRL builds on and bundles two training backends: [Slime](https://github.com/THUDM/slime)
(rollout/training runtime) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
(model training). The agentic RL stack — terminal environments, harnesses,
reward shaping and the rollout orchestrator — was originally developed inside
**OpenClaw-RL** and was extracted and refactored into this lightweight
framework. We thank the authors of all these projects.

## Citation

If LightRL helps your research, please cite:

```bibtex
@misc{lightrl,
  title={LightRL: A Lightweight, Efficient, Scalable RL Post-training Framework for Agentic Environments},
  author={Pu, Yuan and Zhang, Shaoang and Zhang, Chenhao and Li, Xueyan and Lu, Yudong and Tang, Jia and Wang, Guanchu and Niu, Yazhe},
  publisher={GitHub},
  howpublished={\url{https://github.com/opendilab/LightRL}},
  year={2026},
}
```

## License

[MIT](LICENSE)
