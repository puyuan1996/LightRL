# LightRL Architecture

LightRL keeps domain code visible and groups framework plumbing into two flat
packages:

```text
agentic_rl/
├── algorithms/      # optimization and exploration algorithms
├── data/            # dataset conversion and download utilities
├── environments/    # terminal and benchmark environments
├── evaluation/      # evaluation adapters and reports
├── harnesses/       # agent harnesses
├── inference/       # inference clients
├── models/          # model profiles
├── rollout/         # rollout orchestration and trajectory handling
├── platform/        # flat CLI, config, backend, runtime, router, and worker modules
└── misc/            # flat rewards, observability, and third-party integrations
```

`platform/` contains infrastructure shared by multiple domains. Its files are
named by purpose—such as `config_loader.py`, `worker_pool.py`, `router_app.py`,
and `slime_train.sh`—instead of being nested under one-file package layers.

`misc/` contains optional cross-cutting behavior: reward helpers, rollout
logging/formatting, JSONL sinks, and the ClawSentry client. These modules do not
define the main package architecture.

The domain packages remain separate because they are real extension axes:

- **Harness** controls how model output becomes environment actions. Add a
  harness under `agentic_rl/harnesses/` and register it in
  `agentic_rl/platform/registry.py`.
- **Model** describes checkpoint, tokenizer, and model-family defaults. Add
  profiles under `agentic_rl/models/` and config fragments under
  `configs/model/`.
- **Algorithm** owns optimization, exploration, and algorithm-specific reward
  behavior. Add implementations under `agentic_rl/algorithms/` and config
  fragments under `configs/algorithm/`.
- **Environment** owns task runtime semantics. Add implementations under
  `agentic_rl/environments/`.

The maintained training bridge is `agentic_rl/platform/slime_train.sh`.
`slime/` and `Megatron-LM/` remain vendorized backend trees and are not imported
while composing configuration.

DIVE-PO stays under `agentic_rl/algorithms/dive_po/` because its exploration
controller, episodic/lifelong state, and reward processing form one cohesive
algorithm rather than generic platform plumbing.

Site-specific worker URLs, credentials, and scheduler capacity are local state
under ignored `local/cluster/` paths or environment variables; they do not
belong in portable experiment recipes.
