# LightRL configuration

LightRL uses dependency-light YAML composition backed by typed dataclass
defaults. An experiment can inherit another YAML document with `extends`, then
override only the values that differ.

Configuration layers are:

1. typed global defaults in `agentic_rl/platform/config_schema.py`;
2. repository defaults in `configs/defaults.yaml`;
3. dimension configs under `configs/{harness,model,algorithm,environment,backend,cluster}`;
4. an experiment recipe under `configs/experiment/`;
5. command-line dotted overrides such as `cluster.num_gpus=4`.

Example:

```bash
python3 -m agentic_rl.platform.cli compose \
  --config configs/experiment/dive_po_qwen3_8b_seta.yaml \
  cluster.num_gpus=4
```

Training resolves `runtime.launcher`, converts `runtime.env` values into a
process environment, and then replaces the CLI process with the launcher:

```bash
python3 -m agentic_rl.platform.cli train \
  --config configs/experiment/dive_po_qwen3_8b_seta.yaml
```

Use `--dry-run` to inspect the command and exported environment without
starting Slime. Scheduler-specific settings are local state and belong under
the ignored `local/cluster/` directory, not in shared experiment configs.
