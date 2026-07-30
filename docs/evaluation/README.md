# Evaluation utilities

Evaluation utilities are grouped under `tools/evaluation/`.

- Safety: input preparation, official evaluation orchestration, summaries,
  backend validation, and reward-scale validation.
- SWE: official SWE-bench Verified harness execution.
- Development checks: rule-reward and environment-backend validators.

These utilities consume existing checkpoints or run artifacts; they are not
training entrypoints. Run each command with `--help` (or read its shell usage)
before launching an external benchmark.
