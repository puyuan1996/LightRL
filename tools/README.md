# Utility scripts

Scripts are grouped by responsibility:

- `analysis/`: trajectory, metric, case-study, and DIVE-PO comparison tools.
- `evaluation/`: safety and SWE benchmark preparation, execution, and checks.
- `world_model/`: LWM dataset, probe, candidate, and latent-training workflows.
- `dev/`: worker smoke tests used during development.

Training entrypoints do not live here. Stable user commands are in the
repository-level `scripts/`, while versioned recipes are under
`experiments/`.
