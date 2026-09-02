# Benchmark entry points

Benchmark-specific scripts live in one directory per benchmark.  Keep
benchmark adapters, validators, report builders, and launchers together so a
new benchmark can be added without growing the top-level evaluation directory.

- `seta/` — fixed48 dataset construction, auditing, and paired-gate comparison.
- `safety/` — AgentSafetyBench, AgentHarm, ShieldAgent preparation, and safety reports.
- `swebench/` — official SWE-bench Verified harness launcher.
- `world_model/` — world-model batch, offline, stage-A, and candidate-set probes.

The old `tools/evaluation/<script>` paths remain as thin compatibility wrappers.
Use the paths in this directory for new commands and documentation.
