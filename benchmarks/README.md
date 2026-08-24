# Benchmark assets

This directory stores assets, not Python implementations. Runtime code lives in
`agentic_rl/environments/`, data preparation in `agentic_rl/data/`, and scoring
or result export in `agentic_rl/evaluation/` and `tools/evaluation/`.

```text
benchmarks/
├── datasets/       raw sources, converted JSONL, frozen splits, manifests
└── environments/   executable task directories, Dockerfiles, tests, fixtures
```

`metadata.task_path` remains relative to `benchmarks/environments/`; moving the
asset root therefore does not alter task identity or stored trajectories.
