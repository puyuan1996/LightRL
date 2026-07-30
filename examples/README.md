# LightRL runnable examples

All public training entrypoints live in this directory. Each wrapper delegates to
the composed YAML recipe under `configs/experiment/`; cluster-specific rjob
submission files remain under ignored `local/cluster/`.

| Script | Harness | Model | Data | Algorithm |
|---|---|---|---|---|
| `train_qwen3_8b_seta_dapo.sh` | Camel-Agent | Qwen3-8B | SETA | DAPO |
| `train_qwen3_8b_mixed_dapo.sh` | Camel-Agent | Qwen3-8B | SETA + Agent-SafetyBench + AgentHarm | DAPO |
| `train_qwen3_8b_seta_dive_po.sh` | Camel-Agent | Qwen3-8B | SETA | DIVE-PO |
| `train_glm_5_1_seta_dapo.sh` | Camel-Agent | GLM-5.1 | SETA | DAPO |
| `train.sh` | Config-selected | Config-selected | Config-selected | Config-selected |

Inspect any recipe without starting Ray or training:

```bash
bash examples/train_qwen3_8b_seta_dapo.sh --dry-run
bash examples/train_qwen3_8b_mixed_dapo.sh --dry-run
bash examples/train_qwen3_8b_seta_dive_po.sh --dry-run
bash examples/train_glm_5_1_seta_dapo.sh --dry-run
```

Real SETA runs require reachable workers via `WORKER_URLS` or
`WORKER_URLS_FILE`. Copy `configs/site/worker_urls.example.txt` to the ignored
`local/cluster/worker_urls.txt` and replace its placeholder when using the
default file location. The GLM entry also requires `HF_CKPT`, `REF_LOAD`, and a
compatible `MODEL_ARGS_FILE` present under `slime/scripts/models/`.
