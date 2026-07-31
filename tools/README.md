# Utility scripts

Scripts are grouped by responsibility:

- `analysis/`: trajectory, metric, case-study, and DIVE-PO comparison tools.
- `evaluation/`: safety and SWE benchmark preparation, execution, and checks.
- `world_model/`: LWM dataset, probe, candidate, and latent-training workflows.
- `dev/`: worker smoke tests used during development.
- `validation/`: one-command local Docker worker and 4-GPU SETA+DAPO
  end-to-end validation drivers.

Training entrypoints do not live here. Stable user commands are in
repository-level `examples/`, while composed recipes are under
`configs/experiment/`.

The two refactor validation entrypoints are:

```bash
# On the local development/CPU worker host; remains in the foreground.
bash tools/validation/start_local_docker_server.sh

# Inside the already-running 4-GPU rjob.
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
  bash tools/validation/run_4gpu_seta_dapo_validation.sh
```

Both scripts are environment-variable driven so validation shapes and
infrastructure endpoints can be adjusted without editing the scripts.
