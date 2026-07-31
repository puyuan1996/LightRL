# Remote worker (pool server)

This directory runs on the **CPU worker**: a pool server that manages Docker containers and executes terminal tasks on behalf of GPU training nodes.

For from-zero setup, hardening, watchdog, and recovery procedures, see the **operations runbook**: [`../../docs/operations/cpu_workers.md`](../../docs/operations/cpu_workers.md).

---

## Active scripts (current workflow)

### Setup & recovery (run on a fresh / broken CPU worker)

| Script | When to use |
|---|---|
| `setup_new_worker.sh` | Scenario A entry: first-time setup on a brand-new machine. Installs Docker/Compose, writes daemon config, hardens proxy/base images, installs watchdog, and verifies a build. |
| `fix_dockerd_and_proxy.sh` | Scenario B entry: one-shot recovery when Docker/proxy/build path is broken. Watchdog-aware; internally calls `prebuild_proxied_base_images.sh`. |
| `docker_worker_doctor.sh` | Log-aware diagnosis and repair wrapper. Use `diagnose` to analyze GPU train logs plus CPU worker Docker state; use `soft-repair` / `full-repair` for recovery. |
| `prebuild_proxied_base_images.sh` | Wraps the top base images with `apt.conf.d` proxy injection — mandatory in proxied environments because Ubuntu apt does not honor `HTTP_PROXY` env var |
| `restart_docker_force.sh` | Manual force-restart of dockerd (bypasses systemctl, used by watchdog and as escape hatch) |

### Steady-state (every training run)

| Script / file | Role |
|---|---|
| `run_pool_server_pu_v2.sh` | Hardened pool server launcher. Sources `/etc/seta_build_proxy.env`, sanity-checks dockerd, configures capacity, starts uvicorn |
| `agentic_rl/services/worker/app.py` | FastAPI service exposed on port 18081 |
| `agentic_rl/environments/terminal/runtime.py` | Environment client used by pool server |
| `agentic_rl/environments/terminal/docker_compose.py` | Helper to build / up / down compose stacks |

### Watchdog (recommended for >4h runs)

| File | Role |
|---|---|
| `docker_watchdog_v2.sh` | Main watchdog loop. Auto-restarts dockerd on hang; monitors pool_server `/healthz`; cleans address-pool exhaustion |
| `docker-watchdog.service` | systemd unit. Install with `systemctl enable --now docker-watchdog` |

### Manual ops

| Script | Role |
|---|---|
| `cleanup_docker_cache.sh` | Safe cleanup of build cache + stopped containers + dangling images (won't kill running) |

---

## Quick start (assuming machine already set up)

From the repo root:

```bash
# The launcher auto-sources /etc/seta_build_proxy.env when present.
bash deploy/workers/run_pool_server_pu_v2.sh
```

Then on the GPU worker:

```bash
export WORKER_URLS="http://<this-cpu-worker-ip>:18081"
bash examples/training/train_qwen3_8b_seta_dive_po.sh
```

For first-time setup and recovery, follow [`../../docs/operations/cpu_workers.md`](../../docs/operations/cpu_workers.md).

For a failed training run, start with a log-aware diagnosis:

```bash
bash deploy/workers/docker_worker_doctor.sh diagnose \
  --train-log /mnt/shared-storage-user/puyuan/code/LightRL/runs/<run>/logs/train.log
```

---

## Optional environment variables (read by `agentic_rl/services/worker/app.py`)

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | `benchmarks` | Path to the task dataset directory |
| `TBENCH_OUTPUT_ROOT` | `<server-run>/task_outputs` | Root for build/output artifacts under the worker's structured run log directory |
| `ENV_SERVER_PORT` | `18081` | Port the pool server listens on |
| `WORKER_MAX_TASKS` | `16` | Max tasks allocated per worker |
| `WORKER_MAX_RUNS_PER_TASK` | `8` | Max concurrent runs per task |
| `TBENCH_DOCKER_IMAGE_SOURCE` | `build` | `build` or `pull` — build locally or pull from registry |
| `TBENCH_DOCKER_PULL_PREFIX` | — | Image prefix for `pull` mode |
| `COMPOSE_OVERRIDE_PATH` | — | Optional Docker Compose override file |

Capacity sizing rule (from issue #3 §1):
```
WORKER_MAX_TASKS × WORKER_MAX_RUNS_PER_TASK ≥ rollout_batch_size × n_samples_per_prompt
```

For 8×4 (current default), 16×8=128 has been observed to saturate dockerd. v2 launcher defaults to a more conservative 64×16=1024 to leave headroom.

---

## Archived scripts

Historical one-off launchers were removed during the LightRL refactor; use Git history when an older implementation is needed.
