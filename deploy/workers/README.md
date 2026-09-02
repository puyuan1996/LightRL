# Remote worker (pool server)

This directory runs on the **CPU worker**: a pool server that manages Docker containers and executes terminal tasks on behalf of GPU training nodes.

Site-specific proxy credentials must remain outside the repository. The
provisioning, watchdog, recovery, and private-DinD scripts are tracked here;
their credential-bearing environment files are still created only on workers.

---

## Active scripts (current workflow)

### Setup & recovery

| Script | When to use |
|---|---|
| `restart_docker_force.sh` | Manual force-restart of dockerd (bypasses systemctl, used by watchdog and as escape hatch) |
| `prewarm_seta_base_images.sh` | Prewarms task base images without embedding a site path. |
| `setup_new_worker.sh` | Provisions a new CPU/Docker worker. |
| `fix_dockerd_and_proxy.sh` | Repairs Docker and build/runtime proxy state. |
| `docker_worker_doctor.sh` | Diagnoses or repairs a CPU worker. |
| `fix_docker_overlay2_no_space.sh` | Handles Docker overlay2 storage pressure. |
| `docker_watchdog_v2.sh` / `start_watchdog.sh` | Monitors dockerd and pool lifecycle. |
| `start_rjob_dind_worker.sh` | Starts the private RJob dockerd and optional SETA pool. |

### Steady-state (every training run)

| Script / file | Role |
|---|---|
| `run_pool_server_pu_v2.sh` | Hardened pool server launcher. Optionally sources the build-proxy env, sanity-checks dockerd, configures capacity, starts uvicorn. Credential-isolated runtime-proxy deployments must set `SKIP_PROXY_ENV=1`. |
| `agentic_rl/platform/worker_app.py` | FastAPI service exposed on port 18081 |
| `agentic_rl/environments/terminal/runtime.py` | Environment client used by pool server |
| `agentic_rl/environments/terminal/docker_compose.py` | Helper to build / up / down compose stacks |

### Manual ops

| Script | Role |
|---|---|
| `cleanup_docker_cache.sh` | Safe cleanup of build cache + stopped containers + dangling images (won't kill running) |

---

## Quick start (assuming machine already set up)

From the repo root:

```bash
bash deploy/workers/run_pool_server_pu_v2.sh
```

Then on the GPU worker:

```bash
export WORKER_URLS="http://<this-cpu-worker-ip>:18081"
bash examples/training/train_qwen3_8b_seta_dive_po.sh
```

---

## Optional environment variables (read by `agentic_rl/platform/worker_app.py`)

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | `benchmarks` | Path to the task dataset directory |
| `TBENCH_OUTPUT_ROOT` | `<server-run>/task_outputs` | Root for build/output artifacts under the worker's structured run log directory |
| `ENV_SERVER_PORT` | `18081` | Port the pool server listens on |
| `WORKER_MAX_TASKS` | `16` | Max tasks allocated per worker |
| `WORKER_MAX_RUNS_PER_TASK` | `8` | Max concurrent runs per task |
| `WORKER_MAX_TOTAL_RUNS` | `WORKER_MAX_TASKS * WORKER_MAX_RUNS_PER_TASK` | Global reserved run/network ceiling across tasks; counts active leases plus leases whose async Docker cleanup is still retiring |
| `TBENCH_DOCKER_IMAGE_SOURCE` | `build` | `build` or `pull` — build locally or pull from registry |
| `TBENCH_DOCKER_PULL_PREFIX` | — | Image prefix for `pull` mode |
| `COMPOSE_OVERRIDE_PATH` | — | Optional Docker Compose override file |

Capacity sizing rule (from issue #3 §1):
```
The effective capacity is additionally capped by `WORKER_MAX_TOTAL_RUNS`; set it
below the host's available Docker address-pool capacity so retries queue at
`/allocate` instead of failing later during `docker compose up`. Capacity is
released only after close/force-cleanup completes, not when a lease merely
leaves the active map. Monitor `total_active_runs`, `total_retiring_runs`, and
`total_reserved_runs` in `/status`.
```

For 8×4 (current default), 16×8=128 has been observed to saturate dockerd. v2 launcher defaults to a more conservative 64×16=1024 to leave headroom.

---

## Archived scripts

Only the superseded `start_server_legacy.sh` remains under the repository-level
`backup/` directory. It is not used by the current `run_pool_server_pu_v2.sh`
workflow. The files that used to be under `deploy/workers/backup/` are active
worker operations and now live directly under `deploy/workers/` so their
documented and runtime paths agree.
