# Worker operations

`deploy/ops/` contains explicit operator actions for diagnosing, repairing,
provisioning, cleaning, or force-restarting a CPU/Docker worker. Commands may
be destructive when repair flags are enabled; inspect their `--help` and
environment switches before running them.

The normal runtime entrypoints remain in `deploy/workers/`; RJob submission and
queue lifecycle helpers are local-only under `local/rjob/`.
