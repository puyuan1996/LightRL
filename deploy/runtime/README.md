# Worker runtime assets

`deploy/runtime/` contains reusable runtime setup that is installed or sourced
on CPU/Docker workers: proxy rendering/configuration, image prewarm/build
helpers, systemd proxy units, and pinned worker dependencies.

These files do not submit RJobs and do not own Docker lifecycle recovery. Use
`deploy/workers/` for pool/DinD processes and `deploy/ops/` for diagnostics or
repair.
