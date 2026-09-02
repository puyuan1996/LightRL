# Deployment scripts

The deploy tree is deliberately split by execution context:

| Directory | Responsibility | Typical caller |
|---|---|---|
| `workers/` | Reusable Docker/SETA worker and private-DinD runtime | worker host or RJob Pod |
| `runtime/` | Proxy, dependency, image-preparation and systemd assets | worker provisioning |
| `ops/` | Diagnostics, repair, cleanup and worker provisioning actions | operator, usually root |
| `archive/` | Compatibility-only files retained after review | nobody by default |

RJob submission and queue lifecycle helpers are intentionally outside this
public deploy tree, under `local/rjob/`, because they contain site-specific
cluster control-plane behavior and are not deployment payloads.

The stable worker entrypoint is:

```bash
bash deploy/workers/run_pool_server.sh
```

For a private DinD RJob, use the operator-side launcher in
`local/rjob/submit_private_dind.sh`; the resulting Pod calls
`deploy/workers/start_rjob_dind_worker.sh`.
