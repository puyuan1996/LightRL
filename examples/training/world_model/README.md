# Offline tb2.1 latent world-model verification

These entrypoints implement the three phases described in
[`docs/algorithms/lwm_offline_zh.md`](../../../docs/algorithms/lwm_offline_zh.md).
They are portable: the trainer only needs the repository, PyTorch, and (for
semantic runs) a local Hugging Face policy checkpoint.  All output directories
are explicit and can be mounted into an `rjob`.

## Local smoke or an in-job run

`run_tb21_lwm_phase.sh` gives tb2.1 `trajectory.json` priority and optionally
adds SETA training trajectories:

```bash
WM_PHASE=baseline \
WM_ENCODER=hash \
WM_INPUT=/mnt/shared-storage-user/puyuan/code/LightRL/runs/evaluation \
WM_MAX_TRAJECTORIES=32 \
bash examples/training/world_model/run_tb21_lwm_phase.sh
```

For a semantic run use `WM_ENCODER=hf-policy` and set `WM_HF_MODEL` to a local
Qwen3/compatible checkpoint.  The three phase labels are:

- `baseline`: replay off, value head off;
- `replay`: fixed-capacity transition replay sampled every epoch;
- `value_mpc`: discounted return value head plus one-step latent MPC report.

Each output contains `data_manifest.json`, `hidden_cache.pt` (when the policy
is frozen), `metrics.jsonl`, `latent_world_model.pt`, `predictions.jsonl`,
`run_summary.json`, and phase logs.  `value_mpc` additionally writes
`mpc_plan.json` when a value checkpoint and cache are available.

## Streaming A/B (online-style replay)

`run_tb21_lwm_stream.sh` runs the online-style comparison described in the
"流式 A/B" section of the design doc: the train split is cut into
trajectory-contiguous chunks ("rollout arrivals"), and the `noreplay` /
`replay` arms perform identical gradient-step counts per chunk — only batch
composition differs (`replay` mixes `WM_REPLAY_RATIO` samples from the FIFO
buffer with fresh chunk transitions).  Held-out metrics are recorded after
every chunk, so training efficiency is read as held-out loss versus
cumulative fresh transitions.  Use `compare_stream_runs.py` to aggregate
multiple seeded run directories into per-arm final losses, paired deltas,
steps-to-threshold, and guardrail checks.

```bash
WM_ENCODER=hash WM_MAX_TRAJECTORIES=12 WM_STREAM_CHUNKS=3 \
  bash examples/training/world_model/run_tb21_lwm_stream.sh   # local smoke

WM_ENCODER=hf-policy WM_HF_MODEL=/path/to/Qwen3-8B \
  bash examples/training/world_model/submit_tb21_lwm_stream_rjob.sh  # rjob
```

Outputs add per-arm `latent_world_model_{arm}.pt`, `predictions_{arm}.jsonl`,
`replay_buffer_replay.pt`, a shared `hidden_cache.pt`, per-chunk
`metrics.jsonl`, and `stream_summary.json`.

## rjob submission

Submit independent jobs for all phases (defaults are
`baseline,replay,value_mpc`):

```bash
WM_ENCODER=hf-policy \
WM_HF_MODEL=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
bash examples/training/world_model/submit_tb21_lwm_rjob.sh
```

Override `RJOB_IMAGE`, `RJOB_GPU`, `RJOB_CPU`, `RJOB_MEMORY_MB`, `RJOB_MOUNT`,
`RUNS_ROOT`, `WM_INPUT`, and `WM_SUPPLEMENT_INPUT` for the target cluster.
Use `WM_DRY_RUN=1` to print the submission plan without creating jobs.

## Online auxiliary contract

Rollouts may enable `--world-model-enable` and
`--world-model-use-dapo-replay-buffer` to attach metadata and persist an
isolated replay snapshot.  The native GRPO/DAPO loss remains unchanged unless
`--world-model-loss-coef > 0` and a custom adapter supplies
`wm_pred_latents`/`wm_target_latents`; the hook is additive and reports
`wm/*` metrics.
