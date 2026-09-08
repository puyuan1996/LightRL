# GLM-5.1 smoke runbook

Colocated, non-async GLM-5.1 (`model_type: glm_moe_dsa`). Detailed status and
resume guide: `local/records/report/glm51_smoke_status_2026-09-07.md`;
incident notes: `local/records/operations/rjob/glm51-smoke.md`.

## What must be true

- The Ray cluster sees all 4 replicas / 16 GPUs before rollout, and
  `CUDA_VISIBLE_DEVICES` mirrors `NVIDIA_VISIBLE_DEVICES` in each replica.
- `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1`,
  `SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0`.
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,expandable_segments:False`.
- `NCCL_NVLS_ENABLE=0` — the NVLS carve-out does not fit next to the ~133 GiB
  resident engine and aborts NCCL collectives.
- `SLIME_RELEASE_MEMORY_RETRIES=2`, `SLIME_RELEASE_MEMORY_RETRY_DELAY=2` —
  the release is a multi-process collective; retries plus per-attempt logs
  keep it observable.
- The launcher exports every `SGLANG_*` knob it relies on; `lib_launch.sh`
  forwards them to all Ray actors (`SGLANG_PASSTHROUGH_JSON`). Actors
  otherwise inherit the local raylet's shell env, which differs per node.

## Stack layout

The image's SGLang (0.5.5) and Transformers (4.57) predate GLM-5.1. Three
layers avoid rebuilding the image:

- Rollout HTTP servers run as subprocesses under the prebuilt `glm51-stack`
  venv (SGLang 0.5.10 + Transformers 5.3), selected by
  `LIGHTRL_SGLANG_SERVER_PYTHON`; trainer and Ray actors stay on the image
  stack. `ServerArgs` are re-filtered against the target build.
- Repo-root [`sitecustomize.py`](../../sitecustomize.py) registers a minimal
  `glm_moe_dsa` AutoConfig shim and a `TokenizersBackend`→fast-tokenizer
  fallback for Transformers 4.57 (no-op where native classes exist, e.g. in
  the venv).
- `megatron.bridge` comes from a GPFS overlay (fzyzcjy fork pin from
  `legacy-requirements.txt`) plus the GLM5Bridge port at
  `slime/slime_plugins/megatron_bridge/glm_moe_dsa.py`; `sitecustomize.py`
  appends `LIGHTRL_MEGATRON_BRIDGE_OVERLAY` to `sys.path` (namespace
  packages silently drop direct `__path__` appends).

## Colocate memory budget

GLM-5.1 is ~1.45 TB BF16. 16 GPUs (TP16, 90.8 GiB weights/GPU) is the floor;
32 GPUs (TP32) is comfortable. What makes 16 viable (verified in
`slime/train.py` and `slime/slime/backends/megatron_utils/`):

- `--colocate` defaults `offload_train`/`offload_rollout` on: Megatron actors
  `torch_memory_saver.pause()` to a CPU backup while the engine rolls out,
  and vice versa — the two sides are never both resident.
- Weight update streams params from the CPU backup in ~2 GiB buckets
  (`update_weight/common.py` `get_cpu_backup`): the update phase needs engine
  weights + one bucket on GPU, not 2x weights.
- LoRA (`--use-megatron-lora`) keeps optimizer/grad states negligible; the
  merge into base weights is in place per module during update.
- `recompute_granularity=full` + `micro_batch_size=1` keeps activations tiny.

TP16 per-GPU (H200 140 GiB, measured r21/r22): Megatron peak ~124.4 GiB
during init/load; engine steady state = 90.8 weights + KV pool + ~3.5
context.

## Lessons (r5-r22)

- `SGLANG_ENABLE_WEIGHTS_CPU_BACKUP=1` is required: without it the release
  endpoint returns 200 OK but never unmaps the weight pool, and Megatron
  OOMs against a still-resident engine (r11-r13).
- A weights-only release leaves the KV pool plus ~3.5 GiB of context
  resident. Size `SGLANG_MEM_FRACTION_STATIC` so
  `KV + 3.5 + Megatron peak < 140`: 0.74 missed by 16 MiB (r22); 0.68 gives
  ~8 GiB headroom and still ~40k KV tokens for a smoke.
- Do not release `kv_cache` on the 16-GPU profile: after a full KV pause the
  venv SGLang 0.5.10 crashes in `write_cache_indices` (Triton receives a CPU
  index tensor, r17). Optional fallback: `SGLANG_ATTENTION_BACKEND=
  torch_native`.
- `ray job logs -f` can drop its dashboard connection while the job still
  RUNS. The launcher retries `ray job status` and only an explicit terminal
  state triggers cleanup (r14).
- Both CPU backups total ~730 GiB per 4-GPU replica: request >= 900 GiB of
  pod memory and watch dmesg for memcg OOM kills.
- 12 GPUs cannot work (TP4xPP3 leaves ~121 GiB weights/GPU before
  activations; TP12 does not divide 64 attention heads). 16 is the floor.

## Failure modes

- Ray shows only the head node and a pending 16-GPU placement group: the
  worker replicas were not joined. Restart Ray on each non-head replica and
  re-check the cluster before restarting the job.
- On gang-scheduled clusters, individual replicas can die on node-level
  sandbox/network errors. Deleting the failed replica object (`brainctl -n
  ailab-narmodel delete replica <name>`) makes the controller recreate it
  without losing the replicas that are already Running.

## TODO / acceptance

1. First 16-GPU run: `NUM_ROLLOUT=4 SGLANG_MEM_FRACTION_STATIC=0.68
   slime/scripts/run-glm51-seta-dapo-16gpu.sh`. Accept on 4 consecutive
   `global_step` lines plus an explicit Ray job `SUCCEEDED`.
2. Still OOM at Megatron init: capture per-replica `nvidia-smi` samples,
   then retry with full release
   (`SLIME_RELEASE_MEMORY_TAGS=weights,kv_cache,cuda_graph`) plus
   `SGLANG_ATTENTION_BACKEND=torch_native`.
3. 16 GPUs infeasible: switch to the 32-GPU profile
   (`slime/scripts/run-glm51-seta-dapo-32gpu.sh`, `GLM_GPU_PROFILE=32`).
4. Watch the first update phase — it is the first real exercise of the
   sleep + CPU-backup streamed update + LoRA merge path.
5. After acceptance, bump to `NUM_ROLLOUT=10` for the full smoke.
