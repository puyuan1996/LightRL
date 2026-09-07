# GLM-5.1 smoke runbook

This smoke uses the colocated, non-async GLM-5.1 path and expects the HF
checkpoint to advertise `model_type: glm_moe_dsa`.

## What must be true

- Ray head and all worker replicas are Running.
- The Ray cluster sees all 4 replicas and 16 GPUs before rollout starts.
- `CUDA_VISIBLE_DEVICES` mirrors `NVIDIA_VISIBLE_DEVICES` inside each replica.
- `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1`.
- `SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0`.
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,expandable_segments:False`.

## Compatibility note

The pinned Transformers build in this environment does not know
`glm_moe_dsa` out of the box. The repository root now ships
[`sitecustomize.py`](../../sitecustomize.py) to register a minimal
`AutoConfig` shim at Python startup so SGLang can load the checkpoint.

## Failure mode

If `ray status` shows only the head node and a pending 16-GPU placement group,
the worker replicas were not joined correctly. Start Ray on each non-head
replica again and re-check the cluster before restarting the job.
