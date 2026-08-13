# World Model 训练入口

该目录提供 LightRL 的 LWM 示例入口。模型实现位于 `slime/slime/world_model/`，
AgenticRL 公共接口位于 `agentic_rl/algorithms/lwm/`。

## Offline smoke

`hash` encoder 用于检查数据、split、JEPA 训练和 artifact 导出：

```bash
WM_TRAJECTORIES=/path/to/trajectories \
WM_OUTPUT_DIR=runs/world_model/hash_smoke \
WM_ENCODER=hash \
WM_MAX_TRAJECTORIES=8 \
WM_MAX_TRANSITIONS=32 \
WM_EPOCHS=1 \
bash examples/training/world_model/train_seta_latent.sh
```

## Qwen next-belief

```bash
WM_TRAJECTORIES=/path/to/trajectories \
WM_HF_MODEL=/path/to/Qwen3-8B \
WM_OUTPUT_DIR=runs/world_model/qwen_next_belief \
bash examples/training/world_model/train_seta_next_belief.sh
```

默认配置使用 `belief_view_v1`、`next_state`、`has_next`、`task_id` grouped split 和
`best_validation` checkpoint。其他训练参数通过 `WM_*` 环境变量覆盖。

## Rollout metadata smoke

```bash
WORKER_URLS=http://worker:18081 \
WM_TRAIN_SCRIPT=examples/training/train_qwen3_8b_seta_dapo.sh \
bash examples/training/world_model/train_seta_metadata_smoke.sh
```

该脚本保存带 LWM metadata 的 debug rollout，不训练 auxiliary loss。

## Replay collection

```bash
WORKER_URLS=http://worker:18081 \
WM_TRAIN_SCRIPT=examples/training/train_qwen3_8b_seta_dapo.sh \
WM_REPLAY_BUFFER_SIZE=4096 \
bash examples/training/world_model/collect_seta_replay.sh
```

replay snapshot 写入训练 checkpoint 的 `rollout/world_model_replay_<rollout_id>.pt`。
该入口要求 `MAX_CKPT_KEEP` 为正数。所有 world-model 开关默认关闭。
