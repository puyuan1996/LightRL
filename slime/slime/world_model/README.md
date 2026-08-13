# LightRL JEPA Latent World Model

本目录提供可插拔的 text latent world model。默认训练路径保持关闭；启用后可从
AgenticRL rollout 收集经过 redaction 的 turn transition，并在独立 offline replay
上训练 action-conditioned JEPA predictor。

阶段结果、结论边界和后续计划见
[`docs/algorithms/jepa_world_model_progress_zh.md`](../../../docs/algorithms/jepa_world_model_progress_zh.md)。
方法说明见 [`docs/algorithms/lwm_guide_zh.md`](../../../docs/algorithms/lwm_guide_zh.md)。

## 数据流

```text
SETA traj.json / records.jsonl / verified replay.pt
  -> TerminalTransition(h_t, a_t, o_{t+1}, h_{t+1})
  -> policy hidden at prompt-end and action span
  -> source adapters + shared projector
  -> AdaLN predictor(z_state, z_action)
  -> predicted feedback or next-belief latent
  -> retrieval, result-transfer and optional value diagnostics
```

## 主要模块

| 文件 | 作用 |
| --- | --- |
| `metadata.py` | 因果对齐、redaction、canonical hash、multi-interaction gate |
| `seta_dataset.py` | SETA、records JSONL、verified replay 数据适配 |
| `replay_buffer.py` | digest、RNG state、去重和 bounded replay |
| `state_view.py` | `belief_view_v1` 与完整上下文 state view |
| `action_view.py` | `tool_call_bundle_v1` action view |
| `result_view.py` | `result_only_v1` feedback target |
| `hidden_encoder.py` | prompt-end、action-span、feedback、next-state hidden |
| `modules.py` | shared projector、AdaLN/MLP predictor、SIGReg、value head |
| `train_latent.py` | split、cache、训练、checkpoint、prediction |
| `train_direct_latent.py` | parameter-matched raw-hidden Direct baseline |
| `train_result_transfer.py` | frozen JEPA latent 的 result-transfer probe |
| `offline_diagnostics.py` | retrieval、action controls、collapse diagnostics |
| `candidate_set_eval.py` | guarded observational candidate evaluation |

## 最小 smoke

`hash` encoder 只检查数据与训练闭环：

```bash
WM_TRAJECTORIES=/path/to/trajectories \
WM_OUTPUT_DIR=runs/world_model/hash_smoke \
WM_ENCODER=hash \
WM_MAX_TRAJECTORIES=8 \
WM_MAX_TRANSITIONS=32 \
WM_EPOCHS=1 \
bash tools/world_model/run_world_model_seta_latent.sh
```

冻结 policy hidden 的 next-belief 实验：

```bash
WM_TRAJECTORIES=/path/to/trajectories \
WM_OUTPUT_DIR=runs/world_model/qwen_next_belief \
WM_ENCODER=hf-policy \
WM_HF_MODEL=/path/to/Qwen3-8B \
WM_STATE_VIEW=belief_view_v1 \
WM_PREDICTION_TARGET=next_state \
WM_SPLIT_GROUP_KEY=task_id \
bash tools/world_model/run_world_model_seta_latent.sh
```

脚本默认 `local_files_only=true`、`trust_remote_code=false`。模型下载、remote code、
近似 action token 边界、旧 replay、未验证 value label 和 LLM backbone 更新均需要显式开关。
可通过 `PYTHON_BIN` 或兼容变量 `WM_PYTHON_BIN` 指定解释器，`PYTHON_BIN` 优先。

## Rollout 收集

```text
--world-model-enable
--world-model-use-dapo-replay-buffer
--world-model-replay-buffer-size 4096
```

该配置只增加 redacted metadata 和独立 replay snapshot。默认 policy loss、reward、
advantage 和环境执行保持原路径。多 interaction outer turn 缺少 harness adapter 时记录
`world_model_skipped`，避免生成错误 transition。

## 评测约束

- `training loss`、hash smoke、SIGReg 和 action sensitivity 只用于诊断。
- `pred_error` 使用真实 target，只能用于 oracle diagnostic。
- confirmatory eval 需要 group-disjoint split、cache/checkpoint provenance 和独立 test。
- execution accuracy 需要 verified atomic `status/error_type/exit_code` label。
- candidate selection 需要同一环境 snapshot 下的 alternative actions 或可恢复执行环境。
- `.pt`、checkpoint 和 replay 使用 PyTorch pickle loader，只接受可信来源。
