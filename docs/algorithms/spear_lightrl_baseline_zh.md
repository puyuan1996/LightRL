# SPEAR 在 lightrl 中的可比较基线

本文档记录 SPEAR self-imitation replay 在 lightrl 的最小可比较配置。实现
复用了 off-policy GRPO/DAPO 的行为策略 log-prob、policy version、staleness
过滤和 decoupled policy loss 接口，再用一个独立的 SIL FIFO 保存高分完整
trajectory。相关设计可对照
[OpenClaw-RL PR #16](https://github.com/puyuan1996/OpenClaw-RL/pull/16)。

## 运行

单 seed 的 SETA/Qwen3-8B 入口是
`examples/training/train_qwen3_8b_seta_spear.sh`。它要求先设置
`WORKER_URLS`，其余资源参数可以通过环境变量覆盖。例如：

```bash
WORKER_URLS=http://127.0.0.1:18081 \
  TRAJECTORY_SCORE_THRESHOLD=1.0 \
  TRAIN_ITERS_PER_ROLLOUT=2 \
  bash examples/training/train_qwen3_8b_seta_spear.sh
```

提交前可用 `DRY_RUN=1 ... --dry-run` 检查最终命令。入口默认启用：

- `decoupled_policy_loss` 与 rollout behavior log-probs；
- `fifo_staleness` replay，容量 1000、最多滞后 2 个 policy version；
- trajectory SIL FIFO，分数至少为 `1.0`；
- 每个 rollout 额外训练 2 个 replay iteration；
- replay sample 默认保留并最多复用 4 次，确保额外 iteration 确实有数据；
- `loglinear` proximal log-prob 近似。

若要只比较普通 off-policy replay，可加
`ENABLE_TRAJECTORY_REPLAY=0`（通过 `EXTRA_ALGO_ARGS` 覆盖相应开关）；若要
要求正优势轨迹，再设置 `ENABLE_TRAJECTORY_POSADV=1`。SPEAR 相关参数包括：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `trajectory-buffer-size` | 2048 | SIL 中最多保存的 trajectory 数 |
| `trajectory-score-threshold` | 1.0 | admission 的最低 reward |
| `trajectory-posadv` | false | 只保存正优势 trajectory |
| `trajectory-warmup-coef` | 0.0 | warmup 后混入 SIL 的比例 |
| `trajectory-weight-decay` | -1 | -1 用当前 batch baseline 重算，否则按 age 衰减 |

## 数据与 loss 语义

rollout 端只将完整 group 中满足 score threshold、具有 response loss mask
和 behavior log-prob 的样本送入 SIL。采样时保留原始 `policy_version`，并
重新计算或衰减 advantage；actor 端对 SIL 样本覆盖 batch advantage，同时仍
使用 decoupled loss 的 staleness、proximal ratio、importance sampling 和
可选 replay loss 系数。这样 baseline 与 latent WM A/B 共享同一 DAPO 主损失、
replay 容量和更新频率，差异只在 `world_model_*` aux loss。

replay 和 SIL 状态随 rollout checkpoint 保存到
`replay_buffer_state_<rollout_id>.pt`，恢复时会还原 FIFO 内容、计数器和本地
采样 RNG。旧 checkpoint 没有这些字段时会按空 buffer 兼容加载。

## 单 seed 验收

固定 seed、prompt 顺序、rollout 数、batch、硬件和 worker，只比较：

1. DAPO-only：`--loss-type policy_loss`（或现有 DAPO 配置）；
2. SPEAR：上面的入口，保持同样的 DAPO 超参，只打开 trajectory replay。

记录每个 rollout 的 reward、fresh/replay 样本数、有效 replay ratio、
`offpolicy/*`、`sil/*`、optimizer steps 和 wall-clock。报告达到相同 held-out
pass@1 或 reward 门槛所需的 fresh transitions、steps 和总时间；单 seed 只作
工程与方向性比较，不把随机性结论外推为统计显著性。
