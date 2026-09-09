# SPEAR 在 lightrl 中的可比较基线

本文档记录 SPEAR self-imitation replay 在 lightrl 的兼容实现、与官方版本的
差异以及可复现实验入口。官方参考代码为
`/mnt/shared-storage-user/puyuan/code/archive/SPEAR`（`TrajectoryBufferBatch`、
`dp_actor.py` 和 `examples/grpo_trainer/run_*_spear.sh`）。实现复用了
off-policy GRPO/DAPO 的 behavior-policy log-prob、policy version、staleness
过滤和 decoupled policy loss 接口，再用一个独立的 SIL FIFO 保存高分完整
trajectory。

## 与官方实现的差异

| 部分 | 官方 SPEAR | lightrl 兼容实现 |
| --- | --- | --- |
| 轨迹载体 | 按 `DataProto` batch 保存，容量超过 `1.5×buffer_size` 后按 2 的幂次抽样 | 保持 `Sample`/现有 data-source 接口，固定容量 FIFO，并在现有 batch 中混入 SIL 样本 |
| admission | 以 group-relative GRPO advantage 过滤正样本，同时应用 reward threshold | 在 rollout 端计算 group-relative advantage；`--enable-trajectory-posadv` 开启时过滤非正样本；默认配方开启 |
| baseline | 保留最多 10,240 个历史 group reward，使用 p50 重算 replay advantage | `SILBuffer` 持久化同样的历史 reward，`-1` 模式优先使用历史 p50；无历史的旧 checkpoint 回退到当前 batch median |
| stale 轨迹 | `tolerate_steps=min(value,10)`，按 policy/global step 清理 | 相同上限与清理语义，年龄时钟使用 policy version，避免受 batch size 影响 |
| replay loss | 独立 replay dataloader，`is_replay=True`，loss coefficient cosine warm-up | 保持现有 actor 的 `sil_sample_flags` 覆盖接口，在混合 batch 中应用同样的 cosine warm-up；主 DAPO/off-policy correction 不改动 |
| 工具调用 intrinsic reward | 官方 recipe 可选 `use_toolcall_reward=cosine`（每次 tool call 奖励） | 不把工具奖励硬编码到通用 reward；SETA/TB2.1 的现有 verifier reward 原样使用，避免改变任务定义 |
| checkpoint | 官方保存 trajectory buffer 与 reward statistics | `replay_buffer_state_<rollout_id>.pt` 同时保存 FIFO、历史 p50、计数器和采样 RNG，旧状态可兼容加载 |

因此当前实现覆盖官方 recipe 在 lightrl 训练接口中可表达的核心语义；
`DataProto` 独立 replay loader 和工具 intrinsic reward 仍是明确的扩展点，未做
无关的训练流程重构。

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

TB2.1 转换数据的入口是
`examples/training/train_qwen3_8b_tb21_spear.sh`；它默认读取共享盘的
`tb21_full89.jsonl` 与 `tb21_env`，也可用 `TB21_DATASET` 和
`TB21_DATASET_DIR` 切换到 `tb21_smoke4.filtered.jsonl` 等子集。

提交前可用 `DRY_RUN=1 ... --dry-run` 检查最终命令。入口默认启用：

- `decoupled_policy_loss` 与 rollout behavior log-probs；
- `fifo_staleness` replay，容量 1000、最多滞后 2 个 policy version；
- trajectory SIL FIFO，分数至少为 `1.0`，并要求 group-relative positive advantage；
- 历史 group reward p50 baseline（`10240`）与 stale age window（`10`，上限）；
- 每个 rollout 额外训练 2 个 replay iteration；
- replay sample 默认保留并最多复用 4 次，确保额外 iteration 确实有数据；
- `loglinear` proximal log-prob 近似。

若要只比较普通 off-policy replay，可加
`ENABLE_TRAJECTORY_REPLAY=0`（通过 `EXTRA_ALGO_ARGS` 覆盖相应开关）。如需复现
旧版不筛正优势的运行，可设置 `ENABLE_TRAJECTORY_POSADV=0`；新实验建议保留
配方默认值。SPEAR 相关参数包括：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `trajectory-buffer-size` | 2048 | SIL 中最多保存的 trajectory 数 |
| `baseline-buffer-size` | 10240 | 历史 group reward 的保留上限，用于 p50 baseline |
| `trajectory-tolerate-steps` | 10 | SIL trajectory 最大 policy-step 年龄（实现上限为 10） |
| `trajectory-score-threshold` | 1.0 | admission 的最低 reward |
| `enable-trajectory-posadv` | true（配方默认） | 只保存 group-relative positive advantage trajectory |
| `weight-decay-trajectory-replay` | -1 | -1 用历史 p50 重算；`[0,1]` 对 stored advantage 乘一次系数 |
| `replay-loss-coef` | 0.001 | replay loss 的最终系数 |
| `max-replay-loss-steps` | 200 | replay loss cosine warm-up 步数 |

## 数据与 loss 语义

rollout 端只将完整 group 中满足 score threshold、具有 response loss mask
和 behavior log-prob 的样本送入 SIL。采样时保留原始 `policy_version`，并
重新计算或缩放 advantage；actor 端对 SIL 样本覆盖 batch advantage，同时仍
使用 decoupled loss 的 staleness、proximal ratio、importance sampling 和
可选 replay loss 系数。这样 baseline 与 latent WM A/B 共享同一 DAPO 主损失、
replay 容量和更新频率，差异只在 `world_model_*` aux loss。

replay 和 SIL 状态随 rollout checkpoint 保存到
`replay_buffer_state_<rollout_id>.pt`，恢复时会还原 FIFO 内容、计数器和本地
采样 RNG。旧 checkpoint 没有这些字段时会按空 buffer 兼容加载。

## tb2.1 + SPEAR 基线记录

TB2.1 数据已转换为 SETA-compatible terminal-env JSONL；任务 compose 树和
worker 需要从共享盘挂载。在线 smoke 已成功启动训练，但转换任务的部分
Dockerfile 缺少 Terminal-Bench 评估所需的 `tmux`，因此未产生有效分数。具体
尝试与已测结果见
[`spear_tb21_baseline_2026-09-09.md`](spear_tb21_baseline_2026-09-09.md)。

## 单 seed 验收

固定 seed、prompt 顺序、rollout 数、batch、硬件和 worker，只比较：

1. DAPO-only：`--loss-type policy_loss`（或现有 DAPO 配置）；
2. SPEAR：上面的入口，保持同样的 DAPO 超参，只打开 trajectory replay。

记录每个 rollout 的 reward、fresh/replay 样本数、有效 replay ratio、
`offpolicy/*`、`sil/*`、optimizer steps 和 wall-clock。报告达到相同 held-out
pass@1 或 reward 门槛所需的 fresh transitions、steps 和总时间；单 seed 只作
工程与方向性比较，不把随机性结论外推为统计显著性。
