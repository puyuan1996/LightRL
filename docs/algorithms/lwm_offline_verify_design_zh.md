# JEPA Latent World Model：Agentic RL 离线验证设计

状态：`feat/lwm-offline-verify` 的实现规范（2026-09-04）

本分支以 [LightRL PR #2](https://github.com/puyuan1996/LightRL/pull/2)（来源
`MING-ZCH:jepa_wm`，本地起点提交 `3b52ef04`）的 JEPA 文本 latent world model
为起点，目标是验证它是否能从 Terminal-Bench 2.1（简称 tb2.1）完整轨迹中学到
对 Agentic RL 有用的状态转移表示。实现保持与现有 GRPO/DAPO 主干解耦，先做
可复现的离线证据，再通过显式 hook 进行可选在线辅助损失实验。

## 1. 先回答设计约束

### 1.1 Obs、action 与 latent

SETA 的一个 turn 定义为 `(h_t, a_t, o_{t+1}, h_{t+1}, r_t, d_t)`：

- `h_t`（observation/belief）：生成第 `t` 轮动作前的完整 `context_messages`，包括
  system、user、历史 assistant/tool-call 和 tool result。它是 agent 当前可见的
  belief，而不是容器内部不可见状态。
- `a_t`（action）：当前 assistant 的原始输出；若轨迹单独保存了解析后的工具调用，
  追加规范化的 `tool_name(sorted_json_args)`。因此既保留自然语言计划，也保留真正
  执行的工具参数。
- `o_{t+1}`（feedback）：本轮所有工具结果的规范化串；没有工具结果时使用包含
  `status`、`score` 和 `raw_score` 的终止摘要。tb2.1 ATIF 轨迹中的 observation
  对象会先转成稳定 JSON，避免把 Python 表示直接混入模型输入。
- `h_{t+1}`（next belief）：下一次 agent action 前的 context；终止 turn 没有
  next belief。`d_t` 对最后一个 action 或异常终止 action 为 true。
- `r_t`：优先读取 turn-level reward；只有终局分数时，将终局分数作为稀疏回报，按
  `gamma` 计算 return target。缺失 reward 的 transition 不参与 value loss。

对 policy LLM 的 hidden 使用一次 causal forward：

```text
tokens(h_t) | tokens(a_t)
     ^                 ^
 prompt-end       action span
     |                 |
 state hidden     pooled action hidden
```

state hidden 取 prompt-end（因 causal mask 看不到 action），action hidden 对 action
span 做 mean/last pooling。feedback 和 next-belief 使用 detached target forward。
三个 raw hidden 分支分别经过 Adapter，再由共享 Projector `C` 映射到同一 latent
空间：

\[
 z_t^s=C(A_s(H(h_t))),\quad e_t^a=A_a(H(h_t+a_t)),\quad
 z_{t+1}^o=C(A_o(H(o_{t+1}))).
\]

Adapter/Projector 是显式模块，便于后续换成视觉、结构化状态或多模态 encoder。

### 1.2 action 如何作用于 state latent

禁止把 action 作为 self-attention 的独立 token，也禁止简单地
`cat(state, action)` 作为默认融合。默认 predictor 是 action-conditioned AdaLN
Transformer：只有 state latent token 进入 self-attention，action latent 经过每层
线性映射生成 attention/MLP 的 shift、scale 和 residual gate。

```text
state latent tokens ──> self-attention ──> predicted feedback latent
             ^                    ^
             └── AdaLN(shift, scale, gate) <── action latent
```

这样 action 是条件而不是可被注意力绕过的 token；候选 action 可以在同一 state 上
批量评分。轻量 `mlp` 对照同样使用 FiLM/AdaLN 调制，仅改变 predictor 深度，不改变
action 融合契约。

### 1.3 主干解耦与梯度边界

- policy logits 仍由原有 Megatron GRPO/DAPO 路径计算，`world_model_loss_coef=0`
  时字节级保持 no-op。
- LEWM/JЕPA loss 只能通过显式 `apply_world_model_loss` hook 加到最终 scalar，
  其输入必须是 batch 中预计算的 `wm_pred_latents` 和 `wm_target_latents`；hook
  不自行重算 policy logits，也不改 advantage、reward 或 loss mask。
- 离线阶段默认冻结 policy LLM，仅训练 Adapter/Projector/Predictor/Value Head。
  `--backprop-to-llm` 是显式 opt-in，target branch 仍 detached；因此不会悄悄污染
  原始 RL 更新。在线 hook 默认关闭。

### 1.4 轨迹来源与 replay buffer

统一 loader 支持：

1. tb2.1 ATIF `*/trajectory.json` + 邻近 `verifier/reward.txt`；
2. SETA 原生 `*/traj.json`；
3. world-model records JSONL；
4. 已验证的 replay `.pt`。

`--data-source auto|tb21|seta|records|replay|mixed` 控制解析器。`auto` 始终先
按 tb2.1 轨迹排序，再加入 `--supplement-input` 指定的 SETA/training 数据；不会
把 archive 目录隐式混入。loader 按 trajectory 去重，支持 `max_trajectories`、
`max_transitions`、最小 turn 数、tool-feedback 过滤和终端样本保留，且输出
`data_manifest.json`（源文件路径、transition/task/terminal 统计）。

阶段一关闭 replay，按 trajectory 分组后固定 train/validation split。阶段二创建
固定容量 FIFO `TrajectoryReplayBuffer`，每 epoch 按配置比例随机抽样，成功和失败
transition 都保留；buffer 的容量、seed、每 epoch 样本数和是否混入 fresh train
indices 都可配置。这样可以做 `replay=off` 与 `replay=on` 的严格对照，而不是把
replay 当作另一个数据集。

### 1.5 Value Head 与 advantage

Value Head 接收 `pred_latent(s_t,a_t)`，学习折扣 return `G_t`（Smooth-L1），而非
直接复制环境文字。只有 reward 存在的 transition 参与 value loss；checkpoint 和
summary 记录 reward mask 数量、MAE、Spearman、校准误差。

本验证分支不默认替换 GRPO 的 critic-free group-relative advantage。阶段三先离线
验证 `V(s,a)` 的 return 预测，再提供一个 one-step latent MPC：对同一 `s_t` 的
候选 action hidden 批量预测 consequence/value，选择 `argmax(value - beta *
uncertainty)`，并报告 top-1 命中率和 regret。只有显式传入经过验证的 value
checkpoint 才能执行 MPC；因此 value 误差不会静默改变线上 policy。

## 2. 数据流与模块边界

```text
tb2.1/SETA/replay paths
          │
          ▼
  OfflineTransitionLoader ──> manifest + TerminalTransition
          │
          ├─ frozen HF policy / deterministic hash smoke
          │       ├─ state adapter ─┐
          │       ├─ action adapter ─┼─> shared latent projector
          │       └─ target adapter ─┘
          │                         │
          └──── AdaLN predictor(state, action condition)
                                    │
                pred / contrast / SIGReg / alignment / value losses
                                    │
                    checkpoint + JSONL metrics + predictions
```

代码职责：

| 模块 | 职责 |
| --- | --- |
| `slime/slime/world_model/seta_dataset.py` | 统一发现、解析、过滤和 manifest |
| `hidden_encoder.py` | policy hidden 的 prompt/action/feedback 边界 |
| `modules.py` | Adapter、shared Projector、AdaLN predictor、loss/head |
| `replay_buffer.py` | 有界、去重、可复现的 transition replay |
| `train_latent.py` | 三阶段训练、checkpoint、metrics、预测输出 |
| `mpc.py` | 同 state 候选 action 的 latent one-step planning |
| `loss_hook.py` | 与 GRPO/DAPO 的显式、default-off 辅助损失契约 |

模块不依赖具体环境执行器；后续 latent MCTS/MPC 只需实现候选 action encoder 或
替换 planner，不需要修改 predictor 和 policy loss。

## 3. 与 ECHO、Qwen-AgentWorld 的差异化定位

- [ECHO](https://github.com/microsoft/echo-rl) 在 SkyRL policy-gradient 路径中对
  选定的 environment-observation tokens 做 cross-entropy，并通过 hook 把辅助 loss
  接入 FSDP；其优点是目标直观、可直接生成文字 observation，代价是 token-level
  计算和长序列目标带来的显存/暴露面。这里借鉴“环境反馈是免费监督”和 default-off
  hook 思路，但不把 feedback 作为 policy 输出 token。
- [Qwen-AgentWorld](https://github.com/QwenLM/Qwen-AgentWorld) 是原生语言世界模型，
  从 CPT 到 SFT 再到 RL 三阶段训练，直接在文字空间模拟多个领域，并提供统一/解耦
  两种范式。它适合训练通用 simulator；本分支只验证 terminal-agent 的局部动力学，
  不要求大规模 CPT，也不让 simulator 取代真实环境。
- 本方案沿用 PR #2 的 JEPA/joint-embedding 思路：把 policy LLM span hidden
  压到受控连续 latent，预测 `feedback latent`，并用 SIGReg、shuffled-action
  对比和 next-belief alignment 约束几何。相比 token CE，latent 目标不必重建所有
  字符，能压缩长 terminal 输出、降低预测头开销，并允许同一 state 上并行比较候选
  action，适合未来 MPC/MCTS。代价是 latent 可解释性和解码能力较弱，必须报告
  retrieval、action shuffle gap、value calibration，不能只看训练 loss。

## 4. 三阶段实验协议与验收门槛

### 阶段一：无 replay 的 tb2.1 baseline

- 数据：tb2.1 完整 trajectory，优先 `runs/evaluation` 下的真实完整轨迹；若数量
  不足，通过 `--supplement-input runs/training/...` 明确补充。
- 训练：冻结 policy hidden，replay off，固定 seed 列表（默认 42/43/44），AdaLN
  predictor，value off。
- 记录：每 epoch 的 pred/contrast/SIGReg/alignment loss、effective rank、held-out
  MSE/cosine、shuffle gap，以及数据 manifest。
- 通过条件：finite loss、held-out loss 相对初始下降、effective rank 不塌缩、真实
  action 的误差优于 shuffled action；否则降低学习率/latent dim 或回退到 MLP 对照。

### 阶段二：replay + policy gradient opt-in

- 在相同 split 和 seed 下启用 replay，保存 buffer snapshot；比较 no-buffer 和
  buffer 的 wall-clock、samples/step、held-out loss 和 shuffle gap。
- `--backprop-to-llm` 与在线 `--world-model-loss-coef` 都必须显式开启；policy RL
  loss、reward 和 advantage 的日志单独记录，确保辅助项只是 additive scalar。
- 通过条件：replay 不引起 NaN/梯度爆炸，至少在固定预算下不劣于 baseline；若不提升，
  保留 replay 作为可复现实验而不宣称策略收益。

### 阶段三：value + latent MPC

- value head 用折扣 return 训练，报告 held-out MAE/Spearman 和按 reward 分桶校准。
- 对同 state 的候选 action 做 one-step MPC，报告 top-1 reward、regret、覆盖率；
  候选不足或没有 verified value head 时 fail-closed。
- 不将 MPC 决策自动注入 GRPO；只有单独 planner 实验显式消费规划结果。

所有阶段的入口均支持 `rjob`：站点只需在 `rjob` 容器中执行脚本并覆盖
`RUNS_ROOT/WM_INPUT/WM_HF_MODEL`，输出统一位于 `runs/training`（训练）或
`runs/evaluation`（评估），日志、曲线 JSONL、manifest、checkpoint 均保留。

## 5. 结果解释边界

低 hash-smoke loss 只能证明数据、shape 和优化闭环；它不代表语义 world model。
正式报告必须使用真实 policy hidden，并同时给出 tb2.1 task/trajectory 覆盖、terminal
比例、action 分布、held-out 指标、replay 对照、value calibration 和 MPC regret。
如果 policy return 没有提升，结论应写成“latent dynamics representation 有效/无效”
而不是将 offline prediction 误写为 Agentic RL 性能提升。
