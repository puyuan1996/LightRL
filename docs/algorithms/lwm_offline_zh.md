# LWM 离线验证：面向 Agentic RL 的 Latent 世界模型

> **摘要。** 本文描述 LightRL 中 LWM（Latent World Model，潜在世界模型）的
> 设计、实现与离线验证。LWM 复用 policy LLM 的 hidden state 作为
> state/action/feedback 表征，在统一连续 latent 空间中学习“执行动作 $a_t$
> 后环境返回什么反馈”，训练目标由预测损失、SIGReg 防坍塌正则、
> shuffled-action 对比与可选 value 头组成，并与 GRPO/DAPO 主干严格解耦
> （default-off）。验证采用四阶段协议：pooled baseline、replay 对照、
> value + latent MPC、流式（online-style）等算力 A/B。当前工程闭环（数据适
> 配、AdaLN 条件化、replay 生命周期、value/MPC 输出）已在无网络依赖的
> hash smoke 上验证通过；语义级结论需切换至真实 policy hidden 后按 §5 的
> 验收门槛判定。本文自包含，不依赖任何外部 PR 讨论或分支历史。
>
> 权威实现：`slime/slime/world_model/`；训练入口：
> `examples/training/world_model/`。

## 1. 概述（What & Why）

### 1.1 背景与动机

在 agent–environment 交互回路中，policy（state → action）与世界模型
（(state, action) → next observation）是两个互补组件，但当前 Agentic RL
研究几乎全部集中在 policy 侧。critic-free 的 GRPO/DAPO 目标只消费最终
reward，每条 action 的工具反馈文本——一种“免费的监督信号”——被完全丢
弃。

如果模型能在 latent 空间预测动作后果，就可以：在同一 state 上批量评分候
选 action（latent MPC）；训练 value head 辅助 return 估计；把环境反馈转
化为 policy 的辅助训练信号。本工作的目标是在不触碰 GRPO/DAPO 主干的前提
下回答一个可证伪的问题：**LWM 能否从 Terminal-Bench 2.1（tb2.1）真实轨
迹中学到对 Agentic RL 有用的状态转移表示**。

### 1.2 方法概览

给定 SETA turn 轨迹，LWM 构造 transition $(h_t, a_t, o_{t+1}, h_{t+1},
r_t, d_t)$（§2.2），用 policy LLM 的一次 causal forward 提取 state 与
action hidden，用无梯度 target forward 编码 feedback 与 next-state（§3.2），
经源特定 Adapter 与共享 Projector 映射到同一 latent 空间，再由
action-conditioned AdaLN Transformer 预测 feedback latent（§3.3）。训练
与验证分四阶段递进（§4.5、§4.6），所有结论都受 §5.4 的解释边界约束。

### 1.3 核心设计决策

1. **表征复用。** state/action/feedback 三类表征全部来自 policy LLM 自
   身的 hidden state，不引入独立文本 encoder；latent 空间与 policy 的
   “世界观”天然对齐。
2. **动作即条件（Action-as-Condition）。** action 不作为 token 进入
   self-attention，也不与 state 特征拼接，只通过 AdaLN 的
   shift/scale/residual gate 调制 predictor（§3.3），防止 predictor 学
   会忽略 action，并保证候选 action 可在同一 state 上并行评分。
3. **Default-off 主干解耦。** 在线辅助 loss 是显式 opt-in hook，系数为
   零时字节级 no-op；离线默认冻结 policy LLM（§3.5）。
4. **可证伪的 replay 协议。** 提供 pooled 对照与流式等算力 A/B 两种协
   议，把“replay 是否带来样本效率收益”变成可测量问题，而非默认假设
   （§3.6、§4.6）。
5. **Fail-closed 验证。** MPC 仅在显式传入经验证的 value checkpoint 时
   可用；数据不足时脚本在启动前报错退出，不会隐式混入 archive 数据。

> **为什么是 Latent 世界模型，而不是文本世界模型？**
> 文本世界模型（如 Qwen-AgentWorld）在 token 空间直接生成 next
> observation，目标直观、可解码，但长 terminal 输出带来显著的预测头开销
> 与显存暴露面，且逐字符重建中大量 token（时间戳、提示符样式）与决策无
> 关。Latent 路线把预测目标压缩到 policy hidden 张成的连续空间：不必重建
> 全部字符，可以在同一 state 上并行比较候选 action，适合未来接入
> MPC/MCTS。代价是 latent 不可直接阅读，因此本文要求所有结论都必须辅以
> shuffle gap、effective rank、value calibration 等可量化证据（§5.1），
> 不允许只看训练 loss。

### 1.4 与相关工作的关系

表 1 概括本实现与最接近的四类工作的借鉴点与差异；完整讨论见 §7。

**表 1：与相关工作的定位对比。**

| 工作 | 借鉴点 | 关键差异 |
| --- | --- | --- |
| LeWM / JEPA | joint-embedding 预测、SIGReg、AdaLN 条件化 | 像素 encoder 替换为 policy LLM span hidden；面向文本终端轨迹 |
| ECHO | “环境反馈是免费监督”、default-off hook | ECHO 优化 observation token 交叉熵且严格 on-policy；本实现预测连续 latent 并显式研究 replay |
| Qwen-AgentWorld | 世界模型的 (history, state, action) → next-observation 边界、多维评估思想 | 本实现是 latent 级局部动力学验证，不做大规模 CPT，不让 simulator 取代真实环境 |
| SPEAR | replay 纪律：有界 FIFO、陈旧度控制、权重 warmup | 不采用 advantage>0 质量闸——world model 需要完整 outcome 分布，成功与失败都入库 |

## 2. 问题形式化

本节先固定术语（§2.1），再给出 transition 的形式化定义（§2.2）与数据来
源约定（§2.3）。全文符号在此统一定义，后文不再重复说明。

### 2.1 术语约定

- **LWM 训练**：训练世界模型本身（本文全部内容）；**在线辅助训练**：把
  预计算的 LWM latent loss 以 additive hook 形式叠加到 GRPO/DAPO policy
  loss（§3.5）。两者严格区分。
- **LWM 命名辨析**：本文 LWM 均指 *Latent* World Model；Qwen-AgentWorld
  论文中的 LWM 指 *Language* World Model（token 空间世界模型），两者不
  同，引用时注意区分。
- **trajectory（轨迹）**：一个任务的完整多轮交互记录，由若干 turn 组
  成；**transition**：turn 级别的六元组（§2.2），是 LWM 的训练样本。
- **fresh transition**：流式协议中当前 chunk 首次到达的样本；**replay
  transition**：从 buffer 中抽样的历史样本。
- **held-out 集合**：按 trajectory 分组（而非按 transition 随机）切分
  出的验证集，防止同一轨迹的相邻 turn 同时出现在训练与验证中造成泄
  漏。单轨迹输入时自动退化为 train-only。

### 2.2 Transition 数据模型

SETA 的一个 turn 形式化为六元组 $(h_t, a_t, o_{t+1}, h_{t+1}, r_t,
d_t)$，字段定义见表 2。

**表 2：Transition 字段定义。**

| 符号 | 名称 | 来源 | 说明 |
| --- | --- | --- | --- |
| $h_t$ | belief observation | `turns[t].context_messages` | 第 $t$ 轮动作生成前的完整上下文（system、user、历史 assistant/tool-call、tool result），是 agent 可见的 belief，而非容器内部状态 |
| $a_t$ | action | `assistant_output` | assistant 原始输出；若轨迹保存了解析后的工具调用，追加规范化 `tool_name(sorted_json_args)`，同时保留自然语言计划与实际工具参数 |
| $o_{t+1}$ | feedback | `tool_calls[*].result` | 本轮全部工具结果的规范化串；无工具结果时使用含 `status`/`score`/`raw_score` 的终止摘要；ATIF observation 对象先转稳定 JSON，避免 Python 表示混入模型输入 |
| $h_{t+1}$ | next belief | `turns[t+1].context_messages` | 下一次 action 前的上下文（已含工具反馈）；终止 turn 无此字段 |
| $r_t$ | reward | `reward.per_turn_scores[t].score` | 优先 turn-level 分数；只有终局分数时作为稀疏回报按 $\gamma$ 折算 return target；缺失 reward 的 transition 不参与 value loss |
| $d_t$ | done | 轨迹终止标记 | 最后一个 action 或异常终止 action 为 true |

一条实际形态：

```text
h_t:     context_messages=[system, user: "检查当前目录"]
a_t:     assistant_output="..." + bash({"command":"pwd"})
o_{t+1}: <tool_result name=bash>\n/tmp\n</tool_result>
h_{t+1}: [system, user, assistant/tool-call, tool-result]
```

### 2.3 数据来源与统一加载

统一 loader（`seta_dataset.py`）支持四类输入，由
`--data-source auto|tb21|seta|records|replay|mixed` 控制：

1. tb2.1 ATIF `*/agent/trajectory.json` + 邻近 `verifier/reward.txt`；
2. SETA 原生 `*/traj.json`；
3. world-model records JSONL；
4. 已保存的 replay `.pt`。

`auto` 始终先按 tb2.1 轨迹排序，再加入 `--supplement-input` 显式指定的补
充数据；**不会**把 archive 目录隐式混入。loader 按 trajectory 去重，支
持 `max_trajectories`、`max_transitions`、最小 turn 数、tool-feedback 过
滤与终端样本保留，并输出 `data_manifest.json`（源文件路径与
transition/task/terminal 统计），保证每次运行的数据构成可审计。

## 3. 方法（How）

### 3.1 总体架构

图 1 给出从原始轨迹到损失与产物的完整数据流。

**图 1：LWM 离线训练数据流。**

```text
tb2.1 ATIF / SETA traj.json / records JSONL / replay.pt
        |
        v
OfflineTransitionLoader ──> data_manifest.json + TerminalTransition
        |
        +--> policy forward(h_t + a_t)
        |       +--> prompt_end hidden --> state adapter --+
        |       +--> action_span hidden --> action adapter  |
        |                                                   v
        +--> no-grad target forward(o_t+1) --> feedback adapter --> shared latent C
        +--> no-grad target forward(h_t+1) --> state adapter ----> shared latent C
                                                            |
state latent ------------------------------------------> Q/K/V
action latent --> per-layer AdaLN ---------------------> predictor
                                                            |
                                                            v
                                                predicted feedback latent
                                                            |
                            pred / contrast / SIGReg / alignment / value losses
                                                            |
                            checkpoint + metrics.jsonl + predictions.jsonl
```

### 3.2 表征提取：hidden 边界

HF policy 路径在一次 causal forward 中编码 $h_t + a_t$（图 2）：

**图 2：单次 causal forward 的 hidden 提取边界。**

```text
prompt tokens | action tokens
      ^              ^
 prompt_end      action_span
      |              |
 state hidden    pooled action hidden
```

- **state hidden**：`hidden[prompt_end]`；该位置因 causal mask 看不到
  action，是纯状态表征。
- **action hidden**：`hidden[action_span]` 的 mean 或 last pooling；它可
  看到 state 与已生成的 action。
- **feedback hidden**：无梯度 target forward 编码
  `<environment_observation> + o_{t+1}`。
- **next-state hidden**：无梯度 target forward 取 $h_{t+1}$ 的
  prompt-end；仅在 `has_next=true` 时参与 alignment。
- `hidden_layer` 可配置，默认 `-1`；离线 HF 路径可安全选择中间层。

raw hidden 不直接做 MSE：state 与 feedback 先经源特定 Adapter，再经共享
Projector $C$ 映射到同一 latent 空间；action 经独立 Adapter 得到条件向量：

$$
z^s_t = C(A_s(H(h_t))),\qquad e^a_t = A_a(H(h_t + a_t)[a_t]),
$$

$$
z^o_{t+1} = C(A_o(H(o_{t+1}))),\qquad
\hat z^o_{t+1} = P_{\text{AdaLN}}(z^s_t, e^a_t).
$$

Adapter/Projector 是显式模块，便于后续替换为视觉、结构化状态或多模态
encoder。**关于 target 分支的一个常见误解**：target forward 与 current
branch 共享同一 policy checkpoint，但计算图始终 detached——它不是另行加
载的独立文本 encoder，也不是固定 EMA teacher；开启 backbone 更新时
target geometry 会随 policy 参数漂移（§3.5）。

### 3.3 Action 条件化：AdaLN Predictor

默认 predictor 是 LeWM 风格的 action-conditioned AdaLN Transformer
（图 3）：

**图 3：AdaLN predictor——action 只生成每层的调制参数。**

```text
z_state tokens ───────────────> self-attention Q/K/V ──> predicted feedback latent
                                      ▲
e_action ─> SiLU + Linear ──> shift / scale / residual gates（每层）
```

设计约束：action **不**作为独立 token 进入 self-attention，也**不**与
state 做 `torch.cat([state, action], dim=-1)`。原因有二：

1. **防止 action 被绕过。** 若 action 是普通 token，predictor 可以学会
   忽略它，只拟合 state → feedback 的边缘分布；AdaLN 门控让 action 成
   为不可绕过的条件。
2. **保证候选并行性。** 同一 state latent 可以搭配整批候选 action 条件
   一次性评分，这是 latent MPC（§3.7）并行性的来源。

`--predictor-type mlp` 是同样采用 FiLM/AdaLN 调制（不拼接特征）的轻量对
照，仅改变 predictor 深度，不改变 action 融合契约；默认为 `adaln`。

### 3.4 训练目标

$$
L = L_{pred} + \lambda_{sig}\, L_{SIGReg} + \lambda_{cf}\, L_{action\ contrast}
+ \lambda_{align}\, L_{next/feedback} + \lambda_v\, L_{value}.
$$

**表 3：损失分量、作用与开关条件。**

| 分量 | 作用 | 开关条件 |
| --- | --- | --- |
| $L_{pred}$ | 预测 feedback latent 的回归损失 | 恒开；`--stop-grad-target` 控制 target 是否 detach |
| $L_{SIGReg}$ | Sketched Isotropic Gaussian Regularizer，只约束 state latent，防止低秩/常数坍塌 | 恒开，权重 $\lambda_{sig}$ |
| $L_{action\ contrast}$ | 真实 action 的预测误差应优于 batch 内 shuffled action，防止 predictor 忽略动作 | batch > 1 且 $\lambda_{cf} \neq 0$ |
| $L_{align}$ | 对齐 next-state latent 与 feedback latent | 存在 `has_next=true` 样本且 $\lambda_{align} \neq 0$ |
| $L_{value}$ | Smooth-L1 拟合折扣 return $G_t$ | 仅当 `--value-coef` 非零时创建并训练 value head（默认不创建）；只有 reward 存在的 transition 参与 |

### 3.5 主干解耦与梯度边界

- policy logits 仍由原有 Megatron GRPO/DAPO 路径计算；
  `--world-model-loss-coef 0`（默认）时辅助路径字节级 no-op。
- 辅助 loss 只能通过显式 `apply_world_model_loss` hook 叠加到最终
  scalar，其输入必须是 batch 中预计算的 `wm_pred_latents` 与
  `wm_target_latents`；hook 不自行重算 policy logits，也不改
  advantage、reward 或 loss mask。
- 离线阶段默认冻结 policy LLM（`no_grad` forward 并缓存 hidden），只训
  练 Adapter/Projector/Predictor/Value Head：显存占用低，target
  geometry 稳定，适合第一阶段。
- `--backprop-to-llm` 是显式 opt-in：state/action hidden 保留计算图，
  optimizer 以独立 `--llm-lr` 更新 backbone；feedback/next-state 仍为
  detached target。该模式显著增加激活与优化器显存。若要持久化更新后的
  HF backbone，需另加 `--save-updated-llm`；否则 world-model checkpoint
  不含更新后的 backbone，无法单独复现该次端到端训练的 hidden
  geometry。

### 3.6 Replay Buffer：契约与对照协议

`TrajectoryReplayBuffer` 的接口契约为 `push(entries, current_step)` /
`sample(n, current_step, baseline_reward)`，行为如下：

1. 固定容量 FIFO、按 transition 去重、随机 sample；默认成功与失败都入
   库——world model 需要完整 outcome 分布，不采用 SIL/SPEAR 式成功阈
   值过滤（可用 `score_threshold` 显式开启质量闸）。
2. 在线采集时，rollout 附加的 world-model record 进入 buffer，
   checkpoint 保存到 `${SAVE}/rollout/world_model_replay_<rollout_id>.pt`。
3. 离线训练可直接以 `train_latent.py --input <replay.pt>` 消费。

**两种对照协议的分工。** pooled 对照（阶段二）在同一 split、同一 seed
下比较 `replay=off` 与 `replay=on`：pooled 训练中 replay 只是对同一分布
重采样，预期不体现效率差异，其价值是验证 replay 生命周期不引入回归。
效率收益只能在数据分批到达的在线条件下测量，因此引入流式 A/B（阶段
四，§4.6）。在线训练期间，`online_learner.py` 以同一 replay 快照为数据
源做增量学习（§4.7），把离线验证过的训练配方接到真实 rollout 流上。

### 3.7 Value Head 与 Latent MPC

阶段三在 $\hat z^o_{t+1} = P(z^s_t, e^a_t)$ 上训练 value head 拟合折扣
return（Smooth-L1），checkpoint 与 summary 记录 reward mask 数量、MAE、
Spearman 与按 reward 分桶的校准误差。在此之上提供 one-step latent MPC：
对同一 $s_t$ 的候选 action hidden 批量预测 consequence/value，选择
$\arg\max(\text{value} - \beta \cdot \text{uncertainty})$，报告 top-1 命
中率与 regret。

**Fail-closed 保证。** 只有显式传入经过验证的 value checkpoint 才能执行
MPC（无 value head 时直接报错），value 误差不会静默
改变线上 policy；MPC 决策不自动注入 GRPO，只有独立的 planner 实验显式
消费规划结果。

## 4. 使用指南

### 4.1 环境依赖

- Python 3.12+ 与 PyTorch；`hash` encoder 的 smoke 不依赖 GPU、网络或外
  部 checkpoint。
- 语义实验（`WM_ENCODER=hf-policy`）需要本地 Hugging Face policy
  checkpoint（例如 Qwen3-8B），脚本以 `--hf-local-files-only` 强制只读
  本地文件。
- 运行 world-model 单元测试：

```bash
PYTHONPATH=slime:. python -m pytest slime/tests/world_model/ -q
```

### 4.2 快速开始：hash smoke

`hash` encoder 用确定性哈希构造伪 hidden，只验证数据适配、replay 生命
周期、AdaLN 条件化与 loss/prediction 闭环，**不提供语义结论**：

```bash
WM_TRAJECTORIES=/path/to/seta_trajectories \  # 含 */traj.json 的目录（脚本默认值为站点历史路径，须显式覆盖）
WM_USE_DAPO_REPLAY_BUFFER=1 \
WM_MAX_TRAJECTORIES=2 \
WM_MAX_TRANSITIONS=8 \
WM_EPOCHS=1 \
bash examples/training/world_model/train_seta_latent.sh
```

输出包含 `data_manifest.json`、`hidden_cache.pt`、`metrics.jsonl`、
`latent_world_model.pt`、`predictions.jsonl`、`run_summary.json`；启用
replay 时另有 `replay_buffer.pt`。

### 4.3 正式运行：冻结 policy hidden（推荐起点）

```bash
WM_ENCODER=hf-policy \
WM_HF_MODEL=/path/to/Qwen3-8B \
WM_TRAJECTORIES=/path/to/seta_trajectories \
WM_MAX_TRAJECTORIES=100 \
WM_MAX_TRANSITIONS=1000 \
WM_OUTPUT_DIR=runs/training/world_model_seta_latent/qwen_frozen \
bash examples/training/world_model/train_seta_latent.sh
```

### 4.4 端到端 backbone 训练（opt-in）

```bash
WM_ENCODER=hf-policy \
WM_BACKPROP_TO_LLM=1 \
WM_SAVE_UPDATED_LLM=1 \
WM_LLM_LR=1e-6 \
WM_HF_MODEL=/path/to/Qwen3-8B \
WM_TRAJECTORIES=/path/to/seta_trajectories \
WM_OUTPUT_DIR=runs/training/world_model_seta_latent/qwen_e2e \
bash examples/training/world_model/train_seta_latent.sh
```

建议先减小 batch/context，并使用 `--llm-lr 1e-6` 或更低；8B 全参数端到
端训练需要明显多于冻结模式的显存。

### 4.5 三阶段验证协议与集群提交

三阶段在同一 trajectory split 与 seed 列表（默认 42/43/44）上递进，各
阶段目的与通过条件见表 4。

**表 4：分阶段验证协议与验收门槛。**

| 阶段 | 目的 | 通过条件 |
| --- | --- | --- |
| `baseline` | 冻结 hidden、replay off、value off，建立 AdaLN predictor 基线 | finite loss；held-out loss 相对初始下降；effective rank 不塌缩；真实 action 误差优于 shuffled action（否则降低 lr/latent dim 或回退 MLP 对照） |
| `replay` | 启用固定容量 FIFO replay 并保存 buffer snapshot | 无 NaN/梯度爆炸；固定预算下不劣于 baseline；不提升则保留为可复现实验，不宣称收益 |
| `value_mpc` | 训练 value head 并运行 one-step latent MPC | 报告 held-out MAE/Spearman 与分桶校准；MPC 报告 top-1 reward/regret/覆盖率；候选不足或无 verified value head 时 fail-closed |

逐阶段本地运行：

```bash
WM_ENCODER=hf-policy WM_HF_MODEL=/path/to/Qwen3-8B \
  WM_PHASE=baseline bash examples/training/world_model/run_tb21_lwm_phase.sh
WM_ENCODER=hf-policy WM_HF_MODEL=/path/to/Qwen3-8B \
  WM_PHASE=replay bash examples/training/world_model/run_tb21_lwm_phase.sh
WM_ENCODER=hf-policy WM_HF_MODEL=/path/to/Qwen3-8B \
  WM_PHASE=value_mpc bash examples/training/world_model/run_tb21_lwm_phase.sh
```

集群提交器为三个阶段创建独立 RJob，并保留完整 stdout、`metrics.jsonl`、
manifest、checkpoint 与 MPC 结果：

```bash
WM_ENCODER=hf-policy WM_HF_MODEL=/path/to/Qwen3-8B \
  bash examples/training/world_model/submit_tb21_lwm_rjob.sh
```

提交前用 `WM_DRY_RUN=1` 打印提交计划（不创建任务）；用 `RJOB_IMAGE`、
`RJOB_GPU`、`RJOB_CPU`、`RJOB_MEMORY_MB`、`RJOB_MOUNT`、`RUNS_ROOT`、
`WM_INPUT`、`WM_SUPPLEMENT_INPUT` 覆盖目标站点配置。若训练数据或模型不
可用，脚本在启动前 fail closed，不会把 archive/debug 目录当作数据。

### 4.6 流式 A/B（online-style replay）

pooled 训练中 replay 只是对同一分布重采样，无法体现效率差异；“免费监
督”的收益只有在数据分批到达的在线条件下才可测。流式协议把 train split
切成按 trajectory 完整的 chunk（模拟 rollout 到达），两臂执行**相同**
gradient step 数与 batch 大小，仅 batch 组成不同：

- `noreplay`：batch 只含当前 chunk 的 fresh transition；
- `replay`：chunk 先全部 push 进固定容量 FIFO buffer，每个 batch 按
  `replay_ratio` 从 buffer 抽样、其余取当前 chunk 的 fresh transition；
  `--replay-warmup-chunks` 提供 replay 比例的线性 warmup。

两臂在每个 chunk 后于同一 trajectory 分组 held-out 集合上评估；模型权
重用同一 seed 初始化，冻结 encoder 时共享同一份 hidden cache，保证唯一
变量是 batch 组成。

```bash
# 本地 smoke
WM_ENCODER=hash WM_MAX_TRAJECTORIES=12 WM_STREAM_CHUNKS=3 \
  bash examples/training/world_model/run_tb21_lwm_stream.sh

# 集群提交（单 job 内顺序跑两臂，共享编码）
WM_ENCODER=hf-policy WM_HF_MODEL=/path/to/Qwen3-8B \
  bash examples/training/world_model/submit_tb21_lwm_stream_rjob.sh

# 跨 seed 汇总两臂结果（held-out pred loss、steps-to-threshold、guardrail）
python examples/training/world_model/compare_stream_runs.py RUN_DIR [RUN_DIR ...]
```

**通过条件**：在相同累计 fresh transition 数下，replay 臂 held-out pred
loss 不劣于 noreplay 臂；若 replay 臂用更少 fresh 样本达到 noreplay 最
终水平（steps-to-threshold 左移），记为样本效率收益；guardrail 为
effective rank 不塌缩、action_delta > 0、无 NaN（指标定义见表 7）。

### 4.7 在线采集与在线学习（DAPO rollout）

```bash
EXTRA_ALGO_ARGS="--world-model-enable \
  --world-model-use-dapo-replay-buffer \
  --world-model-replay-buffer-size 4096" \
bash examples/training/train_qwen3_8b_seta_dive_po.sh
```

这只收集/保存 world-model replay，不开启辅助 policy loss。在线 loss
hook 仍只消费显式提供的 `wm_pred_latents`/`wm_target_latents`。

rollout 侧的快照（`world_model_replay_<rollout_id>.pt`）可由在线学习器
`slime.world_model.online_learner` 消费：它轮询快照目录，按
`transition_id` 增量编码新到达的 transition，混合 replay buffer 抽样做
LWM 增量训练，并在固定 held-out probe 集上逐快照评估——这是 §4.6 流式
A/B 的在线对应物，也是让 replay 在 RL 训练期间真正产生 LWM 梯度的一环：

```bash
PYTHONPATH=slime:. python -m slime.world_model.online_learner \
  --snapshot-dir <SAVE>/rollout \
  --val-input /path/to/tb21_trajectories \
  --output-dir runs/training/lwm_online/<run_name> \
  --encoder hash   # smoke；语义实验用 --encoder hf-policy --hf-model ...
```

tb2.1 任务若需转换为在线训练运行时的 TB1 风格 env 目录（task.yaml /
compose / Dockerfile / tests），使用
`agentic_rl/data/convert_tb21_to_terminal_env.py`。

### 4.8 配置参数参考

离线训练 CLI（`python -m slime.world_model.train_latent`）常用参数见表
5。入口脚本以 `WM_*` 环境变量透传这些参数，对应关系见
`run_tb21_lwm_phase.sh` 与 `run_tb21_lwm_stream.sh` 顶部的默认值块（如
`WM_LATENT_DIM → --latent-dim`、`WM_STREAM_CHUNKS → --stream-chunks`）。

**表 5：离线训练常用配置参数。**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | 轨迹根目录或 replay `.pt` |
| `--data-source` | `auto` | `auto/tb21/seta/records/replay/mixed` |
| `--supplement-input` | 无 | 显式补充训练数据目录 |
| `--encoder` | `hash` | `hash`（确定性 smoke）或 `hf-policy` |
| `--hf-model` / `--hf-local-files-only` | — | 本地 HF checkpoint；只读本地文件 |
| `--latent-dim` | 128 | 共享 latent 维度 |
| `--predictor-type` | `adaln` | `adaln` 或轻量对照 `mlp` |
| `--phase` | `baseline` | `baseline` / `replay` / `value_mpc` |
| `--epochs` / `--batch-size` / `--lr` | 5 / 32 / 1e-4 | 优化超参 |
| `--value-coef` / `--gamma` | 0.0 / 0.99 | value loss 系数与折扣因子 |
| `--backprop-to-llm` / `--llm-lr` / `--save-updated-llm` | off | backbone 梯度 opt-in（§3.5） |
| `--use-dapo-replay-buffer` / `--replay-buffer-size` / `--replay-ratio` / `--replay-samples-per-epoch` | off / 2048 / 1.0 / 0 | replay 行为（§3.6） |
| `--max-trajectories` / `--max-transitions` | 不限 | 数据规模上限 |
| `--val-ratio` / `--seed` | 0.2 / 42 | 按 trajectory 分组的 held-out 比例与随机种子 |

slime 在线侧总参数（全部 default-off）：
`--world-model-enable`、`--world-model-loss-coef`、
`--world-model-mode {offline,shadow,online}`、
`--world-model-backprop-to-llm`、`--world-model-use-dapo-replay-buffer`、
`--world-model-replay-buffer-size`、`--world-model-loss-hook-path`、
`--world-model-metadata-max-chars`。

### 4.9 常见错误排查

**表 6：常见症状与处理。**

| 症状 | 原因与处理 |
| --- | --- |
| 启动即报“未发现轨迹” | 检查 `--input`/`WM_INPUT` 是否指向含 ATIF `*/agent/trajectory.json` 的目录；loader 不会隐式扫描 archive 目录 |
| 单轨迹时担心样本全部进入 validation | 已按 trajectory 分组 split；单轨迹输入自动退化为 train-only，无需手动干预 |
| `--backprop-to-llm` 后 OOM | 减小 batch/context，确认 `--llm-lr ≤ 1e-6`；8B 全参数端到端需要明显更多显存 |
| 复现端到端 run 的 hidden geometry 失败 | 该 run 未加 `--save-updated-llm`，checkpoint 不含更新后的 backbone；重新训练时必须开启 |
| 集群提交失败/无响应 | 先用 `WM_DRY_RUN=1` 校验提交计划；提交脚本只应在具备 RJob API 的站点环境执行 |
| hash smoke loss 很低但语义存疑 | 预期行为：hash hidden 只验证工程闭环，语义结论必须用 `hf-policy` 重跑 |

## 5. 验证协议与实验结果

### 5.1 评测指标定义

所有指标在按 trajectory 分组的 held-out 集合上计算（除非特别说明），
实现位于 `slime/slime/world_model/metrics.py` 与 `compare_stream_runs.py`。

**表 7：评测指标、定义与期望方向。**

| 指标 | 定义 | 期望方向 |
| --- | --- | --- |
| held-out pred loss | 验证集上 $\hat z^o$ 与 $z^o$ 的回归误差（MSE / cosine distance $1-\cos$） | 越低越好，且相对初始值下降 |
| shuffle gap | shuffled action 的预测误差减去真实 action 的预测误差 | 显著大于 0，说明 predictor 使用了 action 信息 |
| action_delta | 替换 action latent 后预测的平均 $L_2$ 位移：$\mathrm{mean}\lVert \hat z - \hat z_{shuf} \rVert_2$ | > 0 |
| effective rank | latent 矩阵奇异值分布的熵有效秩 $\exp\!\big(-\sum_i p_i \log p_i\big)$，$p_i$ 为归一化奇异值 | 不塌缩（远离 1，接近 latent dim 量级） |
| value MAE / Spearman | 折扣 return 预测的绝对误差与秩相关 | MAE 低、Spearman 高 |
| 分桶校准误差 | 按真实 reward 分桶后预测均值与实际均值的偏差 | 各桶偏差小且单调 |
| MPC top-1 / regret | MPC 所选候选的真实 reward 排名与最优差距 | top-1 高、regret 低 |
| steps-to-threshold | 流式 A/B 中 replay 臂首次达到 noreplay 臂最终 held-out loss 时的累计 fresh transition 数 | 左移（更少样本）记为效率收益 |

### 5.2 实验设置

当前已完成的是**工程闭环验证**：在无外部网络依赖的 hash-hidden smoke
上验证数据适配、AdaLN 条件化、replay 生命周期、value loss 与 MPC 输出。
smoke 输出写到独立临时目录，不污染仓库或运行中的训练目录。正式语义结
论必须将同样的命令切换到 `WM_ENCODER=hf-policy` 与本地 policy
checkpoint，并按表 4 的门槛验收。

### 5.3 工程闭环结果

**表 8：hash-hidden smoke 的三阶段结果（工程闭环证据，非语义结论）。**

| 阶段 | 数据 | 配置 | train loss | 结果 |
| --- | --- | --- | ---: | --- |
| baseline | tb2.1 ATIF，1 trajectory / 4 transitions | replay off，value off，latent=16 | 0.22406 | `metrics.jsonl`、cache、checkpoint、predictions 全部生成 |
| replay | tb2.1 ATIF，2 trajectories / 8 transitions | FIFO replay=16，1 epoch | 0.22309 | replay snapshot 与每 epoch 抽样工作正常 |
| value + MPC | tb2.1 ATIF，1 trajectory / 57 transitions | value coef=0.5，gamma=0.99，1 epoch | 0.42045 | value mask=1；`mpc_plan.json` 评分 57 个候选并选择 index 37 |

单元测试覆盖 tb2.1 ATIF 解析、trajectory split、planner fail-closed、
replay 生命周期与流式 A/B 协议（`slime/tests/world_model/`，运行命令见
§4.1）；全部 shell 入口通过 `bash -n` 语法检查。

### 5.4 结果解释边界

- 表 8 只证明数据、shape、AdaLN“action 只作条件”、replay、value 与
  one-step MPC 的**工程闭环**；hash hidden 不提供语义证据。
- 当前**不能**据此声称 Agentic RL return 提升；offline prediction 也不
  能被误读为策略性能结论。
- 流式 A/B 的结论只覆盖 LWM 训练效率，不外推为 policy return 提升；
  policy 级验证需要在线 latent 生产者，属于独立后续实验。
- 正式验收应比较同一 trajectory split 下的 held-out prediction、
  shuffle gap、effective rank、value MAE/Spearman、MPC regret，以及
  baseline/replay 的 wall-clock。只有这些指标稳定改善，才考虑把预计算
  latent hook 接入线上 DAPO。
- 若 policy return 没有提升，结论应写成“latent dynamics representation
  有效/无效”，而不是将 offline prediction 误读为 Agentic RL 性能提升。

## 6. 分析与讨论

### 6.1 hash smoke 能证明什么、不能证明什么

hash encoder 用确定性哈希把文本映射为伪 hidden，因此它**能**证明：四类
数据源的解析与去重正确、按 trajectory 分组的 split 无泄漏、AdaLN 条件化
的 shape 与梯度路径连通、replay 的 push/sample/checkpoint 生命周期闭
环、value 与 MPC 的 fail-closed 行为生效。它**不能**证明：latent 空间
承载了语义状态转移——哈希表征不含语义，任何 loss 下降都不构成语义证
据。这是本文把“工程闭环”与“语义结论”分开报告的原因。

### 6.2 已知限制

- SETA `traj.json` 没有保存原始 `input_ids`，HF 路径用同 tokenizer/chat
  template 重建 token 边界，不保证与历史 rollout bitwise 一致。
- Megatron 在线训练仍不默认暴露 middle-layer hidden；当前端到端
  backbone 梯度入口是独立的 HF latent trainer。
- world-model value 尚未接管 GRPO advantage；默认保持 DAPO critic-free
  行为。
- hash smoke 没有语义结论；正式结论必须使用 policy hidden，并报告
  shuffle gap、effective rank、held-out prediction 与 reward
  calibration。

### 6.3 未来方向

- 用真实 policy hidden 完成表 4 的三阶段验收与流式 A/B，补齐语义证据；
- 用 `online_learner.py` 在真实 rollout 流上复测流式 A/B 的效率结论；
- 候选 action 真实同-state 执行实验，验证 latent 并行评分的实际优势；
- 评估预计算 latent hook 对线上 DAPO 的辅助收益（§3.5 的 hook 契约已就
  绪）；
- 将 one-step MPC 扩展为多步 latent MCTS（只需替换 planner，不改动
  predictor 与 policy loss）。

## 7. 相关工作

- **LeWM（LeWorldModel）** 与 **JEPA**：借鉴 joint-embedding
  prediction、SIGReg 防坍塌正则与 action-conditioned AdaLN；把像素
  encoder 替换为 policy LLM span hidden。JEPA 谱系（I-JEPA、LeJEPA）提
  供了“在表征空间预测 + 正则防坍塌”的基本框架。
- **ECHO**：在 policy-gradient 路径中对选定的 environment-observation
  token 做交叉熵，并以 hook 接入 FSDP 训练。借鉴其“环境反馈是免费监
  督”与 default-off hook 思路；不把 feedback 作为 policy 输出 token。
  ECHO 严格 on-policy、无 replay buffer，本实现显式研究 replay 机制。
- **Qwen-AgentWorld**：原生语言世界模型，经 CPT → SFT → RL 三阶段在文
  本空间模拟多个领域。借鉴其世界模型的边界定义（历史上下文 + 当前状
  态 + action → 完整 next observation）与多维评估思想。本实现只验证
  terminal-agent 的局部动力学，不要求大规模 CPT，也不让 simulator 取
  代真实环境；其长 system prompt 适合作为未来离散生成 baseline 或
  latent decoder 的模板，不注入 policy hidden。
- **SPEAR**：自我模仿式 agentic RL，用 advantage>0 质量闸回放成功经
  验，并对回放权重做 warmup。借鉴其 replay 纪律（有界 FIFO、陈旧度控
  制、warmup）；world model 需要完整 outcome 分布，故不采用成功阈值过
  滤。

## 8. 结论

本文给出了 LWM 离线验证的自包含规范：一个复用 policy LLM hidden 表征、
以 AdaLN 注入 action 条件的 latent 世界模型；一套 default-off、
fail-closed 的主干解耦契约；一套四阶段、可证伪的验证协议。工程闭环已
在 hash smoke 上验证通过，语义级验收命令与门槛均已就绪，可在具备本地
policy checkpoint 的环境中直接复现。

## 附录

### 附录 A：模块与代码索引

**表 9：模块职责索引。**

| 模块 | 职责 |
| --- | --- |
| `slime/slime/world_model/seta_dataset.py` | 统一发现、解析、过滤 tb2.1/SETA/records/replay，输出 manifest |
| `slime/slime/world_model/hidden_encoder.py` | policy hidden 的 prompt-end/action-span/target 提取 |
| `slime/slime/world_model/modules.py` | Adapter、shared Projector、AdaLN predictor、loss/head |
| `slime/slime/world_model/metrics.py` | effective rank、cosine distance、action delta |
| `slime/slime/world_model/replay_buffer.py` | 有界、去重、可复现的 transition replay |
| `slime/slime/world_model/train_latent.py` | 三阶段训练、checkpoint、metrics、预测输出 |
| `slime/slime/world_model/stream_latent.py` | 流式（online-style）replay A/B 训练协议 |
| `slime/slime/world_model/online_learner.py` | 在线学习器：轮询 rollout replay 快照，增量训练并逐快照评估 |
| `slime/slime/world_model/mpc.py` / `plan_mpc.py` | 同 state 候选 action 的 latent one-step planning |
| `slime/slime/world_model/loss_hook.py` | 与 GRPO/DAPO 的显式、default-off 辅助损失契约 |
| `slime/slime/world_model/metadata.py` | rollout 侧轻量 transition metadata |
| `agentic_rl/data/convert_tb21_to_terminal_env.py` | tb2.x 任务目录 → 在线训练运行时 TB1 风格 env 布局 |
| `examples/training/world_model/train_seta_latent.sh` | 指定 SETA 轨迹的一键训练入口 |
| `examples/training/world_model/run_tb21_lwm_phase.sh` | 三阶段（baseline/replay/value_mpc）可复现入口 |
| `examples/training/world_model/run_tb21_lwm_stream.sh` | 流式 A/B 入口 |
| `examples/training/world_model/compare_stream_runs.py` | 跨 seed 汇总流式 A/B 结果 |

模块不依赖具体环境执行器；后续 latent MCTS/MPC 只需实现候选 action
encoder 或替换 planner，不需要修改 predictor 和 policy loss。

### 附录 B：参考文献

- LeWM（LeWorldModel: Stable End-to-End Joint-Embedding Predictive
  Architecture from Pixels）：<https://le-wm.github.io/>，
  <https://arxiv.org/abs/2603.19312>
- I-JEPA（Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture）：<https://arxiv.org/abs/2301.08243>
- LeJEPA（SIGReg 出处：Provable and Scalable Self-Supervised Learning
  Without the Heuristics）：<https://arxiv.org/abs/2511.08512>
- ECHO（环境反馈作为免费监督 / default-off 辅助 hook）：
  <https://github.com/microsoft/echo-rl>
- Qwen-AgentWorld（Language World Models for General Agents）：
  <https://arxiv.org/abs/2606.24597>，
  <https://github.com/QwenLM/Qwen-AgentWorld>
- SPEAR（Self-imitation with Progressive Exploration for Agentic RL）：
  <https://arxiv.org/abs/2509.22601>
- Terminal-Bench：<https://github.com/laude-institute/terminal-bench>
- Harbor（ATIF 轨迹格式规范）：<https://github.com/laude-institute/harbor>
