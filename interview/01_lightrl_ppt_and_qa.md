# LightRL：PPT 优化、讲稿和面试问答

原始材料：

- [LightRL 页面](<lightrl.png>)

核对材料：

- [LightRL README](<../README.md>)
- [LightRL Architecture](<../docs/architecture.md>)
- [DIVE-PO Dual-Stream](<../docs/algorithms/dive_po_dual_stream.md>)
- [Terminal-RL Latent World Model](<../docs/algorithms/lwm_guide_zh.md>)
- [SETA 训练吞吐分析](<../docs/performance/seta_training_efficiency_zh.md>)

> 当前仓库截至 README 记录的 2026-08-07 已完成 SETA+DAPO、SETA+DIVE-PO 和 mixed safety+DAPO 的 4-GPU bounded end-to-end smoke；这些结果证明 rollout→reward→gradient update 链路正确，不是收敛或 benchmark 结论。PPT 中应明确区分“已实现”“已 smoke”“内部训练趋势”和“已完成公开 benchmark”。

## 1. 当前页面的主要问题

这页试图在 LightRFT 能力图上增加 LightRL 延伸，但结果是两个项目的当前能力、历史系统图和未来研究方向叠在同一页，面试官很难判断 LightRL 到底已经实现了什么。

### 标题和项目身份错误

- 页面标题仍是“LightRFT—轻量、高效、全模态的强化学习微调框架”，但下方真正要介绍的是 Agentic RL 项目 LightRL；
- SafeWork-T1 的分层图与 Table 12 仍占据主要区域，它们不是 LightRL 的架构或性能 benchmark；
- LightRL 只在底部浅蓝色区域出现，视觉权重像未来规划而不是独立完成的项目；
- GitHub 链接使用个人仓库 `puyuan/LightRL`，而当前 README citation 与项目定位指向公开 LightRL 项目；面试前应确认最终公开链接。

### 当前贡献没有被结构化表达

- “Harness × Model × Algorithm 三维可扩展”数学上不完整。仓库明确定义的是 `Harness × Model × Algorithm × Environment` 四个轴；
- `harness`、`environment` 和 `training backend` 没有区分：harness 决定 agent 如何组织多轮交互，environment 决定状态、工具、生命周期和评分，Slime/Megatron 提供训练后端；
- 没有展示真实的数据流：recipe → Slime launcher → custom rollout → remote Docker env → reward/sample → DAPO/DIVE-PO update；
- “不同大小与类型 model”应列出当前配置路径中的 Qwen3-8B、Qwen3-30B-A3B、GLM-5.1，而不是笼统声称任意模型；
- “通用能力训练与安全能力训练协同提升”目前缺少收敛 benchmark，不应作为已证实结果；
- 当前可验证事实是 4-GPU bounded smoke、环境注册与多 harness 扩展点、trajectory/JSONL/W&B 可观测，以及正式执行栈的吞吐分析。

### DIVE-PO 与 latent world model 被写成模糊未来方向

- DIVE-PO 已有代码路径，不只是方向：它将 episodic、lifelong、policy-space 三维探索信号通过独立 intrinsic advantage stream 注入 DAPO；
- 但 DIVE-PO 论文草稿仍明确标注主实验与关键消融待补，不能宣称已经普遍提升 Agentic RL；
- 当前 policy-space arm 共享同一个 LLM backbone，主要改变 intrinsic weight 并使用很小的温度阶梯，更准确是探索预算分配器，不是 Agent57 多独立 policy 的等价复现；
- latent world model v2 已实现数据、replay、AdaLN predictor 和离线训练闭环，但默认是离线/shadow 诊断；它尚未接管 DAPO/GRPO advantage，也不默认修改 policy loss；
- “提升训练效率与稳定性”是研究目标，不是 world-model smoke loss 已经证明的结论。

### 缺少证据口径和个人贡献

- Table 12 是 SafeWork-T1 的 Qwen2.5-VL-7B、512×A800 workload，不能作为 LightRL 结果；
- 当前更适合报告的是 LightRL 自身执行栈：8×H200 下，同 Pod 私有 worker + 4×TP1 + 32 路环境并发相对共享外置 worker + 1×TP4 + cap6，前 66 步 rollout 均值由 1773.56s 降至 696.79s，即 2.55×；
- 该 2.55× 同时改变 worker 位置、引擎拓扑、并发、缓存与竞争条件，是系统级组合优化，不是单变量归因；
- 没有说明本人负责架构、环境、harness、DIVE-PO、world model 或性能优化中的哪部分。

## 2. 推荐页面结构

建议 LightRL 至少单独使用两页：

1. LightRL：Agentic RL 的统一训练与环境执行框架；
2. DIVE-PO 与 latent world model：已实现机制、当前证据与研究边界。

### 第 1 页：Harness × Model × Algorithm × Environment

推荐标题：

> LightRL：面向多轮工具智能体的可扩展 RL 后训练框架

推荐副标题：

> 将训练后端、agent harness、环境 runtime 与 reward/trajectory 组织成可审计闭环

推荐正文：

| 轴 | 当前能力 | 扩展接口 |
|---|---|---|
| Harness | Camel-Agent、Claude Code CLI | factory 注册并满足 `RolloutAgent` protocol |
| Model | Qwen3-8B、Qwen3-30B-A3B、GLM-5.1 配置路径 | rollout config/template |
| Algorithm | Slime 提供 GRPO/DAPO；LightRL 增加 DIVE-PO | custom reward post-process |
| Environment | SETA、Agent-SafetyBench、AgentHarm、tau2、SWE-smith/SWE-verified | `EnvSpec` + `EnvClient` runtime |

建议重画真实执行流：

```text
Training recipe
  → slime_train.sh / Slime async trainer
  → LightRL custom rollout hook
  → Harness 多轮决策 ↔ Docker Environment/Tools
  → verifier + exploration post-process
  → trajectory/sample → GRPO/DAPO update
```

页面底部只报告经过限定的两类证据：

- **链路正确性**：4-GPU bounded smoke 已覆盖 SETA+DAPO、SETA+DIVE-PO、mixed safety+DAPO，并产生非零有限梯度更新；
- **系统吞吐**：8×H200 正式执行栈的组合优化使前 66 步 rollout 平均时长改善 `2.55×`；注明它是多因素系统级对比。

### 第 2 页：研究机制与成熟度矩阵

推荐标题：

> 从“能训练 Agent”到“让 Agent 有效探索并理解环境反馈”

推荐正文：

| 模块 | 已实现机制 | 当前证据 | 尚不能宣称 |
|---|---|---|---|
| DIVE-PO | episodic SimHash-KNN、hierarchical decayed count、UCB beta arms、outcome gate、dual-stream advantage | 代码、单测、bounded smoke；内部共同窗口 raw reward 趋势 | 尚无完整公开主实验与消融，不说普遍优于 DAPO |
| Latent world model | 从多轮轨迹构造 `(h,a,o',h',r,d)`；policy hidden；action-conditioned AdaLN；replay/离线训练 | 数据—loss—checkpoint smoke 已闭环 | 尚未默认影响 policy advantage，不说已经提升 RL 性能 |
| 可观测与评测 | per-turn trajectory、JSONL、W&B、SWE export、固定协议与 throughput 分析 | 工具与验证脚本已存在 | 工具存在不等于 benchmark 已完成 |

页面底部增加个人职责：

> 我的职责：【LightRL 架构/重构】｜【环境与 harness】｜【DIVE-PO】｜【latent world model】｜【性能分析与集群执行】

必须按真实分工删减，并准备一个算法证据链和一个系统故障案例。

### 当前一页版本的最低成本修改

如果暂时只能保留一页：

1. 将标题改为“LightRL：面向 Agentic 环境的统一 RL 后训练框架”；
2. 删除 SafeWork-T1 架构图和 Table 12；
3. 上半页放四轴 capability matrix，下半页放真实执行流；
4. 右下角放 DIVE-PO 与 latent world model 两张“已实现/待验证”卡片；
5. 用 4-GPU smoke 与 2.55× 系统吞吐替代不属于 LightRL 的性能表；
6. 增加个人职责和公开仓库链接。

## 3. 两分钟介绍稿

LightRL 是面向多轮工具智能体的强化学习后训练框架。与单轮数学 RLVR 不同，Agentic RL 中一次 rollout 包含多轮 observation、reasoning、tool action 和 environment feedback；任务往往只在轨迹末端给出结果，环境还涉及容器生命周期、外部工具副作用和长尾延迟。因此问题不仅是换一个 policy loss，而是要把 agent harness、环境 runtime、推理、reward、trajectory 和训练后端组织成一个可扩展、可诊断的系统。

LightRL 将实验明确拆成四个轴：Harness、Model、Algorithm 和 Environment。Harness 决定模型如何组织多轮交互，目前支持 Camel-Agent 与 Claude Code；Environment 通过统一 registry 和 protocol 管理 SETA、Agent-SafetyBench、AgentHarm、tau2 和 SWE 类任务；模型由 rollout template 配置；GRPO/DAPO 使用内置 Slime 后端，LightRL 在其上增加 DIVE-PO。

执行时，训练 recipe 经过轻量 launcher 进入 Slime async trainer，并调用 LightRL 的 custom rollout hook。Agent 通过 SGLang 逐轮生成，远程访问隔离的 Docker worker 执行命令或工具；完整 context、tool result、状态和 verifier score 被保存为 trajectory，再转换成训练 sample。这个设计把第三方训练 backend 与我们新增的环境、harness、reward shaping 和可观测逻辑分开，便于审计每个样本是如何产生的。

在算法上，DIVE-PO 针对多轮稀疏奖励中的探索坍塌。它不直接污染 verifier score，而是把探索拆成 episodic、lifelong 和 policy-space 三个维度。前两个维度分别衡量单局内状态覆盖和跨训练历史的访问稀缺性，policy arm 用 UCB 分配不同 intrinsic 强度。Task reward 与 intrinsic signal 分别做 group normalization，最终在 advantage 空间合并，并用 outcome-aware gate 抑制“新颖但完全失败”的轨迹。当前共享 backbone 的 arm 更准确是探索预算分配，而不是多个独立策略。

我们还实现了 terminal latent world model：从真实轨迹构造历史、动作、下一环境反馈和下一历史，用 policy LLM hidden 建模 state/action/feedback latent，并通过 action-conditioned AdaLN predictor 预测动作后果。但当前它主要用于离线训练和 shadow 诊断，尚未默认接管 DAPO advantage，所以我不会用 smoke loss 宣称已经提升 policy。

系统验证方面，4-GPU bounded smoke 已证明 SETA+DAPO、DIVE-PO 和 mixed safety 路径能产生非零有限更新。性能上，针对 actor 长期等待 rollout 的瓶颈，我们将推理改为 4×TP1、提高环境并发并把 worker 移到同 Pod；在 8×H200 的前 66 个共同 step 上，rollout 平均时长从 1773.56 秒降到 696.79 秒，即 2.55×。这是组合系统优化，不是单一变量结论。

我主要负责【按真实分工填写】。我认为 LightRL 的核心价值是把 Agentic RL 的算法、环境和系统问题放进同一条可追踪数据链，并清楚区分已经跑通的能力和仍待公开实验验证的研究主张。

## 4. 高频问题与参考回答

### Q1：一句话概括 LightRL

LightRL 是把多轮 agent harness、可执行环境、轨迹/reward 和 Slime/Megatron 训练后端连接起来的 Agentic RL 框架，并在其上探索 DIVE-PO 与 latent world model。

### Q2：Agentic RL 与单轮数学 RLVR 最大的区别是什么？

数学 RLVR 通常是 prompt→response→verifier；Agentic RL 是多轮 observation→action/tool→environment feedback 的显式交互链，动作会改变外部状态，最终奖励可能延迟数十轮。它同时带来长链信用分配、探索坍塌、环境隔离和 rollout 长尾问题。

### Q3：为什么是四个轴，不是三维扩展？

仓库定义的是 `Harness × Model × Algorithm × Environment`。Harness 与 Environment 不能合并：同一个环境可以由不同 agent 编排方式驱动，同一个 harness 也可以适配多个环境。PPT 写“三维”会漏掉 environment 或错误合并概念。

### Q4：Harness 和 Environment 分别负责什么？

Harness 负责 prompt/context、工具调用循环和 agent 行为协议；Environment 负责状态、合法操作、容器生命周期、tool execution 和最终评分。两者通过统一 runner/protocol 交互，避免环境实现侵入训练后端。

### Q5：为什么环境要放在独立 Docker worker？

终端任务会执行不可信命令、创建文件和进程，必须与 GPU trainer 隔离；CPU worker 还可以独立扩缩。代价是 HTTP、容器 reset/close 和镜像构建带来延迟与故障面，因此需要 health check、admission control 和生命周期指标。

### Q6：多个 worker 为什么需要 router？

单个 `WORKER_URLS` 可由训练进程直连；多个 worker 或显式开启 pool server 时，router 负责选择后端、容量管理和健康状态。它不是训练算法的一部分，而是环境执行面的调度组件。

### Q7：LightRL 的真实训练链路是什么？

Recipe 调用 `agentic_rl/platform/slime_train.sh`，再进入 `slime/train_async.py`；custom rollout hook 构造环境和 harness，通过 SGLang 完成多轮生成，sample builder 将 verifier 与探索信号写入 sample，最后由 Slime 的 GRPO/DAPO 更新 Actor。

### Q8：哪些能力属于 LightRL，哪些来自第三方？

Slime 提供 rollout/training runtime 与 GRPO/DAPO，Megatron-LM 提供模型训练后端。LightRL 自己维护环境 registry/runtime、Camel/Claude Code harness、trajectory/评测适配、Docker worker 编排、DIVE-PO 和 terminal latent world model。面试时要按代码归属说明贡献。

### Q9：为什么每条 trajectory 都要完整落盘？

Agentic RL 的最终 score 无法解释中间哪里出错。保存 per-turn context、tool calls、observations、status 和 reward 允许复现样本、审计 reward hacking、分析长尾、构造 world-model transition，也能确认训练 sample 与真实执行是否一致。

### Q10：bounded smoke 证明了什么？

它证明部署环境中 rollout、environment、reward、sample builder、backward 和 optimizer step 能连通，并产生非零有限更新。它不证明模型收敛、最终 benchmark 提升、算法优于 baseline，也不等价于长时间稳定训练。

### Q11：LightRL 当前 smoke 覆盖哪些组合？

README 记录的 2026-08-07 验证包括 SETA+DAPO、SETA+DIVE-PO，以及 SETA+Agent-SafetyBench+AgentHarm 的 mixed DAPO；均在 4 GPU 的短运行中观察到有限非零 Actor 更新。

### Q12：DIVE-PO 试图解决什么？

同一 prompt 的多条 agent trajectory 容易过早收敛到类似命令序列，稀疏 terminal reward 又无法告诉模型哪些探索动作有价值。DIVE-PO 将探索信号分解并作为独立 advantage stream 注入，目标是在不修改 verifier score 的前提下改变组内策略梯度。

### Q13：DIVE-PO 的三个探索维度是什么？

Episodic 衡量同一 episode 内是否覆盖新状态/行为；lifelong 衡量该模式在跨 episode 历史中是否罕见；policy-space 通过不同 beta arm 和轻量 temperature ladder分配探索强度。前两者形成 intrinsic signal，policy arm 在 advantage 注入时控制强度。

### Q14：Episodic novelty 如何计算？

终端实例把 tool、command signature、observation fingerprint 和 exit bucket 组成状态，用 SimHash 向量和 KNN cosine distance 估计局内新颖性，再聚合到 trajectory。计算采用 compute-then-add，防止当前 action 立即污染自己的 novelty。

### Q15：Lifelong novelty 为什么使用分层衰减计数？

它在 task、skill 和 global 层面估计历史稀缺性，并对很久未访问的行为逐渐恢复 novelty。相比只在单局内看第一次出现，它能抑制“每局第一次但长期很常见”的动作。当前是 count-based approximation，不是 Agent57 的 RND。

### Q16：为什么不用 `task reward + novelty bonus`？

直接改 score 后再 group normalization，会把任务完成质量与探索启发式混在一起，难以判断提升是否来自真正完成任务，也容易鼓励 reward hacking。DIVE-PO 保持 verifier score 不变，分别归一化 task 与 intrinsic stream，再在 advantage space 合并。

### Q17：Dual-stream 具体是什么意思？

对同一 prompt group，task rewards 得到 `A_ext`，intrinsic signals 得到 `A_int`；两者各自做 group normalization，最后使用有界系数、beta arm 和 quality/reliability gate 构造 intrinsic perturbation并加到 task advantage。Dual-stream 是两个 advantage 流，不是两个 Actor 模型。

### Q18：为什么还需要 outcome-aware gate？

新颖不等于有用。反复 parse error、容器失败或完全偏离任务的轨迹可能非常“新颖”，却不应被强化。Gate 根据完成状态和 task outcome 降低低质量 novelty 的训练强度，同时保留少量可恢复探索空间。

### Q19：当前 policy-space arm 等价于 Agent57 多策略族吗？

不等价。所有 arm 共享同一个 LLM backbone，主要差异是 intrinsic coefficient、UCB 选择和很小的 temperature 变化。它更像探索预算分配器；若 arm 间行为分布并未分化，就不能声称获得独立 policy family 的覆盖收益。

### Q20：DIVE-PO 的 31.5% 可以怎么说？

本地分析在过滤 Docker/server 空运行后，对前 487 个共同有效 rollout-step 比较，raw reward 为 0.4018 对 0.3056，相对约 +31.5%。这只是内部 matched-window 趋势，仍可能受样本准入、时间段和运行条件影响；没有完整公开主实验、seed 和消融前，不应放大为 benchmark SOTA。

### Q21：DIVE-PO 当前还缺哪些关键实验？

至少需要公开 Agentic benchmark、多个 seed、三维分解消融、乘法/加法融合、score-space/dual-stream、gate、arm/UCB、reward hacking 审计、task/intrinsic 量级和长期稳定性；还要与同算力 DAPO 对齐 rollout acceptance 和环境成功率。

### Q22：Terminal latent world model 的输入输出是什么？

它从轨迹构造 `(h_t,a_t,o_{t+1},h_{t+1},r_t,d_t)`，用 policy LLM hidden 提取 state/action/feedback latent，并预测执行 action 后的 environment feedback latent；可选对齐 next-state latent 和预测 value。

### Q23：Action-conditioned AdaLN 如何工作？

State latent 作为 Transformer self-attention 的 Q/K/V；action embedding 不作为额外 token 拼接，而是为每层生成 AdaLN shift、scale 和 residual gates，从而条件化 dynamics predictor。Concat-MLP 只作为兼容与消融路径。

### Q24：World-model target 是固定 EMA teacher 吗？

不是。Feedback/next-state target forward 使用同一个 policy checkpoint并 detach；没有独立 EMA encoder。若开启 backbone 更新，target geometry 也会随 policy 参数变化，因此不能称为固定 teacher。

### Q25：World model 已经提升 DAPO 训练了吗？

当前不能这样说。默认路径是离线训练或 shadow 诊断，world-model value 尚未接管 GRPO/DAPO advantage；在线 hook 也只消费显式提供的 predicted/target latents。现有 smoke 证明数据、loss 与 checkpoint 闭环，不证明 policy return 提升。

### Q26：World model 的下一步最关键验证是什么？

使用真实 policy hidden 报告 held-out feedback prediction、shuffled-action gap、latent effective rank、reward calibration，并执行同一 state 下多个候选 action 的真实环境反事实对照。只有预测排序与真实后果相关，才适合进入 planning、reranking 或 advantage。

### Q27：为什么 rollout 是 LightRL 的主要吞吐瓶颈？

SETA 多轮任务需要模型生成、Docker tool execution、环境 reset/evaluate，并受到最慢 trajectory 的 batch barrier 影响。历史 8 卡运行中 Actor 每 step 真正训练只约 75–96 秒，却有 94%–95% 时间等待 rollout，因此先优化 Actor kernel收益有限。

### Q28：2.55× 吞吐提升来自什么？

前 66 个共同 step 中，系统从共享外置 worker、cap6、1×TP4 改为同 Pod 私有 worker、32 路环境并发、4×TP1，并结合本地链路、NVMe cache、镜像预热与减少共享竞争，rollout 均值从 1773.56s 降到 696.79s。它是整个正式执行栈的组合收益。

### Q29：为什么不能说“把 Docker 搬进 Pod 就快 2.55×”？

因为对比同时改变 worker locality、环境并发、推理 TP 拓扑、cache 和共享竞争，无法做单因素归因。要拆分贡献，应固定 checkpoint/seed，依次只改 worker 位置、cap 和 TP topology，并报告 queue、generation、tool 和 evaluator 分项。

### Q30：为什么不直接 fully async 避免最慢 trajectory？

Fully async 可以提高设备利用率，但会引入 policy staleness、不同难度样本的入批选择偏差和 reward 分布漂移，可能改变 DAPO/DIVE-PO 的算法口径。它应作为独立系统/算法实验，而不是无声明替换同步 baseline。

### Q31：安全任务如何与通用任务共同训练？

当前 mixed recipe 能把 SETA、Agent-SafetyBench 和 AgentHarm 接到统一 rollout/reward 路径，并已完成短 smoke。但“协同提升”必须通过分任务 benchmark、reward scale、采样比例、冲突分析和遗忘评估验证，不能由链路跑通推出。

### Q32：如何防止 agent reward hacking？

保留原始 verifier score、环境 status、tool trajectory 和 post-process 后 advantage；审计 novelty 与失败/超时/parse error 的相关性；对低质量 novelty 使用 gate；固定 evaluator；对可疑高分样本做 replay。只看训练 reward 上升是不够的。

### Q33：如何新增一个环境？

在 registry 增加一个 `EnvSpec`，实现 `EnvClient` protocol，并明确 local/remote、scoring mode、安全 reward 与 trajectory alias；再补 lifecycle、reward 和 dry-run/smoke 测试。理想情况下无需修改 Slime trainer 或其他环境逻辑。

### Q34：如何新增一个 harness？

在 harness factory 注册名称与惰性 import target，并实现 runner 需要的 `RolloutAgent` 协议；测试 prompt、tool routing、optional dependency isolation 和完整 trajectory。Harness 不应自行复制环境评分或 SGLang client。

### Q35：你个人到底做了什么？

准备两个案例：一个算法、一个系统。

> 算法上，我负责【DIVE-PO/world model 具体模块】，发现【信号漂移、探索坍塌或动作忽略】，通过【公式/消融/日志】修复。系统上，我负责【worker/harness/rollout/重构】，用【timeline、P95、GPU 等待比例】定位瓶颈，使【同口径指标】从【】变为【】。第三方 Slime/Megatron 与我的代码边界是【】。

### Q36：下一步研究与工程优先级是什么？

1. 完成 DIVE-PO 的公开主实验、seed、消融与 reward-hacking 审计；
2. 将 latent world model 从 shadow prediction 推进到候选 action reranking，再谨慎进入 advantage；
3. 拆分 2.55× 系统收益，优化 rollout P95 而非只看均值；
4. 建立 safety/general mixed training 的冲突和遗忘评测；
5. 评估 policy staleness 可控的 partial async；
6. 为 registry、harness、reward post-process 建立端到端 contract tests。

## 5. 过度表述检查

- 不再使用 LightRFT 标题介绍 LightRL；
- 不说“三维扩展”，应写 `Harness × Model × Algorithm × Environment` 四轴组合；
- 不把 SafeWork-T1 架构图和 Table 12 当作 LightRL 证据；
- 不说“支持任意 harness/model/algorithm/environment”，应列当前注册与配置路径；
- 不说“4-GPU 验证证明收敛”，它只证明端到端链路正确；
- 不说“通用能力与安全能力已经协同提升”，除非补完整 benchmark；
- 不说“DIVE-PO 已全面优于 DAPO”，当前公开主实验与消融仍待完成；
- 不说“policy arms 等价于 Agent57 多独立策略”，它们共享 backbone；
- 不说“dual-stream 完全不改变训练目标”，它不改 verifier score，但会改变最终 advantage；
- 不把内部 31.5% matched-window 趋势称为公开 benchmark 或因果结论；
- 不说“latent world model 已用于规划或提升 policy”，当前默认是离线/shadow；
- 不说“2.55× 来自单一 Docker locality 优化”，它是多因素系统级对比；
- 不把训练后端 Slime/Megatron 的贡献全部归为 LightRL 或个人贡献。

## 6. 面试前待本人补充

- [ ] 确认 LightRL 最终公开仓库 URL 与项目归属表述；
- [ ] 将 LightRL 从 LightRFT 页面拆成独立页或至少修正标题；
- [ ] 准备四轴各一个当前可运行示例；
- [ ] 明确本人负责的目录、PR、实验与合作者边界；
- [ ] 准备一个完整 trajectory 到 training sample 的字段级数据流；
- [ ] 准备 DIVE-PO task/intrinsic advantage 的分布、clip 与 gate 审计；
- [ ] 补 DIVE-PO 多 seed 公开主实验与关键消融后再强化算法结论；
- [ ] 准备 world-model shuffled-action gap、effective rank 和 held-out prediction 结果；
- [ ] 拆分 2.55× 中 worker locality、环境并发和 TP topology 的单变量贡献；
- [ ] 准备一次 Docker lifecycle 或 rollout 长尾故障的 timeline；
- [ ] 准备说明 bounded smoke、短训练、收敛和 benchmark 四种证据等级；
- [ ] 对 mixed safety/general recipe 补采样比例、reward scale、分任务结果与遗忘评估。
