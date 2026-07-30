# DIVE-PO: Decomposed Intrinsic adVantage Exploration for AgenticRL

**面向 AgenticRL 的三维正交探索分解与双流优势注入**

v2 draft, 2026-07-07

## 摘要

AgenticRL 将语言模型置于多轮环境交互中训练：模型观察自然语言状态、调用工具或执行命令、接收环境反馈，并在长程稀疏奖励下优化策略。相比单轮数学 RLVR，AgenticRL 更接近显式 MDP：工具调用就是 action，终端输出和环境反馈构成 observation，任务完成度通常只能在轨迹末端由 verifier 或 evaluator 给出。因此，探索坍塌在 AgenticRL 中更突出：同一 prompt 下的多条 rollout 容易过早收敛到相似命令序列，直接把探索 bonus 加到任务分数又会污染 verifier reward。本文提出 **DIVE-PO**，一种面向 AgenticRL 的多维内在探索框架。DIVE-PO 的核心论点是：agentic exploration 不应被建模为单一 novelty bonus，而应分解为三个正交维度：局内状态覆盖（episodic）、跨 episode 历史访问（lifelong）和策略空间探索强度（policy-space）。这些维度通过任务自适应的轻量估计器得到轨迹级 intrinsic signal，并以乘法方式融合；为了使该分解在 group-based policy optimization 中成立，DIVE-PO 将 intrinsic signal 作为独立归一化的 advantage stream 注入 DAPO，而不是修改 verifier score。我们以终端智能体为主要实例：SimHash-KNN episodic novelty、hierarchical decayed count lifelong novelty、共享 backbone 下的 beta-arm/UCB policy allocator，以及 outcome-aware gate。数学 RLVR（MATH、AMC23、AIME24、AIME25）被用作泛化性验证：它检验三维分解能否在非工具调用任务中通过替换状态抽取器继续工作，但不作为本文主场景。[TODO: 填入主实验与消融结果]

## 1 引言

语言模型智能体在终端、网页、软件工程和工具使用环境中需要执行连续决策：读取文件、运行命令、解析反馈、修复错误，并在有限 turn budget 内完成目标。这类任务是典型 AgenticRL：状态由上下文、工具输出和环境反馈组成，action 是自然语言回复、工具调用或 shell 命令，环境 transition 由外部系统执行，奖励通常稀疏、延迟且只在轨迹末端可验证。失败轨迹高度相似，标准 group-based policy optimization 容易在训练早期强化少数高频动作模式。

数学 RLVR 也存在模板坍塌和 pass@k 多样性不足的问题，但它通常没有真实环境 transition、工具副作用和多轮 observation-action loop。因此本文把 MATH、AMC23、AIME24、AIME25 定位为泛化性验证，而不是主要问题设定。DIVE-PO 的主论点来自 AgenticRL：长程稀疏奖励、多轮工具调用和可执行 action 使探索更困难，也更需要显式分解。

探索奖励是缓解这一问题的自然工具，但 AgenticRL 的探索设计不同于传统低维控制、Atari 或单轮数学 RLVR。第一，状态不是低维环境变量，而是由自然语言、工具调用、观察文本、退出状态和任务上下文共同构成。第二，action 具有真实环境副作用：错误命令、无意义读写、重复查询和超长输出会消耗 turn budget，甚至改变后续状态。第三，低质量的“新颖”行为很容易被奖励，例如无意义 echo、参数扰动、重复查看文件、格式失败或错误推理模板。第四，verifier reward 本身具有任务语义；若把内在奖励直接加到 scalar score，组内归一化后的优势会混合任务完成质量和探索启发式，既难解释，也容易偏离真实目标。

本文的出发点是：AgenticRL 中的“探索”不应被当作一个单一标量 bonus，而应拆成几个互补但正交的维度。局内维度回答“同一 episode 中当前轨迹是否覆盖了新的行为/状态”；局间维度回答“跨训练历史看，这类行为是否长期少见”；策略维度回答“当前 rollout 是否来自不同探索强度或采样族”。单一 novelty bonus 很难同时表达这三件事。例如，一个命令在当前 episode 中第一次出现但在全局历史中极常见，不应该长期获得高 bonus；一个历史罕见的行为如果总是导致 parse error，也不应该被强化；一个 policy arm 如果只改变梯度权重而不改变采样行为，也不能被夸大为独立策略族。

DIVE-PO 因此把探索建模为如下 synthesis：

> 在 AgenticRL 中，将探索分解为 episodic、lifelong 和 policy-space 三个正交维度是可行且有益的；但这一分解只有在不污染 verifier reward 的注入机制下才成立。DIVE-PO 用独立归一化的 intrinsic advantage stream 作为 enabling mechanism，使多维探索信号能够与 task advantage 共存。

这一定位避免把“advantage-space vs score-space”单独作为核心贡献。advantage-space 注入本身更像正确的工程机制；真正的主线是多维正交分解。二者不可分割：没有分解，双流注入缺少承载对象；没有双流注入，分解后的 intrinsic signal 容易退化为 reward hacking 或 verifier reward 污染。

需要澄清一个命名问题：本文中的 **dual-stream** 指 advantage 空间里的两个训练流，即 task advantage stream 与 intrinsic advantage stream；**three-dimensional** 指 intrinsic stream 内部的探索分解，即 episodic、lifelong 和 policy-space 三个维度。DIVE-PO 的 D 在本文中解释为 **Decomposed**，而不是只表示 dual-stream。

本文贡献如下：

- 提出面向 AgenticRL 的三维正交探索框架：把探索分解为 episodic、lifelong 和 policy-space 三个维度，并说明为何该分解比单一 novelty bonus 更适合多轮工具调用智能体。
- 识别将多维探索落到 AgenticRL group-based policy optimization 上的关键障碍，并提出 independently-normalized intrinsic advantage stream，使探索信号与 verifier reward 在训练目标中共存而不污染 score space。
- 给出终端智能体的一个具体实例：SimHash-KNN episodic novelty、hierarchical decayed count lifelong novelty、outcome-aware gate、共享 backbone 下的 beta-arm/UCB policy allocator。
- 给出数学 RLVR 的可替换实例作为泛化性验证，说明框架的任务无关部分是三维分解与优势注入，而不是 SimHash key 或终端状态 fingerprint 本身。
- 给出理论动机、实验方案和消融设计，重点验证乘法融合、三维分解、score-space 对比、arm 分布和 reward hacking 风险。

## 2 预备知识与问题设定

### 2.1 AgenticRL with Verifiable Outcome Rewards

本文将 AgenticRL 视作语言智能体上的 MDP。给定任务 prompt $q$，第 $t$ 步状态可写为

$$
s_t=(q, h_t, o_t),
$$

其中 $h_t$ 是历史消息、工具调用和模型回复，$o_t$ 是环境观察，例如 shell stdout/stderr、退出码、文件系统变化、测试结果或网页反馈。策略输出 action：

$$
a_t\sim\pi_\theta(\cdot\mid s_t),
$$

其中 $a_t$ 可以是自然语言回复、工具调用、shell 命令、代码 patch 或其他环境可执行操作。环境 transition

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t)
$$

由外部工具和任务环境决定。奖励通常稀疏且延迟，主要由轨迹末端 verifier 或 evaluator 给出：

$$
R^{task}=R(q,\tau).
$$

这一定义比“单轮 answer verification”更贴近本文主场景：工具调用就是 action，环境反馈决定后续状态，探索失败会消耗有限 turn budget 并改变后续可达状态。数学 RLVR 可以看作退化情形：环境 transition 主要由模型继续生成文本组成，verifier 多数只检查最终答案。

### 2.2 Group-based Policy Optimization

给定 prompt $q$，行为策略生成 $G$ 条候选轨迹：

$$
\{\tau_i\}_{i=1}^{G}\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q).
$$

任务 verifier 返回轨迹级任务奖励：

$$
R_i^{task}=R(q,\tau_i).
$$

在 GRPO/DAPO 类方法中，同一 prompt group 内的奖励被转换为相对优势：

$$
A_i^{task}=\operatorname{Norm}_G(R_i^{task}),
$$

其中本文使用

$$
\operatorname{Norm}_G(x_i)=
\frac{x_i-\frac{1}{G}\sum_{j=1}^{G}x_j}
{\sqrt{\frac{1}{G-1}\sum_{j=1}^{G}(x_j-\bar x)^2}+\epsilon}.
$$

本文默认启用标准差归一化，$\epsilon=10^{-6}$。若组内样本数不足或方差为零，优势退化为零或中心化值。

### 2.3 DAPO Objective

DIVE-PO 建立在 DAPO 上。设 $\rho_{i,t}$ 为 token 级 importance ratio：

$$
\rho_{i,t}=
\exp\left(
\log\pi_{\theta}(a_{i,t}\mid h_{i,t})
-
\log\pi_{\theta_{\mathrm{old}}}(a_{i,t}\mid h_{i,t})
\right).
$$

DAPO 使用非对称 clipping：

$$
\rho_{i,t}^{clip}=
\operatorname{clip}(\rho_{i,t},1-\epsilon_{low},1+\epsilon_{high}).
$$

本文当前终端实现默认设置为

$$
\epsilon_{low}=0.2,\qquad \epsilon_{high}=0.28.
$$

令 $\hat A_i$ 表示 DIVE-PO 后处理后的轨迹级训练优势。该优势被广播到轨迹中所有参与 loss 的生成 token。策略目标可写为：

$$
\mathcal{J}(\theta)=
\mathbb{E}_{q,\{\tau_i\}}
\left[
\sum_{i=1}^{G}\sum_{t\in\tau_i}
\min\left(
\rho_{i,t}\hat A_i,\,
\rho_{i,t}^{clip}\hat A_i
\right)
\right].
$$

本文训练使用 token-level loss，因此长轨迹会贡献更多 token 级项。这是实现事实，也是 limitations 中讨论的 credit assignment 风险。

### 2.4 为什么朴素移植 Agent57 会失败

NGU/Agent57 的核心思想是将 episodic novelty、lifelong novelty 和不同探索强度的策略族结合起来。但在 AgenticRL 中，朴素移植至少遇到三个障碍。

第一，reward-space 注入会污染 verifier。Atari 中 intrinsic reward 可以自然进入环境 reward 的 shaped return；而 AgenticRL 中的 verifier/evaluator score 是任务完成语义本身。若设置

$$
\tilde R_i = R_i^{task}+\alpha I_i,
$$

则 $\operatorname{Norm}_G(\tilde R_i)$ 同时反映任务质量和探索启发式，难以判断性能提升来自真实完成能力还是 novelty hacking。

第二，语言智能体状态无法直接使用低维 visit count。终端轨迹包含工具名、命令、输出摘要、错误码、任务上下文和 turn phase；网页或软件工程任务还包含 DOM、文件变化和测试反馈；数学轨迹包含推理角色、公式模式和答案格式。状态抽取器必须是任务自适应的。

第三，Agent57 的 policy dimension 来自多个独立训练的策略或 UVFA 条件化策略族；在大模型 AgenticRL 中训练多个独立 policy 通常不可承受。本文当前实现采用共享 Qwen3-8B backbone 下的 lightweight arm approximation：arm 主要通过 post-normalized intrinsic advantage 的 beta 权重、UCB 分配和极小温度阶梯产生差异。因此本文不声称复现 Agent57 的独立策略族理论收益，而是将 policy dimension 定位为共享 backbone 约束下的探索预算分配器。

## 3 方法

### 3.1 Abstract Framework

DIVE-PO 将每条轨迹 $\tau_i$ 的探索信号拆成三个维度。

**Episodic dimension.** 局内新颖性是一个轨迹映射：

$$
N_{\mathrm{epi}}:\tau_i\rightarrow[0,1].
$$

它度量同一 episode 内的状态覆盖。直觉上，若当前轨迹在本次交互中访问了少见 action state，则 $N_{\mathrm{epi}}$ 较高。

**Lifelong dimension.** 局间新颖性是一个轨迹映射：

$$
N_{\mathrm{life}}:\tau_i\rightarrow[1,M].
$$

它度量跨训练历史的访问稀缺性。$N_{\mathrm{life}}$ 是 modifier 而非独立加项；当某类行为长期罕见时，它放大 episodic novelty，当该行为已被频繁访问时，它回到接近 1。

**Policy-space dimension.** 策略维度由 arm allocator 给出：

$$
a_i\sim \mathcal{C}(\cdot\mid q,\mathcal{H}),\qquad
w_i=w(a_i)\in[0,1],
$$

其中 $\mathcal{H}$ 是历史 arm 统计。当前终端实现使用 beta arm 与 UCB；更强实现可使用 arm-conditioned prompt、decoding family 或 conditioning token。

DIVE-PO 的轨迹级 intrinsic signal 为

$$
I_i=N_{\mathrm{epi}}(\tau_i)\cdot N_{\mathrm{life}}(\tau_i).
$$

policy dimension 不直接乘到 $I_i$ 中，而是在 advantage 注入阶段作为强度权重：

$$
A_i^{int}=\operatorname{Norm}_G(I_i),
$$

$$
B_i^{int}
=
\operatorname{clip}_{[-c,c]}
\left(
\lambda\,w_i\,q_i\,A_i^{int}
\right),
$$

最终训练优势为

$$
\hat A_i=A_i^{task}+B_i^{int}+P_i^{trunc}.
$$

其中 $q_i$ 是 outcome-aware quality gate，$P_i^{trunc}$ 是可选 truncation penalty。当前 v0707 终端实现使用 $\lambda=0.08$、$c=0.35$。

#### 3.1.1 Instantiation Interface

框架本身只要求每个任务提供状态抽取、相似度或计数后端、arm allocator 和 outcome gate。下表给出终端实例与数学实例的对应关系。

| 组件 | 抽象需求 | AgenticRL / 终端智能体实例 | 数学 RLVR 泛化实例 |
|---|---|---|---|
| episodic state | 从轨迹提取局内状态序列 | tool、command signature、observation fingerprint、exit、turn bucket | step role、equation pattern、operation family、answer format、position bucket |
| episodic estimator | 估计同 episode 内新颖性 | 64-bit SimHash + KNN cosine distance | reasoning-state SimHash + KNN 或 template distance |
| lifelong key | 构造跨 episode 计数 key | task/skill/global 三层 terminal action key | problem/skill/global 三层 reasoning key |
| lifelong backend | 维护历史访问频率 | sqlite/local decayed count | decayed count 或 dataset-partitioned count |
| policy allocator | 分配探索强度或采样族 | beta arm + UCB + 轻量温度阶梯 | beta arm + decoding family；可选 prompt conditioning |
| quality gate | 抑制低质量新颖性 | raw score/status floor | answer correctness 或 partial verifier |
| injection | 与 verifier reward 共存 | independent intrinsic advantage stream | same |

因此，SimHash-KNN 和 hierarchical count 是 terminal instantiation，而不是 DIVE-PO 的全部方法。DIVE-PO 的任务无关部分是三维分解、乘法融合和双流优势注入。

#### 3.1.2 为什么使用乘法融合

乘法融合表达的是空间与时间两个探索维度的 conjunction：只有当当前 episode 中的行为新颖，且跨历史仍有探索价值时，intrinsic signal 才显著。若使用加法

$$
I_i^{add}=N_{\mathrm{epi}}(\tau_i)+N_{\mathrm{life}}(\tau_i),
$$

则任一维度很高都可能产生高 bonus。例如某个命令模式在当前 episode 中首次出现，但训练历史中已经频繁访问；加法仍会给较高信号，乘法则因 $N_{\mathrm{life}}$ 接近 1 而限制其长期增益。相反，一个历史罕见的行为如果在当前 episode 内只是重复查看同一文件，乘法也会被低 episodic novelty 压制。

附录 B 给出 tabular count approximation 下的非正式分析。该分析不是对深度语言模型的严格性能保证，而是解释为什么 DIVE-PO 选择

$$
I_i=N_{\mathrm{epi}}(\tau_i)\cdot N_{\mathrm{life}}(\tau_i)
$$

而不是简单加法或单一 count bonus。

### 3.2 Dual-stream Advantage Injection

DIVE-PO 保持 verifier reward 不变：

$$
R_i^{score}=R_i^{task}.
$$

task stream 由 verifier reward 得到：

$$
A_i^{task}=\operatorname{Norm}_G(R_i^{task}).
$$

intrinsic stream 由三维探索框架得到：

$$
I_i=N_{\mathrm{epi}}(\tau_i)\cdot N_{\mathrm{life}}(\tau_i),
\qquad
A_i^{int}=\operatorname{Norm}_G(I_i).
$$

再经 policy weight 与 quality gate 得到：

$$
B_i^{int}
=
\operatorname{clip}_{[-c,c]}
\left(
\lambda w_i q_i A_i^{int}
\right).
$$

#### Proposition 1：未门控 intrinsic stream 的均值保持

若 $B_i^{int}=\lambda A_i^{int}$，且 $\operatorname{Norm}_G$ 含组内去均值，则

$$
\frac{1}{G}\sum_{i=1}^{G}B_i^{int}=0.
$$

因此，未门控的 intrinsic stream 不改变同一 prompt group 的平均 advantage，只改变组内相对排序。

**证明。** 由 $\operatorname{Norm}_G$ 的定义，$\sum_i A_i^{int}=0$。乘以常数 $\lambda$ 后仍为零。证毕。

#### 默认门控实现的边界

当前 v0707 实现使用 sample-dependent 的 $w_iq_i$，因此严格零均值性质对最终 $B_i^{int}$ 不再成立：

$$
\sum_i w_i q_i A_i^{int}\neq 0.
$$

这不是数学上的无关细节。本文应在实验中报告 `center-after-gate` 消融：

$$
\tilde B_i^{int}=
\operatorname{Center}_G(w_iq_iA_i^{int}),
$$

再与默认实现比较。无论是否二次中心化，DIVE-PO 保持两个关键性质：task score 不被修改；intrinsic stream 的最终扰动被 $\lambda$、$w_i$、$q_i$ 和 $c$ 限制，便于记录和审计。

### 3.3 Terminal Instantiation

本节描述当前代码实现对应的终端智能体实例。它是 DIVE-PO 的一个实例，而不是唯一实现。

#### 3.3.1 Episodic Novelty: SimHash-KNN

终端轨迹由多个 action state 组成。第 $t$ 步 action state 记为：

$$
s_t=(tool, signature, observation, exit, turn).
$$

当前 v0707 默认包含粗粒度 turn bucket：

$$
turn\in\{\mathrm{first},\mathrm{early},\mathrm{mid},\mathrm{late}\},
$$

避免精确位置本身成为可刷新的 novelty 来源。将状态映射到 $d=256$ 维向量：

$$
z_t=\phi(s_t)\in\mathbb{R}^{256}.
$$

SimHash 使用随机超平面矩阵 $P\in\mathbb{R}^{64\times256}$：

$$
h_t=\mathbf{1}[(Pz_t)_j\ge0]_{j=1}^{64}.
$$

候选集合来自当前 bucket 及 Hamming radius 1 的 probe buckets：

$$
C_t=\{z_j\mid h_j\in\mathcal{N}_1(h_t)\}.
$$

若 $C_t$ 为空，说明该状态落入未访问区域，取最大局内 novelty：

$$
r_t^{epi}=1.
$$

否则，用最近 $k=5$ 个候选的平均 cosine distance：

$$
d(z,z')=\frac{1-\cos(z,z')}{2},
\qquad
\bar d_k=\frac{1}{k}\sum_{j\in\operatorname{KNN}_k(t)}d(z_t,z_j),
$$

$$
r_t^{epi}=\max\left(\eta,\frac{\bar d_k}{\bar d_k+1}\right),
\qquad \eta=0.02.
$$

轨迹级局内新颖性为：

$$
N_{\mathrm{epi}}(\tau_i)=r_i^{epi}
=
\frac{1}{T_i}\sum_{t=1}^{T_i}r_t^{epi}.
$$

实现采用 compute-then-add：先计算当前 action 的 novelty，再写入 episodic memory，避免当前 action 污染自己的 novelty。

#### 3.3.2 Lifelong Novelty: Hierarchical Decayed Count

局间新颖性衡量跨轨迹的历史访问频率。DIVE-PO 为每个状态构造三层 key：

$$
l\in\{\mathrm{task},\mathrm{skill},\mathrm{global}\}.
$$

task 层最具体，包含任务、动作族、命令签名、观察 fingerprint 与退出状态；skill 层抽象到工具/动作模式；global 层只保留跨任务的粗粒度动作族信息。

对 key $k$，读取写入当前轨迹前的 decayed count：

$$
\tilde c_k=c_k\cdot\delta^{\Delta_k},
\qquad \delta=0.995,
$$

其中 $\Delta_k$ 是距离上次访问的轨迹间隔。单 key novelty 为：

$$
u(k)=\frac{1}{\sqrt{\tilde c_k+1}}.
$$

每层 raw novelty 为：

$$
r_l=\frac{1}{|K_l|}\sum_{k\in K_l}u(k).
$$

三层融合为：

$$
r_i^{life}=
\operatorname{clip}_{[0,2]}
\left(
0.50r_{task}+0.35r_{skill}+0.15r_{global}
\right).
$$

随后用历史 running mean/std 标准化，并经过 softplus 得到 lifelong modifier：

$$
z_i=
\operatorname{clip}
\left(
\frac{r_i^{life}-\mu_{before}}{\sigma_{before}},
-5,5
\right),
$$

$$
N_{\mathrm{life}}(\tau_i)=m_i^{life}
=
\operatorname{clip}_{[1,5]}
\left(1+\operatorname{softplus}(z_i)\right).
$$

最终终端实例的 intrinsic signal 为：

$$
I_i=r_i^{epi}\cdot m_i^{life}.
$$

Decayed count 的作用是避免早期访问永久压制后续探索。长期未出现的行为会随时间恢复部分新颖性，更符合非平稳策略训练过程。

#### 3.3.3 Outcome-aware Gate 与 Truncation Penalty

DIVE-PO 不鼓励所有新颖轨迹，而是用 outcome-aware gate 抑制低质量探索。令 $o_i\in[0,1]$ 为 outcome score，终端实例优先使用任务 raw score；若缺失，则回退到 accuracy、success score、unit-test pass rate 等任务相关指标。

不同轨迹状态的 floor 为：

$$
f_i=
\begin{cases}
0.50, & \text{completed},\\
0.15, & \text{truncated},\\
0, & \text{failed or aborted}.
\end{cases}
$$

质量门控为：

$$
q_i=f_i+(1-f_i)o_i.
$$

当前实现保留 outcome-aware truncation penalty：

$$
P_i^{trunc}
=
-0.01\cdot\mathbf{1}[\mathrm{truncated}_i]\cdot(1-o_i).
$$

这避免把高 outcome 的 truncated trajectory 一律视为坏样本，同时惩罚低 outcome 的超长或未完成轨迹。

#### 3.3.4 Policy-space Arm: 共享 Backbone 下的轻量近似

当前 v0707 终端实现使用 8 个 beta arm：

$$
\beta\in
\{0,0.004,0.006,0.008,0.010,0.012,0.016,0.020\}.
$$

arm weight 为：

$$
w_i=\frac{\beta_{a_i}}{\max_a\beta_a}.
$$

因此 arm 0 是 task-only baseline，最大 arm 的 intrinsic advantage 有效系数为 $\lambda=0.08$。这不是把原始 Agent57 的 $\beta$ 直接加到 reward，而是在 post-normalized advantage 空间中使用 normalized beta 作为相对探索强度。

UCB 只选择 rollout 使用哪个 arm，不直接产生 reward。每个非 evaluation group 保留一个 arm 0 baseline，其余位置由 UCB 或小概率随机探索选择。对 arm $a$，默认 score 为：

$$
UCB_a=
\bar R_a^{base}
-0.5\cdot parse\_rate_a
-0.5\cdot trunc\_rate_a
+0.5\sqrt{\frac{\log(N+1)}{n_a}}.
$$

其中 $\bar R_a^{base}$ 是窗口内 normalized base reward，窗口大小为 256；$n_a$ 是 arm 访问次数；若 $n_a<4$，score 设为 $+\infty$ 以保证冷启动覆盖；随机探索概率为 0.02。

默认设置还为 arm 配置轻量温度阶梯：

$$
T_a\in
\{1.00,1.00,1.005,1.010,1.015,1.020,1.025,1.030\}.
$$

该阶梯在 24 个 rollout warmup 后生效，top-p 均为 1。其目的是让不同 arm 不只在训练梯度中因 beta 不同而区分，也在采样侧产生轻量差异。

本文必须诚实限定这一 policy dimension：当前 arm 共享同一个 LLM backbone，且温度差异很小，因此它不是 Agent57 中多个独立策略的等价替代。它更准确地说是共享 backbone 下的探索预算分配器。若实验显示 arm 间 intrinsic advantage、task reward 或采样行为分布几乎重合，应在最终版本中降低 policy dimension 的贡献定位，甚至删除温度阶梯或将其放到 future work。更强的 policy-space 实例可引入 arm-conditioned system prompt、decoding parameter family 或 arm-conditioning token。

#### 3.3.5 Off-policy 边界

温度阶梯带来 correctness 问题：若 rollout 实际来自 tempered policy $\pi_{\theta_{\mathrm{old}}}^{T_a}$，但训练 ratio 使用未按 arm 温度修正的 $\pi_{\theta_{\mathrm{old}}}$，则 PPO/DAPO 的 on-policy 假设存在偏差。默认温度范围很小，但论文不能默认其无影响。投稿前必须报告：

$$
\left|\log p_{\mathrm{train\ old}}-\log p_{\mathrm{rollout}}\right|,
\qquad
KL(\pi_{\theta_{\mathrm{old}}}^{T_a}\|\pi_{\theta_{\mathrm{old}}}),
$$

以及各 arm 的 ratio 分布。若偏差不可忽略，应采用 temperature-aware old log-prob、启用 rollout log-prob ratio，或删除温度阶梯以保持方法更干净。

### 3.4 Algorithm

**Algorithm 1: DIVE-PO**

```text
Input: prompts q, group size G, policy pi_theta, arm allocator C
for each training iteration do
    for each prompt q do
        assign G arms with one task-only baseline arm and UCB-selected arms
        sample trajectories tau_i with arm-specific beta and optional temperature
        compute verifier reward R_i^task
        extract task-specific states from each trajectory
        compute episodic novelty N_epi(tau_i)
        compute lifelong modifier N_life(tau_i)
        set intrinsic signal I_i = N_epi(tau_i) * N_life(tau_i)
    end for

    for each prompt group do
        compute A_i^task = Norm_G(R_i^task)
        compute A_i^int  = Norm_G(I_i)
        compute quality gate q_i from outcome and trajectory status
        compute arm weight w_i = beta_i / max(beta)
        compute B_i^int = clip(lambda * w_i * q_i * A_i^int, -c, c)
        compute truncation penalty P_i^trunc
        set training advantage Ahat_i = A_i^task + B_i^int + P_i^trunc
    end for

    update policy with DAPO clipped objective using token-level loss
    update arm statistics with base reward, parse rate, and truncation rate
end for
```

## 4 实验设计

本节给出投稿前应完成的实验。所有表格中的数值均为占位符，不代表实验结论。

### 4.1 研究问题

实验应回答五个问题。

1. 三维分解是否比单一 novelty bonus 更有效？
2. 乘法融合是否优于加法融合或单独使用 episodic/lifelong？
3. 独立归一化的 intrinsic advantage stream 是否比 score-space bonus 更稳定？
4. outcome-aware gate 是否减少低质量 novelty 和 reward hacking？
5. 共享 backbone 下的 policy arm 是否真的带来可观察的行为或训练信号差异？

### 4.2 主实验：AgenticRL 终端与工具智能体

主实验应覆盖内部 SETA 与至少一个公开 AgenticRL benchmark。若只报告内部 SETA，审稿人难以校准结果；若只报告数学 RLVR，则无法支撑本文关于多轮工具调用和长程稀疏奖励的核心 claim。

| Benchmark | 目的 | 指标 |
|---|---|---|
| SETA terminal tasks | 验证目标训练场景中的多轮工具调用能力 | raw score、success rate、truncation rate、parse error、turn efficiency |
| Terminal-Bench 或同类公开终端 benchmark | 外部可复现 AgenticRL 验证 | pass rate、normalized score、turns、wall-clock |
| SWE 类软件工程任务 | 验证是否能迁移到更复杂代码修改与测试环境 | resolved rate、test pass rate、edit/test loop count |
| Web/task automation，可选 | 验证网页或 API 工具调用场景 | task success、step count、invalid action rate |

主表应同时报告等训练步数和等 wall-clock 两种口径，并报告 intrinsic 计算 overhead。

| 方法 | 三维分解 | dual-stream | score-space bonus | policy arm | SETA score | SETA pass | Agentic public pass | overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base / SFT | 否 | 否 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| GRPO | 否 | 否 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO | 否 | 否 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + score-space intrinsic | 部分 | 否 | 是 | 可选 | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + dual-stream single novelty | 否 | 是 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| DIVE-PO no policy arm | episodic/lifelong | 是 | 否 | 否 | [TODO] | [TODO] | [TODO] | [TODO] |
| DIVE-PO | 是 | 是 | 否 | 是 | [TODO] | [TODO] | [TODO] | [TODO] |

### 4.3 泛化性验证：数学 RLVR

数学实验用于回答一个次要但重要的问题：DIVE-PO 的三维分解是否能在非工具调用任务中通过替换状态抽取器继续工作。为了与 CDE 类 RLVR 工作可比，数学实验可包含 MATH、AIME24 和 AIME25；AMC23 可作为中等难度补充。数学任务没有 terminal command state，也缺少真实环境 transition，因此它不能替代 AgenticRL 主实验，只能作为泛化性验证。

| Benchmark | 推荐指标 | 说明 |
|---|---|---|
| MATH | Avg@1，Avg@8/16 | 标准数学推理能力 |
| AMC23 | Avg@16，Pass@16 | 中等难度补充 |
| AIME24 | Avg@16，Pass@16 | 高难竞赛推理，多样采样收益 |
| AIME25 | Avg@16，Pass@16 | 更新、更难泛化集 |

数学泛化表：

| 方法 | MATH Avg@1 | AMC23 Avg@16 | AIME24 Avg@16 | AIME24 Pass@16 | AIME25 Avg@16 | AIME25 Pass@16 |
|---|---:|---:|---:|---:|---:|---:|
| Base / SFT | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| GRPO | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + entropy bonus | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DAPO + CDE-style PPL bonus | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| DIVE-PO-math | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

CDE-style baseline 必须真实复现 actor PPL bonus 或使用作者公开实现，不能只作为占位行。若数学结果强于 AgenticRL 结果，正文仍应避免把论文主 claim 改写成数学 RLVR claim。

### 4.4 探索指标

只报告 Avg@1 或 pass rate 不足以支撑探索 claim。应加入：

| 指标 | 目的 |
|---|---|
| pass@k / success@k 曲线，$k=1,\ldots,16$ | 验证多样采样是否带来更多可解轨迹 |
| policy entropy 曲线 | 观察熵坍塌是否缓解 |
| distinct command/action/tool rate | 衡量 AgenticRL 动作多样性 |
| state/action coverage | 衡量多轮 MDP 中覆盖的状态与操作模式 |
| distinct reasoning path rate | 衡量数学泛化实验中的推理多样性 |
| exact repeat rate | 衡量重复动作或重复模板是否减少 |
| episodic novelty / empty bucket rate | 验证局内探索 |
| lifelong raw / modifier | 验证跨轨迹探索 |
| arm distribution | 验证 UCB 是否自适应分配探索预算 |
| per-arm intrinsic/task advantage | 验证 policy arm 是否真的分化 |
| parse/truncation rate | 排除探索导致格式或长度失败 |

### 4.5 多 Seed 与显著性

探索方法方差通常较大。AgenticRL 主实验至少使用 3 个 seeds，报告 mean±std；关键对比使用 paired bootstrap 或近似随机化检验。若算力不足，至少对 DAPO、DIVE-PO、乘法/加法融合、score-space 对比和 no-policy-arm 做多 seed。数学泛化实验可以较小规模补充，但不应替代 AgenticRL 多 seed。

## 5 消融实验

消融应围绕新核心 claim 排优先级：先验证三维分解与乘法融合，再验证 advantage-space enabling mechanism，最后验证具体实现细节。

### 5.1 核心消融

| 消融 | 验证假设 | 设置 | 预期结论 | 支撑 claim |
|---|---|---|---|---|
| episodic-only / lifelong-only / full | episodic 与 lifelong 互补 | 分别关闭 lifelong 或 episodic，与 full 对比 | full 在 pass@k、repeat rate、coverage 上更优 | 三维分解的空间×时间基础 |
| 加法 vs 乘法融合 | 乘法是必要设计而非任意组合 | $I=r^{epi}m^{life}$ 对比 $I=r^{epi}+m^{life}$ | 乘法减少单维高值造成的 novelty hacking | 理论动机与设计选择 |
| no policy arm / fixed arm / UCB | policy dimension 是否有效 | 固定 $w$、round-robin、UCB 对比 | 若收益小，应降格 policy claim | policy-space 维度 |
| score-space vs dual-stream | advantage-space 是分解落地的 enabling condition | 将 $I_i$ 加到 score 后再归一化，对比默认双流 | dual-stream 更稳，parse/truncation 更低 | 不污染 verifier |
| 去掉 outcome gate | 质量门控抑制低质量新颖性 | 令 $q_i=1$ 或只用 status scale | 高 novelty 低 outcome 样本增多 | controllable exploration |

### 5.2 次级消融

| 消融 | 目的 | 处理建议 |
|---|---|---|
| normalized beta vs no beta | 验证 arm 是否对应不同训练探索强度 | 若分布几乎重合，应降低 arm 定位 |
| 温度阶梯 | 验证采样侧差异是否值得保留 | 若 off-policy 偏差或收益不明显，应删除 |
| arm-conditioned prompt/token | 强化 policy dimension 的可选版本 | 若短期无法实现，放入 future work |
| $\lambda$ sweep | 验证 intrinsic 强度 | 测试 $0.04,0.08,0.12,0.16$ |
| beta ladder 密度 | 验证低端密集 ladder 的必要性 | 对比线性 ladder 与默认低端密集 ladder |
| center-after-gate | 验证 Proposition 1 的严格变体 | 对 $w_iq_iA_i^{int}$ 二次中心化 |
| clip $c=0.35$ | 验证 clip 是否实际激活 | 报告 clip 激活率；若接近 0，可解释为安全网 |

## 6 分析实验

### 6.1 Pass@k 与多样性曲线

目的：证明 DIVE-PO 真正提升探索，而不是只改变平均分。

设置：每个 checkpoint 在 AgenticRL benchmark 上评估 success@k/pass@k、策略熵、distinct command/tool rate、state/action coverage、invalid action rate 和 exact repeat rate；在数学泛化实验中额外评估 distinct reasoning path rate。

预期：DIVE-PO 的 pass@k 曲线在较大 $k$ 上更明显，重复率下降，熵下降速度更慢。

### 6.2 Policy Arm 分化分析

目的：正面处理 policy 维度偏薄的风险。

设置：按 arm 记录 $I_i$、$A_i^{int}$、$B_i^{int}$、task advantage、raw score、temperature、parse/truncation rate、采样长度、tool/action diversity 和 state coverage。比较 arm 间分布是否分化。

预期：若高 beta arm 获得更高 intrinsic advantage 且不显著降低 task score，可支持轻量 policy dimension；若 arm 分布高度重合，则正文应将 policy dimension 降格为实现细节或 future work。

### 6.3 Off-policy 与温度偏差

目的：回答不同 arm 温度是否破坏 DAPO ratio 假设。

设置：按 arm 记录 rollout log-prob、训练侧 old log-prob、二者差值、sequence-level KL、clip fraction 和 ratio 分位数。

预期：默认温度 1.00-1.03 的偏差应较小；若高温 arm 显著偏离，应改为 temperature-aware ratio 或删除温度阶梯。

### 6.4 混 Arm 组内归一化

目的：分析 $\operatorname{Norm}_G(I_i)$ 在同一 group 内混合不同 beta/temperature arm 的影响。

设置：统计每个 arm 的 $I_i$、$A_i^{int}$、$B_i^{int}$、task advantage 分布，并比较 group-level normalization 与 per-arm normalization。

预期：若高温 arm 系统性获得正 intrinsic advantage、低温 arm 系统性为负，需要判断这是有意的跨 arm 比较，还是对 baseline arm 的不公平压制。

### 6.5 Reward Hacking 审计

目的：验证 novelty 不会被表面变化刷高。

设置：抽样高 novelty 低 outcome 轨迹，按模式分类：无意义 echo、参数顺序扰动、重复查看文件、无效工具调用、破坏性或无关命令、过长输出、格式失败等。数学泛化实验中另行统计错误但形式新颖的推理。

预期：outcome gate 应显著降低这些样本的最终 bonus；若仍大量出现，需要改进 state fingerprint 或 gate。

### 6.6 Intrinsic 与 Task Advantage 量级

目的：确认 intrinsic stream 足够可见但不主导任务优化。

设置：记录 $A_i^{task}$、$A_i^{int}$、$B_i^{int}$、$\hat A_i$ 的均值、标准差、分位数，以及 clip 激活频率。

预期：最大有效系数 0.08 使 $B_i^{int}$ 处于 task advantage 的辅助量级；若 clip 0.35 几乎不激活，应将其解释为安全网或在最终实现中简化。

### 6.7 Compute-matched 对比

目的：排除额外计算带来的不公平收益。

设置：报告等训练步数与等 wall-clock 两种结果，并拆分 SimHash-KNN、sqlite lifelong store、UCB controller 的时间开销。

预期：DIVE-PO 的主要收益不应只来自更多 wall-clock 或更多有效样本。

## 7 相关工作

### 7.1 AgenticRL、RLVR、GRPO 与 DAPO

PPO（Schulman et al., 2017）奠定了 clipped policy optimization 的基础。AgenticRL 将语言模型放入多轮交互环境中训练，常以工具执行、网页操作、代码修改或终端任务作为 action space，并用稀疏 evaluator/verifier 反馈优化长程策略。GRPO 在 RLVR 中用组内相对优势替代显式 critic，被 DeepSeekMath 等工作用于数学推理训练。DAPO 在 GRPO 类估计器上加入更适合长输出任务的 clipping、token-level loss、overlong shaping 和可选 dynamic sampling。DIVE-PO 建立在 DAPO objective 上，但关注点不同：它研究如何在 AgenticRL 的 group-based policy optimization 中引入可解释的多维探索信号。

### 7.2 Count-based 与 Intrinsic Motivation

经典 count-based exploration 使用访问次数或伪计数奖励罕见状态（Bellemare et al., 2016；Tang et al., 2017）。ICM（Pathak et al., 2017）和 RND（Burda et al., 2019）使用预测误差作为新颖性信号。DIVE-PO 与这些方法共享“访问不足区域应获得探索压力”的思想，但当前实现不训练额外 dynamics 或 predictor；它使用 action-state fingerprint、SimHash-KNN 和 hierarchical decayed count 作为轻量估计。理论动机也按 count-based pseudo-count 写出，避免把 RND 的假设误套到本文实现。

### 7.3 NGU、Agent57 与 UCB

NGU 和 Agent57（Badia et al., 2020a,b）结合 episodic novelty、lifelong novelty 和不同探索强度的策略族，在 Atari 等环境中实现深度探索。DIVE-PO 借鉴其正交分解思想，但解决了直接移植到 AgenticRL 的三个障碍：reward-space 注入会污染 verifier；语言智能体状态需要任务自适应 fingerprint；训练多个独立 LLM policy 不现实。

因此，DIVE-PO 不是简单的“Agent57 移植”。它把 Agent57-style decomposition 改写成 AgenticRL 可用的三维 intrinsic stream，并用 independently-normalized advantage injection 保持 verifier reward 的语义。与此同时，当前 policy-space arm 是共享 backbone 下的轻量近似，理论收益弱于 Agent57 的独立策略族；本文将在 limitation 和 arm 分析中如实处理这一点。

### 7.4 语言模型中的探索与 CDE

近期 RLVR 和 agent training 探索工作研究了 entropy bonus、actor perplexity bonus、critic uncertainty 和多头 critic。CDE 类方法把 actor PPL 或 critic 方差作为好奇心信号。DIVE-PO 的定位不同：默认实现不使用 actor PPL，也不使用多头 critic，而是把 AgenticRL 轨迹中的状态-动作新颖性变成独立 intrinsic advantage。与 CDE 的实验对比应聚焦“PPL/uncertainty curiosity”与“state/action novelty advantage stream”的差异。

## 8 局限性

DIVE-PO 仍有若干限制。

第一，当前 policy dimension 较薄。所有 arm 共享同一个 Qwen3-8B backbone，arm 间主要差异是 beta 权重和极小温度阶梯，因此不能宣称具备 Agent57 多独立策略族的完整能力。若 arm 分析不能证明行为差异，应将 policy-space 定位为探索预算分配器，并把更强 arm conditioning 留作 future work。

第二，intrinsic signal 是轨迹级的，而 token-level loss 会把同一 bonus 广播到所有生成 token，credit assignment 较粗。后续可探索 step-level intrinsic advantage 或 token/turn attribution。

第三，轻量温度阶梯可能引入 off-policy 偏差，必须用 log-prob/KL 统计验证。若偏差不可忽略，应删除温度阶梯或修正 old log-prob。

第四，SimHash novelty 可能被表面扰动 hack，outcome gate 只能缓解，不能完全解决。高 novelty 低 outcome 的人工审计是必须的。

第五，state fingerprint 的设计与任务强相关，迁移到网页、软件工程、机器人工具调用或数学 RLVR 需要重新定义状态抽取器。本文将其定位为 instantiation interface，而不是固定算法。

第六，lifelong sqlite store 在分布式训练下存在一致性、锁竞争和恢复问题，需要报告 overhead、lock wait、失败恢复策略和清理策略。

第七，若 SETA 无法公开，主结论必须由公开 AgenticRL benchmark 支撑。数学 benchmark 可以证明框架泛化，但不能单独支撑关于多轮工具调用探索的主结论。

## 9 可复现性声明

最终投稿应报告以下信息：base model 与 checkpoint；训练数据可得性；SETA 是否可公开；公开 AgenticRL benchmark 的版本、prompt、verifier/evaluator、工具环境和容器环境；训练 GPU 型号、GPU 数、总卡时；随机种子；rollout group size；采样温度；DAPO clip；是否使用 rollout log-probs；所有 exploration 超参；sqlite lifelong store 的持久化路径与清理策略；arm controller 的随机种子与窗口；reward post-process 代码版本。若数学泛化实验完成，还应报告 verifier、答案抽取规则和 pass@k 采样预算。

## 10 结论

本文提出 DIVE-PO：Decomposed Intrinsic adVantage Exploration for Policy Optimization in AgenticRL。DIVE-PO 的核心不是把探索 bonus 从 reward 搬到 advantage 这一单点修正，而是把 AgenticRL 中的探索拆成 episodic、lifelong 和 policy-space 三个正交维度，并通过乘法融合与独立归一化 intrinsic advantage stream 将其接入 DAPO。advantage-space 注入是使这套分解不污染 verifier reward 的 enabling mechanism。当前终端实例使用 SimHash-KNN、hierarchical decayed count、outcome-aware gate 和共享 backbone 下的 beta-arm/UCB；数学实例只替换状态抽取和计数 key，作为泛化性验证而非主场景。投稿前的关键工作是完成 AgenticRL 主实验、三维分解与乘法融合消融、score-space 对比、arm 分化分析、reward hacking 审计和 off-policy 诊断。

# 附录 A：数学 RLVR 泛化验证中的 DIVE-PO 适配

数学任务没有终端命令、工具副作用和环境 observation，因此不能直接复用 terminal action state。适配原则是把 AgenticRL 中的“工具动作新颖性”替换为“推理状态新颖性”，同时保留三维分解与 dual-stream advantage injection。该附录用于验证框架可迁移性，不改变正文以 AgenticRL 为主场景的定位。

## A.1 Reasoning State

将模型输出切分为 reasoning states：

$$
s_t^{math}=(problem\_type, step\_role, equation\_pattern, answer\_format, position\_bucket).
$$

其中 $step\_role$ 可包括 lemma、case split、calculation、verification、final answer 等；$position\_bucket$ 可取 early/mid/late/final，避免仅因输出长度获得 novelty。

## A.2 数学 Episodic Estimator

将 $s_t^{math}$ 映射为向量 $z_t=\phi(s_t^{math})$，复用 64-bit SimHash 与 Hamming radius 1 multi-probe：

$$
h_t=\mathbf{1}[(Pz_t)_j\ge0]_{j=1}^{64}.
$$

step novelty 与 terminal 设置相同，但建议对错误答案使用更强 gate，因为数学中“新颖但错误”的推理路径很多。

## A.3 数学 Lifelong Key

数学 lifelong key 可分三层：

| 层级 | key 示例 |
|---|---|
| task | dataset/problem id/problem type/step fingerprint/equation pattern/final answer type |
| skill | problem type/operation family/equation pattern/answer format |
| global | operation family/answer format |

operation family 可包括 algebra、geometry、number theory、combinatorics、probability 等。

## A.4 Outcome Gate

若只有最终答案 verifier，可设：

$$
o_i=\mathbf{1}[\mathrm{answer\ correct}].
$$

若有 partial verifier 或 process reward，可使用 $[0,1]$ 连续分数。无效格式、无法抽取答案和超长输出应使用低 floor。最终仍使用：

$$
B_i^{int}=
\operatorname{clip}_{[-c,c]}
\left(
\lambda w_i q_i A_i^{int}
\right).
$$

## A.5 数学实验注意事项

数学实验若无法在投稿前完成，不应作为未兑现 promise 放在正文中。若完成，应作为泛化性验证报告，并与 CDE-style PPL bonus 做严格对比：相同 base model、相同 prompt、相同 verifier、相同采样预算和相同 pass@k 统计。即使数学结果显著，AgenticRL 主实验仍是论文核心证据。

# 附录 B：乘法融合的理论动机（Informal）

本附录给出 tabular approximation 下的动机性分析。它不是对深度语言模型、SimHash 表征或 DAPO 优化的严格保证，而是解释为何 episodic 与 lifelong 采用乘法融合，以及为何 policy arm 在共享 backbone 下不能照搬 Agent57 的独立策略族理论收益。

## B.1 Assumptions

考虑一个简化 tabular MDP。每个轨迹访问状态 $s$，局内计数为 $C_{\mathrm{ep}}(s)$，跨 episode decayed count 为 $C_{\mathrm{life}}(s)$。设 episodic novelty 近似为

$$
r^{epi}(s)\propto \frac{1}{\sqrt{C_{\mathrm{ep}}(s)+1}},
$$

lifelong modifier 近似为

$$
m^{life}(s)\propto \frac{1}{\sqrt{C_{\mathrm{life}}(s)+1}}.
$$

则乘法 intrinsic signal 近似为

$$
I(s)\propto
\frac{1}{\sqrt{(C_{\mathrm{ep}}(s)+1)(C_{\mathrm{life}}(s)+1)}}.
$$

这对应“局内未覆盖”和“跨历史少见”两个条件的几何平均式压力。

## B.2 Informal Proposition

**Informal Proposition.** 在 tabular、uniform behavior、单瓶颈状态且局内/局间计数近似独立的假设下，使用乘法 intrinsic signal 的策略更新会优先提高同时满足低 $C_{\mathrm{ep}}$ 与低 $C_{\mathrm{life}}$ 状态的访问概率。对于目标状态 $s^\star$，探索压力与

$$
\frac{1}{\sqrt{(C_{\mathrm{ep}}(s^\star)+1)(C_{\mathrm{life}}(s^\star)+1)}}
$$

同阶。

该命题只作为设计动机。完整证明需要明确行为策略、归一化、clip、函数逼近和策略更新步长；本文不把它作为严格定理。

## B.3 乘法相对加法的性质

加法 bonus

$$
I_{add}(s)=r^{epi}(s)+m^{life}(s)
$$

在任一维度较高时都给出高值，因此容易奖励“当前 episode 中新但全局已经很常见”的行为，或“全局少见但当前 episode 内重复”的行为。乘法

$$
I_{mul}(s)=r^{epi}(s)m^{life}(s)
$$

则要求两个维度同时支持探索，天然抑制单维度异常值。实验中必须加入加法 vs 乘法消融来验证这一动机。

## B.4 Lifelong Count 而非 RND

Agent57 使用 lifelong RND 预测误差作为长期新颖性估计，而当前 DIVE-PO 终端实现使用 hierarchical decayed count。本文理论动机因此应围绕 count-based pseudo-count 展开，而不是直接引用 RND 分析。RND 可作为相关工作中的替代估计器，但不是当前实现的理论对象。

## B.5 Policy Dimension 的理论缩水

Agent57 的 policy dimension 来自多个独立探索策略或条件化策略族，因此可以产生强行为多样性。当前 DIVE-PO 实现共享同一个 LLM backbone，policy arm 主要改变 intrinsic advantage 权重，并只用极小温度差异影响采样。因此，policy dimension 的理论收益退化为 UCB 对探索预算的自适应分配，而不是独立策略族带来的完整覆盖因子。正文和实验必须如实呈现这一点。

# 附录 C：当前实现配置映射

本附录仅用于代码 release 和复现实验；正文不依赖环境变量名叙事。配置对应当前 v0707 终端实例。

| 模块 | 当前实现 |
|---|---|
| base model | Qwen3-8B |
| policy optimization | DAPO with GRPO advantage estimator |
| dataset | SETA |
| dynamic sampling | off |
| rollout batch size | 8 prompts |
| group size | 8 samples per prompt |
| max turn | 10 |
| rollout config | `configs/rollout_qwen3_think.yaml` |
| DAPO clip | $\epsilon_{low}=0.2,\epsilon_{high}=0.28$ |
| token-level loss | on |
| score-space exploration bonus | off (`EXPLORE_SCORE_BONUS_COMPONENTS=none`) |
| intrinsic framework | three-dimensional decomposition + dual-stream advantage |
| intrinsic signal | `explore_agent57_intrinsic_signal` |
| episodic backend | SimHash-KNN |
| episodic bits / dim / k | 64 / 256 / 5 |
| episodic distance / radius | cosine / Hamming radius 1 |
| episodic turn mode | include coarse bucket |
| episodic novelty floor | 0.02 |
| lifelong backend | sqlite |
| lifelong decay / capacity | 0.995 / 200000 |
| lifelong modifier | standardized softplus, clipped to [1,5] |
| lifelong hierarchy weights | task 0.50, skill 0.35, global 0.15 |
| intrinsic fusion | episodic novelty * lifelong modifier |
| advantage mode | dual-stream post-normalized advantage |
| intrinsic lambda / clip | 0.08 / 0.35 |
| arm weight | normalized beta |
| beta arms | 0, 0.004, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020 |
| outcome gate mode | outcome_status |
| outcome key | raw_score |
| outcome gate floors | completed 0.50, truncated 0.15, failed 0, aborted 0 |
| truncation penalty | $-0.01(1-o_i)$ for truncated trajectories |
| UCB C / window / epsilon / min | 0.5 / 256 / 0.02 / 4 |
| UCB value | normalized base reward |
| UCB dataset-aware | on |
| arm temperatures | 1.00, 1.00, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030 |
| temperature warmup | 24 rollouts |

# 附录 D：投稿前优先级

1. 完成 SETA 主实验、至少一个公开 AgenticRL benchmark 主实验、三维分解消融、乘法 vs 加法消融、score-space vs dual-stream 消融。
2. 补 policy arm 分化分析；若 arm 分布几乎重合，降低 policy dimension claim 或强化 arm conditioning。
3. 补 off-policy 温度偏差与混 arm 组内归一化分析，决定是否保留温度阶梯。
4. 补 reward hacking 审计、intrinsic/task advantage 量级、clip 激活频率和 UCB 敏感性。
5. 落实 CDE-style baseline 与相关工作引用。
6. 视算力决定数学实验是否作为泛化性验证并入正文；未完成时不要把 MATH/AIME 作为核心结论。

# 附录 E：参考文献占位

- Schulman et al. Proximal Policy Optimization Algorithms. 2017.
- Auer et al. Finite-time Analysis of the Multiarmed Bandit Problem. 2002.
- Bellemare et al. Unifying Count-Based Exploration and Intrinsic Motivation. 2016.
- Tang et al. #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning. 2017.
- Pathak et al. Curiosity-driven Exploration by Self-supervised Prediction. 2017.
- Burda et al. Exploration by Random Network Distillation. 2019.
- Badia et al. Never Give Up: Learning Directed Exploration Strategies. 2020.
- Badia et al. Agent57: Outperforming the Atari Human Benchmark. 2020.
- Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. 2024.
- [TODO] DAPO 原始论文或技术报告。
- [TODO] GSPO 原始论文或技术报告。
- [TODO] CDE 参考论文正式 BibTeX。
