# Math RLVR 评测与 DAPO 配方设计

本文是 `feat/math-rlvr-eval` 的设计约束和实验协议。实现先固定协议，再把
数据、抽取、验证、打分、重评分和统计拆成可替换的模块。这样训练日志中的
reward 与离线评测可以追溯到同一份代码和同一份数据 manifest。

## 1. 目标和默认协议

核心评测集是 AIME2025、AIME2024（holdout）、AMC23 和 MATH-500；训练数据
可选 DAPO-Math-17k。所有数据行在进入 rollout 前被规范成
`{id, prompt, label, source, metadata}`，并以 SHA-256 manifest 固化顺序、去重
结果和来源 revision。默认 response cap 为 32768，默认 reward 为 `math`（语义
答案验证），`dapo` 只作为显式的格式敏感消融。训练脚本不提供 checkpoint 默认
值，必须显式传入 `HF_CKPT` 和 `REF_LOAD`。

## 2. 多格式 answer extractor

`extractor.py` 输出带 provenance 的候选列表，而不是只返回一个字符串：候选包含
`value`、`format`（`answer_line`、`boxed`、`natural_language`）、字符区间和
是否完整闭合。优先级固定为：

1. 最后一个完整的 `Answer:` 行（大小写不敏感，允许 Markdown 加粗和空白）；
2. 最后一个完整的 `\\boxed{...}`/`\\fbox{...}`（支持嵌套大括号）；
3. 最后一个带 final-answer 语义的自然语言片段（例如
   `final answer is 42`、`因此答案为 42`）。

同一格式出现多次时取最后一个，因为推理过程可能包含中间答案。不同格式同时
出现时不静默覆盖：按上述优先级选 canonical 候选，同时保留全部候选并设置
`conflict=true`（值无法归一化为同一答案时）。因此训练不被要求输出某个模板，
而统计仍能回答“模型是否只改变了格式”。无候选或未闭合的 marker 会产生
`format_mismatch`，但不会被当作正确答案。

## 3. byte-identical verifier 与 reward tracks

`verifier.py` 是训练 custom RM 和离线评测共同导入的唯一实现；运行记录包含
该模块文件的 SHA-256。语义 `math` verifier 对 canonical candidate 使用
Minerva/MathD 归一化和数值等价判断；`dapo` verifier 复刻 Slime 的
`Answer:` 整数规则，仅作为显式 ablation。两者都返回结构化结果
`{correct, extracted, format, scorable, error}`，不会把解析异常吞成普通错误。

每条样本同时保存：

- `strict_correct`：配置指定的训练 reward track；
- `lenient_correct`：允许 `Answer:`、boxed 和自然语言的语义 track；
- `boxed_correct`：只接受 boxed 的可重算 track；
- candidate provenance、冲突和 verifier 错误。

因此 `format_penalty = lenient_correct and not strict_correct` 是可观测的分解，
而不是用第二套不可比的评测器替换训练 reward。若实验选择 `math` 训练，严格
track 仍会记录用于对照，但训练更新本身不奖励格式。

## 4. Format penalty、truncation 与 zero-variance

评测汇总报告以下分母明确的比例：

- `format_mismatch_rate`：在可评分且完成的样本中，lenient 正确而 strict 错误；
- `truncation_rate`：`finish_reason=length` 或 completion token 达到 cap；
- `zero_variance_group_rate`：每个 prompt 的 rollout group 中 reward 方差为零的
  比例；同时报告 group 数和被跳过的 group 数。

训练端把这些数写入 structured JSONL；zero-variance group 不伪造 advantage，
而是计数并按 DAPO dynamic filter 配置跳过。format penalty 高时先检查 extractor
冲突和 verifier hash，再判断是否需要改变 reward；不能用“提高格式奖励”掩盖数学
能力下降。

## 5. Response cap 的一阶影响

设未截断时单次正确率为 `p`，超过 cap 的概率为 `t`，且截断近似独立于答案正确
性，则 cap 后 `p_cap ≈ p(1-t)`：

```
Avg@k_cap ≈ (1-t) Avg@k_full
Pass@k_cap ≈ 1 - (1 - p(1-t))^k
```

这是诊断近似，不替代逐样本重算。评测保留 completion token 和 finish reason，
重评分把超 cap 样本计为错误而不是丢弃分母，可得到任意 cap 的 counterfactual。
默认 32768 是为了让 truncation 不主导 AIME；若 `truncation_rate > 5%`，报告必须
同时给出更高 cap 或敏感性曲线，不能只公布一个 Pass@k。

## 6. DAPO 训练隔离

默认配方如下：

| 项目 | 默认值 | 原因 |
|---|---|---|
| reward | `math` | 语义验证，避免把格式学习当能力增益 |
| rollout cap | 32768 | 与基线评测一致，降低截断偏差 |
| in-training eval | `aime-2025` + `aime-2024` holdout | 训练集趋势与泛化分开 |
| batch / eval | 从同一 config 生成 | 切换 17k 时不会遗忘联动修改 |
| seed | 显式参数 | 小规模 AIME 需要 paired/multi-seed 统计 |

切换到 DAPO-Math-17k 必须同时设置数据路径、`ROLLOUT_BATCH_SIZE` 和 eval 清单；
脚本会在启动时打印解析后的行数、唯一 id 数、batch 对齐和 manifest hash。只有
`REWARD_TYPE=dapo` 的对照实验才启用严格 `Answer:` 格式奖励，并必须报告
format/capability 分解。

## 7. 可复现流水线与阶段

```
prepare_math_data.py -> launch_sglang_math.sh -> eval_math.py
                  -> rescore_math_eval.py -> math_paired_stats.py
```

阶段一在 base checkpoint 上跑四个核心集并保存 per-sample JSONL；阶段二以 AIME2025
训练、AIME2024 holdout 做单 seed DAPO；阶段三分别消融 reward、cap 和训练规模。
每一步都可以通过 `rjob` 包装脚本提交，运行目录写入 config、日志、manifest、
逐样本记录和 checkpoint 路径。若复现偏差大或 penalty/truncation 异常，优先回退
检查 extractor、verifier hash、cap 和数据 manifest。

## 8. 相对 PR #1 的改进

PR [#1](https://github.com/puyuan1996/LightRL/pull/1) 的脚本提供了四数据集 sweep、Avg@k/Pass@k、boxed 对照和重评分入口。本
分支保留这些兼容入口，但把逻辑拆成 `data/extractor/verifier/scorer/rescorer/
stats` 模块；数据源、模型、cap、reward 和 eval 列表均为参数；每条记录保存候选
来源、冲突、truncation 与 zero-variance 证据；训练和评测共享 verifier 并记录
hash；默认 checkpoint 必填，避免旧服务名静默指向错误模型。
