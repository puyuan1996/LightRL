# Math RLVR 评测与 DAPO 运行规范

本文是 Math RLVR 代码、评测协议、训练配置和已完成实验的单一说明入口。
它不依赖特定分支、Pull Request 或集群站点；路径、镜像、队列和 checkpoint
均由运行环境显式提供。

## 1. 范围与原则

支持四个核心评测集：AIME2025、AIME2024 holdout、AMC23、MATH-500；训练集可选
去重后的 DAPO-Math-17k。训练与评测共享同一个 `Verifier` 实现和 verifier hash，
以便把数学能力、格式学习、截断和 rollout 组退化分开报告。

默认协议：

| 项目 | 默认值 | 说明 |
|---|---|---|
| `REWARD_TYPE` | `math` | 语义答案验证，不要求固定输出模板 |
| `RESPONSE_CAP` / `MAX_TOKENS` | `32768` | 评测默认值；超过 5% 截断时必须做 cap 消融 |
| `EVAL_DATASETS` | `aime-2025,aime-2024` | 训练中监控与 holdout 分开统计 |
| `N` | `16` | 评测每题采样数 |
| checkpoint | 无默认值 | 必须显式传入，避免静默评测错误模型 |

共享数据通过 `MATH_DATA_ROOT` 或 `LIGHTRL_DATA_ROOT` 注入；未设置时只搜索
仓库内 `data/math_rlvr` 和 `benchmarks/math`。标准数据文件如下：

| 数据集 | 文件 | 题目数 |
|---|---|---:|
| AIME2025 | `aime-2025.jsonl` | 30 |
| AIME2024 | `aime-2024.jsonl` | 30 |
| AMC23 | `amc23.jsonl` | 40 |
| MATH-500 | `math-500.jsonl` | 500 |
| DAPO-Math-17k | `dapo-math-17k.jsonl` | 17,255 unique |

数据加载器将不同来源规范为 `{id, prompt, label, source, metadata}`，可选稳定去重、
限制采样和 manifest。训练切换到 DAPO-Math-17k 时，应同时核对数据 manifest、
`ROLLOUT_BATCH_SIZE`、`GLOBAL_BATCH_SIZE` 和 in-training eval 清单。

## 2. 答案抽取与 verifier

`extractor.py` 保留所有候选及 provenance，canonical 候选按以下顺序选择：

1. 最后一个完整的 `Answer:` 行（大小写不敏感，允许 Markdown 加粗和空白）；
2. 最后一个完整的 `\\boxed{...}` 或 `\\fbox{...}`（支持嵌套大括号）；
3. 最后一个具有 final-answer 语义的自然语言片段，例如 `final answer is 42`。

同一格式重复出现时取最后一个，因为推理过程可能包含中间答案。多格式同时出现
时按上述优先级选择 canonical 候选，同时保留全部候选并记录 `conflict`；未闭合
标记或无候选不会被静默当作正确答案。这样 `math` reward 不要求模型学习某种
模板，而 strict/boxed 轨迹仍可用于诊断格式变化。

`verifier.py` 是训练 custom reward 和离线评测共同导入的唯一语义实现。它返回
`correct`、`extracted`、`format`、`scorable` 和 `error`，解析异常不会被吞成普通
错误。`math` track 进行归一化和数学等价判断；`dapo` track 仅用于显式的
`Answer:` 格式敏感消融。

## 3. 指标与质量门槛

每个样本保存配置 reward、lenient/math、strict/dapo、boxed、候选冲突、完成 token、
finish reason、truncation 和 verifier error。汇总至少包含：

- `Avg@k`、`Pass@k`：配置 reward 的主指标，同时保留 lenient/strict/boxed 轨迹；
- `format_mismatch_rate`：可评分样本中 lenient 正确但 strict 错误的比例；
- `truncation_rate`：`finish_reason=length` 或 token 数达到 cap 的比例；
- `zero_variance_group_rate`：每个 prompt 的 rollout reward 方差为零的 group 比例；
- `verifier_error_count`、`format_compliance_rate` 以及完整 per-sample 记录。

对未截断单次正确率 `p`、截断概率 `t` 的一阶诊断近似为：

```text
p_cap ≈ p * (1 - t)
Avg@k_cap ≈ (1 - t) * Avg@k_full
Pass@k_cap ≈ 1 - (1 - p * (1 - t)) ** k
```

这只是告警模型，不替代逐样本重算；`rescore_math_eval.py` 可在不调用模型的情况
下改变 reward type 或 response cap。截断率过高时不能只公布 Pass@k，必须附带更高
cap 或敏感性曲线。zero-variance group 不伪造 advantage，应记录并按训练器策略
跳过或过滤。

## 4. 代码结构与入口

| 路径 | 职责 |
|---|---|
| `tools/evaluation/math_rlvr/data.py` | 路径解析、别名、JSON/JSONL/HF 加载、去重、manifest |
| `tools/evaluation/math_rlvr/extractor.py` | 多格式候选抽取与冲突记录 |
| `tools/evaluation/math_rlvr/verifier.py` | 训练/评测共享的语义与 strict 验证 |
| `tools/evaluation/math_rlvr/scorer.py` | 评分、分组、汇总和离线重算 |
| `tools/evaluation/math_rlvr/stats.py` | 配对差值与 bootstrap CI |
| `tools/evaluation/eval_math.py` | OpenAI-compatible endpoint 评测入口 |
| `tools/evaluation/run_math_base_evals.sh` | 四个核心数据集批量评测 |
| `examples/training/train_qwen3_8b_dapo_math.sh` | Qwen3 DAPO 训练配方 |
| `local/rjob/` | scheduler submitter、payload、checkpoint 转换入口 |

## 5. 本地与 RJob 使用

本地评测：

```bash
export MATH_DATA_ROOT=/path/to/math_rlvr_data
MODEL_PATH=/path/to/checkpoint \
  bash tools/evaluation/launch_sglang_math.sh
MODEL=my-model DATASETS='aime-2025 aime-2024 amc23 math-500' \
  bash tools/evaluation/run_math_base_evals.sh
python tools/evaluation/rescore_math_eval.py \
  /path/to/results/aime-2025_T1.0_n16.detail.json --response-cap 65536
```

训练必须显式提供 `HF_CKPT`、`REF_LOAD`、`TRAIN_DATASET`、`REWARD_TYPE` 和
`RESPONSE_CAP`。RJob 入口位于 `local/rjob/`，复制 `rjob.env.example` 为未跟踪的
`rjob.env`，通过环境变量提供 namespace、charged group、镜像、挂载和持久化目录；
仓库不保存站点地址。`DRY_RUN=1`（CLI submitter）或 `RJOB_DRY_RUN=1`（Python
client submitter）只生成命令/spec，不提交作业。

建议的最小训练/holdout 流程：

```bash
HF_CKPT=/path/to/base REF_LOAD=/path/to/reference \
  RJOB_NAME=math-dapo-seed1 NUM_EPOCHS=10 \
  bash local/rjob/submit_math_rlvr_train.sh

MODEL_PATH=/path/to/converted-hf MODEL=my-model DATASETS=aime-2024 \
  REWARD_TYPE=math N=4 MAX_TOKENS=8192 \
  bash local/rjob/submit_math_rlvr_eval.sh
```

每次运行应保留 config、manifest、per-sample detail、summary、服务/训练日志和
checkpoint 路径；所有结果需记录 verifier hash、数据 hash、seed、reward、cap 和
评测清单。

## 6. 已完成验证与当前实验状态

静态/冒烟验证已覆盖 Python 编译、Shell 语法、RJob dry-run、extractor/verifier/
scorer/stats 重点测试和 DAPO-Math-17k 17,255 条唯一数据校验。基线 `n=16` 结果：

| 数据集 | lenient Avg@16 | Pass@16 | format mismatch | truncation | zero-variance groups |
|---|---:|---:|---:|---:|---:|
| AIME2025 | 69.17% | 80.00% | 68.96% | 11.04% | 14/30 |
| AIME2024 | 77.92% | 90.00% | 77.50% | 6.88% | 15/30 |
| AMC23 | 95.78% | 100.00% | 0.00% | 0.78% | 34/40 |
| MATH-500 | 87.51% | 90.20% | 61.28% | 0.24% | 460/500 |

一次 actor update 的训练后 checkpoint 在 AIME2024 `N=4, cap=8192` holdout 上为
Avg@4/Pass@4 = 0/0，truncation=100%，zero-variance=100%，verifier errors=0；
输出为高重复非答案序列，因此该结果只说明 checkpoint/训练不稳定，不能作为数学
能力下降结论。十 epoch 训练尚未形成可验收 checkpoint，后续应先解决训练稳定性，
再进行 paired AIME2025/AIME2024 评测和 reward/cap/data 消融。

详细的时间戳实验原始记录位于
[`local/records/iteration/math_dapo_aime25_seed1_20260906-200223/experiment.md`](../../local/records/iteration/math_dapo_aime25_seed1_20260906-200223/experiment.md)；
该记录只描述可复现实验事实，不替代本规范中的协议。
