# Math RLVR 实验报告（可复现实验记录）

本文件是阶段一至三的结果表。脚本会把原始逐样本记录写入运行目录；完成
`rjob` 作业后，将 `summary.json` 和 `math_paired_stats.py` 的输出填入这里，
不要手工从日志抄取单一准确率。

PR #1 提供的历史锚点（Qwen3-8B、不同数据/运行环境，不能替代本分支复跑）为：
AIME2025 strict/lenient Avg@16 = 20.62%/67.71%，AMC23 strict/lenient
Avg@16 = 42.19%/93.44%，MATH-500 over all 500 problems strict/lenient
Avg@4 = 36.70%/86.20%。MATH-500 的 349 个可评分题目 lenient 基线为 91.98%；
复现时必须注明采用的是 500 题还是 349 题分母。

## 配置指纹

| 字段 | 值 |
|---|---|
| branch | `feat/math-rlvr-eval` |
| base checkpoint | `/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B` |
| reference checkpoint | `/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B_torch_dist` |
| train manifest SHA-256 | AIME2025 `a6ec56fd...c3c1a84b`; DAPO-Math-17k `ecd252ba...2bbfd1a` |
| verifier / extractor SHA-256 | `9e916a60...3f9af5` / `370637b8...c22dfe` |
| reward / cap / seed | `math / 8192 (retry11) / 1` |

## 阶段一：base 评测

| dataset | n | strict Avg@k | lenient Avg@k | Pass@k | boxed Avg@k | format mismatch | truncation | zero-var groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIME2025 | 16 | 0.00% | 69.17% | 80.00% | 69.17% | 68.96% | 11.04% | 14/30 |
| AIME2024 holdout | 16 | 0.00% | 77.92% | 90.00% | 77.71% | 77.50% | 6.88% | 15/30 |
| AMC23 | 16 | 0.00% | 95.78% | 100.00% | 95.78% | 0.00% | 0.78% | 34/40 |
| MATH-500 | 16 | 0.025% | 87.51% | 90.20% | 87.50% | 61.28% | 0.24% | 460/500 |

## 阶段二：单 seed DAPO

训练集为 AIME2025、holdout 为 AIME2024。能力增益定义为 lenient
`Avg@k` 的 paired delta；格式增益定义为 strict 与 lenient delta 的差异，
并同时报告 penalty/truncation，防止把输出模板变化误报为能力提升。

| checkpoint | AIME2025 lenient Avg@k | AIME2024 lenient Avg@k | strict-lenient gap | format penalty | truncation |
|---|---:|---:|---:|---:|---:|
| base | 69.17% | 77.92% | strict is diagnostic only | 68.96% / 77.50% | 11.04% / 6.88% |
| retry11 `iter_0000000` (one actor update; holdout N=4, cap=8192) | — | 0.00% | 0.00% | 0.00% | 100.00% |

## 阶段三：消融

每个变体固定数据 manifest 和 seed，仅改变一项：`REWARD_TYPE`（math/dapo/boxed）、
`RESPONSE_CAP`（8192/32768/65536）或训练数据规模（AIME2025/17k）。使用 paired
bootstrap 95% CI；AIME 小样本差异若 CI 跨越零，不下确定性结论。

| variant | changed knob | train Avg@k | holdout Avg@k | Pass@k | format penalty | truncation | zero-var |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | — | 69.17% / 77.92% | 77.92% / 90.00% | — | 68.96% / 77.50% | 11.04% / 6.88% | 46.67% / 50.00% |
| reward-dapo | reward | — | — | — | — | — | — |
| cap-8192 | cap | — | — | — | — | — | — |
| dapo-math-17k | data/batch/eval linked | — | — | — | — | — | — |

## 结论与回退检查

阶段一四个 base detail 文件的 `verifier_sha256` 均为
`9e916a602acab6da615a2e0722c8be9bd71012450a5ec821bce643a3133f9af5`，DAPO-Math-17k
去重后为 17,255 题。retry11 holdout 的 verifier error 为 0，但 truncation=100%、
zero-variance=100%，且输出为高重复非答案序列；因此该一次更新检查点被判定为训练
不稳定，不能作为能力提升结论。后续迭代应先检查 advantage/loss、梯度裁剪、响应
cap 和训练/评测数据配置，再进行 reward/cap/data 消融。
