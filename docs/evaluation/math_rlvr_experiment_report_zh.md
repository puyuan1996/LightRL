# Math RLVR 实验报告（可复现实验记录）

本文件是阶段一至三的结果表。脚本会把原始逐样本记录写入运行目录；完成
`rjob` 作业后，将 `summary.json` 和 `math_paired_stats.py` 的输出填入这里，
不要手工从日志抄取单一准确率。

## 配置指纹

| 字段 | 值 |
|---|---|
| branch | `feat/math-rlvr-eval` |
| base checkpoint | `<HF_CKPT>` |
| reference checkpoint | `<REF_LOAD>` |
| train manifest SHA-256 | `<填入>` |
| verifier SHA-256 | `<填入>` |
| reward / cap / seed | `math / 32768 / <填入>` |

## 阶段一：base 评测

| dataset | n | strict Avg@k | lenient Avg@k | Pass@k | boxed Avg@k | format mismatch | truncation | zero-var groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIME2025 | 16 | — | — | — | — | — | — | — |
| AIME2024 holdout | 16 | — | — | — | — | — | — | — |
| AMC23 | 16 | — | — | — | — | — | — | — |
| MATH-500 | 16 | — | — | — | — | — | — | — |

## 阶段二：单 seed DAPO

训练集为 AIME2025、holdout 为 AIME2024。能力增益定义为 lenient
`Avg@k` 的 paired delta；格式增益定义为 strict 与 lenient delta 的差异，
并同时报告 penalty/truncation，防止把输出模板变化误报为能力提升。

| checkpoint | AIME2025 lenient Avg@k | AIME2024 lenient Avg@k | strict-lenient gap | format penalty | truncation |
|---|---:|---:|---:|---:|---:|
| base | — | — | — | — | — |
| final | — | — | — | — | — |

## 阶段三：消融

每个变体固定数据 manifest 和 seed，仅改变一项：`REWARD_TYPE`（math/dapo/boxed）、
`RESPONSE_CAP`（8192/32768/65536）或训练数据规模（AIME2025/17k）。使用 paired
bootstrap 95% CI；AIME 小样本差异若 CI 跨越零，不下确定性结论。

| variant | changed knob | train Avg@k | holdout Avg@k | Pass@k | format penalty | truncation | zero-var |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | — | — | — | — | — | — | — |
| reward-dapo | reward | — | — | — | — | — | — |
| cap-8192 | cap | — | — | — | — | — | — |
| dapo-math-17k | data/batch/eval linked | — | — | — | — | — | — |

## 结论与回退检查

在填写结果前确认：每个 detail 文件中的 `verifier_sha256` 相同；manifest 的
unique count 对 DAPO-Math-17k 约为 17,255；response cap 的 truncation 不超过
预设阈值；zero-variance group 没有被伪造为非零 advantage。若任一条件失败，
先检查 extractor 候选冲突、verifier 版本、cap 和数据路径，再比较模型能力。
