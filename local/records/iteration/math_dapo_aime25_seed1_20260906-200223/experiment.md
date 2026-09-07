# Math RLVR：AIME2025 单 seed DAPO + AIME2024 holdout

实验记录目录：`math_dapo_aime25_seed1_20260906-200223`

## 1. 实验目的

提交一个单 seed 的 AIME2025 DAPO 训练任务，并在训练结束后用同一 checkpoint
对 AIME2024 holdout 做独立评测。训练和评测均
使用 `reward_type=math`，不要求模型输出固定的 `Answer:` 或 `\\boxed{}` 格式；
`Answer:`、`\\boxed{}`、自然语言 final-answer 由同一 extractor/verifier 处理。
strict/boxed 指标仅作为诊断轨迹保留，不参与本次训练更新。

## 2. 固定实现与一致性约束

- 训练入口：[train_qwen3_8b_dapo_math.sh](../../../../examples/training/train_qwen3_8b_dapo_math.sh)
- RJob payload：[submit_math_rlvr_train.sh](../../../../local/rjob/submit_math_rlvr_train.sh)（Pod 内直接调用训练入口）
- 训练 custom RM：`tools.evaluation.math_rlvr.reward.reward_func`
- 评测 verifier：`tools/evaluation/math_rlvr/verifier.py`
- verifier SHA-256：`9e916a602acab6da615a2e0722c8be9bd71012450a5ec821bce643a3133f9af5`
- extractor SHA-256：`370637b80c404374f8d26efc8cb302298be492708292b594fbe185dfa0c22dfe`
- reward adapter SHA-256：`4cb76bda26b83fd1bc924f9cf26a4e72ba93c18c4789341ff5a4cfd6b434646a`

训练端和离线评测端都使用同一个 `verifier.py`。`math` track 的 canonical 候选
优先级为：最后一个完整 `Answer:` 行、最后一个完整 boxed 值、最后一个自然语言
final-answer 片段；候选冲突会进入 per-sample 记录，不会静默覆盖。

## 3. 数据与模型

| 项目 | 值 |
|---|---|
| base checkpoint | `${HF_CKPT}` |
| reference checkpoint | `${REF_LOAD}` |
| 数据根目录 | `${MATH_DATA_ROOT}`（默认由 `data.py` 解析 canonical data root） |
| 训练集 | `aime-2025.jsonl`，30 题 |
| holdout | `aime-2024.jsonl`，30 题 |
| AIME2025 SHA-256 | `a6ec56fd56a5f3aa4f3ff74697dd528968c4e79b682658f32d7ce744c3c1a84b` |
| AIME2024 SHA-256 | `c39f884fbf71972ad5766e0870d4077f7ce695fe947e19ea60f283dccfb0c788` |

## 4. 训练配置

| 参数 | 值 |
|---|---:|
| reward | `math` |
| response cap | `8192` (retry11 OOM-safe training validation; 32768 remains the evaluation default) |
| seed | `1` |
| rollout batch size | `1` (retry7 OOM-safe validation) |
| samples per prompt | `4` |
| global batch size | `4` |
| rollout steps | `4` |
| in-training eval | `aime-2025,aime-2024` |
| eval samples/prompt | `4` |
| eval interval / save interval | `20 / 20` |
| actor / rollout GPUs | `2 / 2`（总计 4 卡） |
| rollout topology | 非 colocate；`train_async.py` 明确禁止 `--colocate` |
| train backend | `megatron` |

选择 `math` 而不是 `dapo` 是本实验的关键隔离条件：模型不因是否写出
`Answer:` 而获得或失去训练奖励。AIME2024 只用于 in-training 监控和训练后的
独立 holdout，不混入训练数据。

## 5. RJob 记录

集群 namespace、quota group、镜像、挂载和持久化目录均从未提交的
`local/rjob/rjob.env` 或环境变量注入；仓库不保存站点地址。以下仅记录任务
逻辑状态和资源规模，不复制集群凭据或地址。

- 资源：4 GPU / 50 CPU / 560000 MiB
- priority：9
- 首次训练 job：`math-dapo-aime25-seed1-20260906-200223`（失败，容器内误调用 `rjob`，exit 127；日志已保留）
- 修复后训练 job：`math-dapo-aime25-seed1-20260906-200223-retry1`（失败，容器缺少 `megatron.training` 的 PYTHONPATH，日志已保留）
- 二次修复训练 job：`math-dapo-aime25-seed1-20260906-200223-retry2`（失败，Qwen3 HF/Megatron 架构参数未对齐，日志已保留）
- 三次修复训练 job：`math-dapo-aime25-seed1-20260906-200223-retry3`（失败，`train_async.py` 禁止 colocate，已改为 2+2 卡非 colocate）
- 四次修复训练 job：`math-dapo-aime25-seed1-20260906-200223-retry4`（失败，Ray worker 无法导入协作者未提交的 `bridge_compat.py`，日志已保留）
- 五次修复训练 job：`math-dapo-aime25-seed1-20260906-200223-retry5`（失败，Megatron tokenizer vocab=151643 与 HF/SGLang vocab=151936 不一致）
- retry6：`math-dapo-aime25-seed1-20260906-200223-retry6`（已停止；首轮 rollout
  在 KV cache 接近满载时触发 SGLang CUDA OOM，完整日志已保留）
- retry7：`math-dapo-aime25-seed1-20260906-200223-retry7`（失败；低并发后在训练
  batch 反向阶段触发 Transformer Engine/cuDNN `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`，无 checkpoint）
- retry8：`math-dapo-aime25-seed1-20260906-200223-retry8`（失败；`local` 实现与 Qwen3
  RMSNorm 不兼容，Megatron `FusedLayerNorm` 断言失败，无 checkpoint）
- retry9：`math-dapo-aime25-seed1-20260906-200223-retry9`（失败；显式环境变量
  `NVTE_FUSED_ATTN=0` 与 Megatron auto attention backend 冲突，无 checkpoint）
- retry10：`math-dapo-aime25-seed1-20260906-200223-retry10`（失败；32768-token
  response 的 actor log-prob logits 约 9.31 GiB，训练阶段 CUDA OOM，无 checkpoint）
- retry11：`math-dapo-aime25-seed1-20260906-200223-retry11`（已失败，但首个 actor
  update 成功并保存了有效的 `iter_0000000`；第二次 rollout/actor train 在
  `float16_to_fp32` 阶段申请约 4.78 GiB 时 CUDA OOM）。训练使用
  `response_cap=8192`，显式设置 `--transformer-impl transformer_engine` 和
  `--attention-backend flash`。
  首个更新日志指标为 `global_step=1`、`loss=1359.877`、`entropy_loss=11.931`、
  `ppo_kl=1.7805`；随后第二次 rollout/训练失败，未宣称完成多步 DAPO 收敛。
- 10-epoch v1：`math-dapo-aime25-seed1-10epoch-v1`（启动阶段失败；Megatron TP
  要求 `CUDA_DEVICE_MAX_CONNECTIONS=1`，未进入模型训练）。
- 10-epoch v2：`math-dapo-aime25-seed1-10epoch-v2`（已失败；rollout 0--10 已
  产生训练指标，随后 RJob 挂载的工作区被切换到不含 Math RLVR 文件的分支，
  `reward.py` 的逐样本 `verifier_digest()` 抛出 `FileNotFoundError`）。该故障不是
  显存或 verifier 语义错误；日志保留在对应 RUN_DIR。
- 10-epoch v3：`math-dapo-aime25-seed1-10epoch-v3`（使用修复后的 verifier digest
  缓存，但未形成可验收的十 epoch 结果）。
- isolated-v1：使用独立 source worktree 验证共享分支切换问题；任务最终失败，未
  产生可验收的十 epoch checkpoint。当前分支不宣称 DAPO 已完成十 epoch。
- 训练输出根目录：
  `${PERSIST_ROOT}/runs/training/math-dapo-aime25-seed1-20260906-200223-retry11`
- 训练 checkpoint 目录：
  `.../math-dapo-aime25-seed1-20260906-200223-retry11/checkpoints/iter_0000000`
  （来自首个 actor update；不是完整多步训练结果，但可作为明确的 step-0
  训练后检查点进行 holdout）
- 训练日志：上述目录下的 `logs/train.log`；retry6--retry10 日志在各自同名目录下
训练检查点转换与 holdout 评测已提交：

```bash
MODEL_PATH=<retry11 iter_0000000 转换后的 HF checkpoint>
MODEL=<服务名>
DATASETS=aime-2024
REWARD_TYPE=math
N=4
MAX_TOKENS=8192
OUTPUT_DIR=<本目录>/holdout_eval
bash local/rjob/submit_math_rlvr_eval.sh
```

转换 RJob：`math-dapo-aime25-seed1-retry11-convert-retry1`（Succeeded）。HF 输出：
`${PERSIST_ROOT}/runs/conversion/math-dapo-aime25-seed1-retry11-hf`。
holdout exploratory RJob：`math-dapo-aime25-seed1-retry11-aime24-holdou-67b37`（已停止，
4 卡 TP、N=16、cap=32768、concurrency=4；因长思考输出显示 cap 可能主导耗时，
已停止并保留 SGLang 日志，未将其当作最终指标。主 holdout RJob：
`math-dapo-aime25-seed1-retry11-aime24-holdou-d139d`（Succeeded，4 卡 TP、N=4、
cap=8192、concurrency=4），与 retry11 训练 response cap 一致。评测输出需同时
保留 detail JSON、summary JSON 和 SGLang 日志，并在本文件的结果表中登记 verifier
hash、checkpoint 路径、任务名和时间。

## 6. 结果（提交后更新）

| checkpoint | AIME2025 math Avg@k | AIME2025 Pass@k | AIME2024 holdout math Avg@k | AIME2024 Pass@k | format penalty | truncation | zero-var |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 69.17% | 80.00% | 77.92% | 90.00% | 68.96% / 77.50% | 11.04% / 6.88% | 46.67% / 50.00% |
| retry11 `iter_0000000`（一次 actor update；AIME2024 N=4 cap=8192） | — | — | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |

主 holdout 已完成：`math-dapo-aime25-seed1-retry11-aime24-holdou-d139d`（Succeeded，
耗时 866.43 s）。逐样本文件、summary 和 SGLang 日志已复制到本目录的
`holdout_cap8192/`；原始持久化路径为
`${PERSIST_ROOT}/runs/evaluation/math-dapo-aime25-seed1-retry11/aime24_holdout_cap8192/`。

本次结果的关键诊断是：120/120 样本 `finish_reason=length`、truncation=100%，
zero-variance group=30/30，`verifier_error_count=0`，且模型输出出现高重复/非答案
token 序列，因此 0 分不能解释为 extractor 严格限制格式。它表明 retry11 的单次
actor update 已造成明显训练不稳定/能力崩溃，需在后续迭代回查 advantage/loss、
梯度裁剪和响应 cap；不应把该 checkpoint 当作能力提升结论。

表中 base 数值来自此前 n=16 基线；holdout 结果从 detail/summary 文件读取，不能
从训练 stdout 手工估算。当前没有可与 base 严格配对的 AIME2025 训练后评测，因此
不填 AIME2025 的 retry11 数值。格式增益与能力增益分开计算：能力使用
`lenient/math Avg@k` 的 paired delta，格式变化使用 strict 与 lenient 的差值及
format penalty 变化。

## 7. 验收与异常处理

1. 训练日志确认 `reward_type=math`、训练行数、AIME2025/AIME2024 eval 列表和
   verifier hash 均已打印。
2. checkpoint 必须来自本次 job 的 `RUN_DIR`，禁止使用默认或旧 checkpoint。
3. holdout 评测 `verifier_error_count` 应为 0；若 truncation 超过 5%，追加更高
   cap 的敏感性任务；若 zero-variance 占比很高，记录为 DAPO 梯度风险而不是
   伪造非零 advantage。
4. 若 job 在启动阶段失败，先保存 rjob describe/logs，再检查模型路径、Ray/SGLang
   资源拓扑和 shared-FS 挂载，不更改 reward 规则来绕过基础设施错误。

## 8. 10-epoch 迭代诊断

v2 的首轮有效训练步证明 TP=2、Transformer Engine/Flash attention、full
recompute 和 4096 response cap 的组合可以完成 actor update；但指标显示
`rollout/truncated_ratio` 为 0.75--1.0，多组 reward zero-variance，且部分步的
`grad_norm_pre_clip` 达到 `1e9--1e12` 后被 clip。该运行在基础设施异常前没有
宣称能力收敛。v3 保留同一数据、reward 和 batch 配置，先验证缓存 digest 修复；
若 v3 完成后仍出现梯度崩溃，再以独立任务调整 response cap/LR，并将每次变更与
checkpoint 路径追加到本节。
