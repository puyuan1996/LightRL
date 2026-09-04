# JEPA LWM 离线验证实验报告（分支版）

分支：`feat/lwm-offline-verify`
日期：2026-09-04
起点：[LightRL PR #2](https://github.com/puyuan1996/LightRL/pull/2)，本地
`MING-ZCH:jepa_wm` 提交 `3b52ef04`。

## 当前可复现实验

本报告先记录无外部网络依赖的 hash-hidden smoke，用于验证数据适配、AdaLN
条件化、replay 生命周期、value loss 和 MPC 输出。正式语义结论必须将同样的命令
切换到 `WM_ENCODER=hf-policy` 和本地 Qwen3 checkpoint。

| 阶段 | 数据 | 配置 | train loss | 结果 |
| --- | --- | --- | ---: | --- |
| baseline（修复后） | tb2.1 ATIF，1 trajectory/4 transitions | replay off，value off，latent=16 | 0.22406 | `metrics.jsonl`、cache、checkpoint、predictions 全部生成 |
| replay | tb2.1 ATIF，2 trajectories/8 transitions | FIFO replay=16，1 epoch | 0.22309 | replay snapshot 与每 epoch 抽样工作 |
| value + MPC | tb2.1 ATIF，1 完整 trajectory/57 transitions | value coef=0.5，gamma=0.99，1 epoch | 0.42045 | value mask=1；`mpc.json` 评分 57 个候选并选择 index 37 |

smoke 使用的 Python 为 `/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python`；
输出写在 `/tmp/lwm-smoke-final/`，不会污染仓库或运行中的训练目录。此前单轨迹
被全部划入 validation 的边界已修复为 train-only split。测试命令：

```bash
PYTHONPATH=slime:. /mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python \
  -m pytest slime/tests/world_model/test_seta_dataset.py \
  slime/tests/world_model/test_modules.py slime/tests/world_model/test_mpc.py \
  slime/tests/world_model/test_loss_hook.py -q
```

结果：`11 passed`。`bash -n` 已通过新增 phase/rjob 脚本；`python -m py_compile`
已通过变更 Python 模块。

## tb2.1 数据审计与正式运行

loader 已能从当前 `runs/evaluation` 中识别 ATIF `*/agent/trajectory.json`，并从
相邻 `verifier/reward.txt` 读取终局分数；`data_manifest.json` 会记录 trajectory、
task、terminal、tool-feedback、reward 和 source histogram。默认 `auto` 顺序为
tb2.1 → SETA → records/replay；训练补充数据必须通过 `WM_SUPPLEMENT_INPUT` 或
`--supplement-input` 明确提供。

正式三阶段命令：

```bash
# 先做 smoke，再改为本地真实 policy hidden
WM_ENCODER=hf-policy WM_HF_MODEL=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
WM_PHASE=baseline bash examples/training/world_model/run_tb21_lwm_phase.sh
WM_ENCODER=hf-policy WM_HF_MODEL=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
WM_PHASE=replay WM_BACKPROP_TO_LLM=1 \
  bash examples/training/world_model/run_tb21_lwm_phase.sh
WM_ENCODER=hf-policy WM_HF_MODEL=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
WM_PHASE=value_mpc bash examples/training/world_model/run_tb21_lwm_phase.sh
```

集群提交器会为三个阶段创建独立 RJob，并保留完整 stdout、`metrics.jsonl`、
manifest、checkpoint 和 MPC 结果：

```bash
WM_ENCODER=hf-policy WM_HF_MODEL=/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B \
  bash examples/training/world_model/submit_tb21_lwm_rjob.sh
```

当前开发容器到 RJob API 的 DNS 不可达，因此本次只完成了 submission command
与 `rjob submit --help` 的参数校验，没有擅自创建消耗 GPU 的远程任务。可达集群上
执行上述命令即可开始三阶段实验；若训练数据或模型不可用，脚本会在启动前 fail
closed，而不会把 archive/debug 目录当作数据。

## 结论与下一步

- 数据、shape、AdaLN “action 只作条件”、replay、value 和 one-step MPC 的工程闭环
  已通过；新增单元测试覆盖了 tb2.1 ATIF 解析、trajectory split 和 planner fail-closed。
- hash hidden 不提供语义证据；目前不能据此声称 Agentic RL return 提升。
- 正式验收应比较同一 trajectory split 下的 held-out prediction、shuffle-action gap、
  value MAE/Spearman、MPC regret，以及 baseline/replay 的 wall-clock 与 policy
  return。只有这些指标稳定改善，才考虑把预计算 latent hook 接入线上 DAPO。
