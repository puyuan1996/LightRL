# JEPA World Model Research Log

## 2026-08-13：迁移到 LightRL

- Motivation：OpenClaw-RL 已迁移为 LightRL，需要保留 PR #19 的严格评测约束、PR #21 的 replay/offline trainer，以及 7 月 23 日之后的 next-belief 实验实现。
- 假设：在 LightRL 当前 package 边界内，world model 可以保持 default-off，通过 `agentic_rl.rollout` 生成 metadata，通过 Slime data source 保存独立 replay，并复用原 offline trainer。
- 修改内容：迁入 latest world-model modules/tests；增加 LightRL rollout metadata 接口、独立 replay adapter 和通用训练脚本；历史 `openclaw_*` schema 保留用于 artifact compatibility。
- 成功现象：240 项 world-model CPU tests 通过；8 条 transition 的 script-level hash smoke 完成，输出 `records.jsonl`、cache、checkpoint、predictions 与 `run_summary.json`；迁移文件通过 `py_compile`、`bash -n` 和 `git diff --check`。
- 失败现象：开发机系统 Python 存在 NumPy/Torch ABI 不兼容且缺少 `transformers`；测试改用 `/root/miniconda3/bin/python`。该环境缺少 `ray`，因此未运行 Slime data-source runtime test。
- 结论：offline JEPA 路径与通用脚本在 LightRL 中可运行。rollout replay 接入已完成静态检查，仍需在完整 LightRL 训练镜像执行一轮 default-off/default-on smoke。

## 2026-08-13：阶段结果复核

- Motivation：迁移提交需要区分已完成结论、开发性结果和运行中实验，避免把局部 retrieval 变化解释为 latent 机制结论。
- 实验假设：dual-target 配置可能改善 observed result retrieval；提高 SIGReg 强度或使用精确配置可能缓解 predicted-latent collapse。
- 成功现象：既有 8-fold 实验中，JEPA 相对 parameter-matched Direct 的 next-belief MRR 差值为 `+0.10994`，`8/8` folds 为正；dual-target 在两个开发 seed 上相对 next-only 的 result-transfer MRR 分别提高 `+0.04924/+0.08658`。
- 失败现象：dual-target 的 result MRR 仍低于 Direct，next-state MRR 同时下降；上一轮 paired run 的 seed 13 configured arm 提前停止，原因未查明，且已完成 arm 的 collapse 诊断未通过。
- 修改内容：阶段文档将 dual-target 结论限定为当前 task-heldout split 上的 observational result retrieval；新增 recovery anti-collapse 实验状态。
- 当前状态：recovery 实验的 preflight 与第一阶段 barrier 已通过，seed 11/13 的 configured 和 unconfigured 四个 arm 均在运行，尚无 aggregate 结果。
- 结论：保留“当前 SETA 数据内 JEPA 提高 next-belief retrieval”的阶段结论。result prediction、tool choice 与 online Terminal-RL 增益继续列为待验证目标。

## 2026-08-13：LightRL 算法入口与示例

- Motivation：LightRL 需要明确的 AgenticRL 算法归属和可直接使用的示例入口，旧 metadata smoke 仍包含当前 parser 未注册的参数。
- 实验假设：使用轻量 facade 复用 `slime.world_model`，可以在不复制训练实现的情况下形成稳定公共 API；示例脚本统一调用通用 trainer，可以减少参数漂移。
- 修改内容：新增 `agentic_rl/algorithms/lwm` 的 collection/replay 接口；rollout 改用该接口；新增 offline、next-belief 和 replay collection 示例；删除 metadata smoke 的失效参数。
- 成功现象：world-model 与 public API 共 `242 passed`；四个示例脚本通过 `bash -n`；metadata smoke dry-run 未出现旧参数；replay dry-run 的最终 `train_async.py` 命令包含四个 collection 参数并启用 checkpoint 保存。offline trainer 的 hash smoke 已在上一轮迁移验证完成。
- 失败现象：暂无。
- 结论：LightRL 已具备稳定的 LWM 算法入口、offline 示例和 rollout replay collection 示例。online auxiliary policy loss 仍未接入训练主路径。
