# LightRL training recipes

训练配置直接写在 `examples/training/` 的 recipe 脚本中。每个脚本包含模型、数据集、
算法、GPU 拓扑和 rollout 配置，环境变量可覆盖默认值；不再经过 Python CLI、配置组合
或插件 registry。

```bash
# 查看最终 Slime 参数，不启动训练
bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run

# 在 4-GPU rjob 内启动完整训练（前台）
bash examples/training/train_qwen3_8b_seta_dapo.sh

# 显式后台启动
BACKGROUND=1 bash examples/training/train_qwen3_8b_seta_dapo.sh
```

调用链固定为：

```text
examples/training/<recipe>.sh
  -> agentic_rl/platform/slime_train.sh
  -> slime/train_async.py
```

`configs/rollout/` 只保留传给 rollout 的模型模板配置。站点地址、凭据和调度参数应通过
环境变量或被 Git 忽略的 `local/cluster/` 提供。
