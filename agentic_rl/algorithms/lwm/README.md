# LWM

`lwm` 是 LightRL 中面向 terminal agent 的可插拔 latent world model 组件。它负责
rollout transition 收集、独立 replay 和 offline JEPA 训练，不改变 GRPO/DAPO 的
policy loss、reward 或 advantage。

## 目录边界

| 路径 | 作用 |
| --- | --- |
| `collection.py` | AgenticRL rollout 使用的 metadata 公共接口 |
| `replay.py` | 独立 world-model replay 公共接口 |
| `__init__.py` | model、config、transition 与 rollout/replay 的 lazy public API |
| `slime/slime/world_model/` | 数据适配、hidden encoder、JEPA 模型、训练与 eval 实现 |
| `examples/training/world_model/` | 可移植启动脚本 |

包入口使用 lazy import。默认 rollout 路径只加载 metadata 接口；replay 和训练依赖仅在
显式启用相应功能时加载。

## 开关

```text
--world-model-enable
--world-model-use-dapo-replay-buffer
--world-model-replay-buffer-size 4096
--world-model-metadata-max-chars 4096
```

只启用 `--world-model-enable` 时，sample 中增加经过 redaction 和 provenance 校验的
transition metadata。继续启用 replay 开关后，Slime data source 保存独立 replay
snapshot。offline 训练入口见 `examples/training/world_model/README.md`。

当前公共入口覆盖 offline model 与 rollout/replay。online auxiliary policy loss 尚未接入
LightRL 训练主路径。
