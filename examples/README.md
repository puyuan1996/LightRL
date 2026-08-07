# LightRL 用户工作流

`examples/` 是仓库唯一的训练与端到端验证入口；`tools/` 只保存分析、评测和
开发诊断等辅助工具。

```text
examples/
├── training/                   # 正式训练配方
│   ├── train.sh
│   ├── train_qwen3_8b_seta_dapo.sh
│   ├── train_qwen3_8b_seta_dive_po.sh
│   ├── train_qwen3_8b_mixed_dapo.sh
│   ├── train_glm_5_1_seta_dapo.sh
│   └── world_model/             # LWM/WIP 训练与 metadata smoke
└── validation/                 # 有界端到端验证与 worker 启动
    ├── start_docker_worker.sh
    ├── validate_4gpu_seta_dapo.sh
    ├── validate_4gpu_dive_po_or_mixed.sh
    └── internal/               # 仅由上层验证入口调用
```

## 训练入口

| 脚本 | Harness | Model | 数据 | 算法 |
|---|---|---|---|---|
| `training/train_qwen3_8b_seta_dapo.sh` | Camel-Agent | Qwen3-8B | SETA | DAPO |
| `training/train_qwen3_8b_seta_dive_po.sh` | Camel-Agent | Qwen3-8B | SETA | DIVE-PO |
| `training/train_qwen3_8b_mixed_dapo.sh` | Camel-Agent | Qwen3-8B | SETA + Agent-SafetyBench + AgentHarm | DAPO |
| `training/train_glm_5_1_seta_dapo.sh` | Camel-Agent | GLM-5.1 | SETA | DAPO |
| `training/train.sh` | 由配置选择 | 由配置选择 | 由配置选择 | 由配置选择 |

`training/world_model/` 中的流程仍处于 WIP，不属于稳定训练配方。

先用 `--dry-run` 检查配置：

```bash
bash examples/training/train_qwen3_8b_seta_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_mixed_dapo.sh --dry-run
bash examples/training/train_qwen3_8b_seta_dive_po.sh --dry-run
bash examples/training/train_glm_5_1_seta_dapo.sh --dry-run
```

真实 SETA 训练需要通过 `WORKER_URLS` 或 `WORKER_URLS_FILE` 提供可达的 Docker
worker。站点专用地址放在环境变量或被 Git 忽略的
`local/cluster/worker_urls.txt`，不要写入公共配置。

若已在 rjob 内，并要复用当前的 Docker worker 启动完整 4-GPU SETA+DAPO 训练：

```bash
bash examples/training/train_qwen3_8b_seta_dapo.sh
```

脚本启动前会检查 worker 健康状态和 4 张可见 GPU，并以前台方式运行；默认
worker 地址来自 `local/cluster/worker_urls.txt`（站点专用、不入库）。可用
`WORKER_URLS` 或 `RUN_ID` 覆盖默认值；`BACKGROUND=1` 时日志写入
`runs/<RUN_ID>/launcher.log`。

## 验证入口

CPU/Docker worker 主机：

```bash
bash examples/validation/start_docker_worker.sh
```

已准备好的 4-GPU 节点：

```bash
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
  bash examples/validation/validate_4gpu_seta_dapo.sh

EXPERIMENT=dive_po \
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
  bash examples/validation/validate_4gpu_dive_po_or_mixed.sh

EXPERIMENT=mixed \
WORKER_URLS=http://<CPU_WORKER_IP>:18081 \
  bash examples/validation/validate_4gpu_dive_po_or_mixed.sh
```

`validation/internal/` 负责生成小规模数据和设置安全的训练参数，不是稳定的用户
接口；请从上述三个公开脚本进入。

GLM-5.1 训练还需要 `HF_CKPT`、`REF_LOAD` 和兼容的
`MODEL_ARGS_FILE`。完整部署、输出目录和排障说明见仓库根目录
[README](../README.md)。
