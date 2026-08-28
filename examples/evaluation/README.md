# 评估配方入口(Evaluation Recipes)

本目录是**离线评估的用户入口**:给出典型评测场景的配方(配置 +
命令)。评测机制本身(适配层、runner、CLI)在
[`tools/evaluation/`](../../tools/evaluation/README.md),
harness 适配器在 `agentic_rl/harnesses/eval/`;本目录不放代码,
只放"怎么跑"的配方,与 `examples/training/`(训练配方)对称。

三个目录的分工:

| 位置 | 角色 |
|---|---|
| `examples/evaluation/`(本目录) | 用户配方:选哪个 harness、配什么参数、跑哪条命令 |
| `tools/evaluation/` | 通用评估工具集:`eval_cli.py`、core/、configs/、site/ |
| `agentic_rl/harnesses/eval/` | harness 适配层:terminus-2 / claude-code-cli / camel-agent |

## 配方 1:Terminal-Bench 风格 benchmark(terminus-2)

Harbor + terminus-2,适用于 terminal-bench-2.x 这类 docker 任务集:

```bash
cp tools/evaluation/configs/tb21_terminus2.example.yaml my-tb-eval.yaml
# 编辑 dataset.path / serving.model_path / serving.model_name / output_dir
python3 tools/evaluation/eval_cli.py run --config my-tb-eval.yaml --dry-run
python3 tools/evaluation/eval_cli.py run --config my-tb-eval.yaml
```

先单题冒烟再全量:

```bash
python3 tools/evaluation/eval_cli.py smoke --config my-tb-eval.yaml --task <task-name>
```

常用覆盖(不改配置文件):

```bash
python3 tools/evaluation/eval_cli.py run --config my-tb-eval.yaml \
  --set serving.model_path=/path/to/ckpt --set serving.model_name=my-ckpt \
  --set run.concurrency=16 --set run.max_input_tokens=32768
```

## 配方 2:批量评估多个 ckpt 并对比

```bash
cp tools/evaluation/configs/batch.example.yaml my-batch.yaml
# 在 models: 列表里逐个填 model_path / model_name
python3 tools/evaluation/eval_cli.py batch --config my-batch.yaml
python3 tools/evaluation/eval_cli.py report \
  --results "my-batch-output/*/eval_result.json" --output my-batch-output/compare
```

产物:`compare.md`(模型 × pass@1 / mean_reward / 异常分布 对比表)+
`compare.csv`。

## 配方 3:LightRL 自研链路(camel-agent / SETA)

走 slime `eval_only` 重量级运行时(自行拉起推理引擎,`serving` 段不生效):

```bash
cp tools/evaluation/configs/seta_camel.example.yaml my-seta-eval.yaml
# 编辑 extra.slime_root / extra.slime_args(--hf-checkpoint、--load、--prompt-data 等)
python3 tools/evaluation/eval_cli.py run --config my-seta-eval.yaml
```

## 配方 4:Claude Code CLI 作为 agent

```bash
cp tools/evaluation/configs/tb21_claude_code.example.yaml my-cc-eval.yaml
# agent_kwargs / agent_env 需按所用 Harbor 版本核对(见 tools/evaluation/README 备注)
python3 tools/evaluation/eval_cli.py run --config my-cc-eval.yaml
```

## 站点差异怎么处理

节点/集群强相关的配置(代理、docker 网络覆盖、relay、远端执行)一律
不进通用配置,见 `tools/evaluation/site/` 的说明与 profile 示例。
