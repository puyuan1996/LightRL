# LightRL 结构重构验证记录

验证日期：2026-08-24；分支：`refactor/structure`。为不引入新仓库依赖，worker/Slime 的可选依赖被隔离安装在 `/tmp/lightrl-*-deps`，导入专用的 torch/ray/sglang 占位也仅位于 `/tmp`。

## 1. 根包与递归子模块导入

```bash
python3 -c "import agentic_rl; print('OK', agentic_rl.__all__)"
```

```text
OK ['Interaction', 'RunContext', 'TaskSpec', 'TaskTimeouts', 'TurnContext', 'TurnResult']
```

递归检查命令的核心为：

```python
modules = sorted(
    item.name
    for item in pkgutil.walk_packages(agentic_rl.__path__, "agentic_rl.")
)
for name in modules:
    importlib.import_module(name)
```

输出（LiteLLM 的无网络 fallback 和可选 PyTorch warning 省略）：

```text
ROOT_IMPORT_OK ['Interaction', 'RunContext', 'TaskSpec', 'TaskTimeouts', 'TurnContext', 'TurnResult']
MODULES_DISCOVERED 86
IMPORT_FAILURES 0
EXIT=0
```

该检查发现并修复了 `convert_task_to_dataset.py` 使用顶层 `load_tasks` 而非包内导入的旧问题。

## 2. 旧路径残留与循环 import

扫描扩展名为 `.py .md .yaml .yml .sh .toml .cfg .json .ipynb`。`docs/refactor/` 必须保留“旧路径 → 新路径”的交付映射，因此仅对该历史记录目录做豁免；所有可执行文件和其他 docs 都在扫描内。

```text
agentic_rl.inference => 0
agentic_rl.rollout.sglang_factory => 0
agentic_rl.platform.(types|env|http_client|http) => 0
agentic_rl/inference => 0
agentic_rl/rollout/sglang_factory => 0
benchmarks/agent_safetybench/ => 0
benchmarks/agent_safetybench_convert/ => 0
benchmarks/agentharm/ => 0
benchmarks/agentharm_convert/ => 0
benchmarks/mcpsafety/ => 0
benchmarks/seta_env/ => 0
benchmarks/seta_env_convert/ => 0
benchmarks/seta_env_retry/ => 0
tools/network/ => 0
tools/reproducibility/ => 0
tools/world_model/ => 0
```

注：扫描首次发现 `slime/slime/world_model/README.md` 引用了基线中就不存在的 `tools/world_model/run_world_model_seta_latent.sh`，已修正为真实入口 `examples/training/world_model/train_seta_latent.sh`后重跑为 0。

AST 依赖 DFS 会解析所有 `agentic_rl/**/*.py` 的 `Import` / `ImportFrom`，再对内部模块图查找 back edge。

```text
MODULES 87
INTERNAL_EDGES 124
CYCLES 0
<root>: types
algorithms: env, types
environments: data, env, http_client, types
harnesses: env, types
http_client: env
misc: algorithms, env, evaluation
platform: env, environments, http_server, types
rollout: algorithms, env, environments, harnesses, http_client, types
```

结果符合单向约束：没有下层包 import `platform`，没有 harness import rollout/backend，没有 evaluation 反向 import rollout。

## 3. 编译、shell 与测试

```bash
python3 -m compileall -q agentic_rl tests/agentic_rl tools
```

```text
compileall: OK
```

```bash
for f in $(git diff --name-only 64e2fb91 -- '*.sh' | rg -v '^benchmarks/'); do
  bash -n "$f" || exit 1
done
```

```text
changed operational shell files bash -n: OK
10
```

`benchmarks/environments` 中的 `solution.sh` 是任务 fixture，包含由任务 shell 在运行时启用的 extglob，不属于仓库运维 shell 语法检查范围。

仓库跟踪测试集：

```bash
PYTHONPATH="/tmp/lightrl-test-stubs:/tmp/lightrl-test-deps:$PWD" \
  .venv/bin/python -m pytest -q \
  $(git ls-files 'tests/test_*.py' 'tests/**/test_*.py')
```

```text
........................................................................ [ 28%]
........................................................................ [ 56%]
.......................s................................................ [ 84%]
........................................                                 [100%]
255 passed, 1 skipped in 7.90s
EXIT=0
```

跳过项是基线已有的条件跳过，不是重构失败。当前工作区额外存在 Git-ignore 的本地运维测试/脚本；其中一个本地测试断言旧 proxy IP，而本地脚本已使用新 IP。这些均非 Git 交付集，本次未覆盖或借机收入用户本地状态。

本库有充分的单元/集成测试，因此不再用伪造 backend 的冒烟链替代它们。跟踪测试已覆盖 dataset 转换、environment/runtime、rollout hook 导入与 SWE evaluation export。

## 4. 可扩展性指标

“文件数”按新增一个最小可用组件时需要新建或修改的 Python 文件计算，不把原始数据/Docker fixtures 的个数当作架构扩展成本。

| 扩展项 | 重构前 | 重构后 | 变化 |
|---|---:|---:|---|
| environment runtime | 2（runtime + registry） | 2（runtime + 一个 `EnvSpec`） | 文件数不变；不再触碰 platform 基础类型 |
| dataset converter | 1 | 1 | 文件数不变；资产位置由 `benchmarks/datasets` 唯一规则确定 |
| rollout backend | 4（impl + factory + 2 harness 具体 import） | 2（impl + factory registration） | **-2**；harness 只面向 `TurnClient` |
| 完整 benchmark（data+env+eval） | 6 个代码/注册点，5 类目录 | 6 个代码/注册点，3 个明确角色 | 必需文件不虚假减少；跨角色改动可选 |

环境仍需一条显式 registry 记录，这是 docs 声明的单一发现点；本次不以隐式 auto-discovery 换取表面上的 `2 → 1`。dataset 也不强制创建空 environment/evaluation 文件，所以每种关注点可独立扩展。

## 5. 删除与合并清单

| 项目 | 处置 | 依据 |
|---|---|---|
| `agentic_rl/inference/__init__.py` | delete | 1 行空包壳；具体实现已 `git mv`；全库无旧 import；非 CLI；文档已更新 |
| Tau2 两份 helper 函数体 | merge | 同名同语义，两方改为 import `data.tau2_support` |
| Dockerfile precheck 两份函数体 | merge | dev CLI 文档明示要求 byte-for-byte 与 runtime 一致，改为共用 validation |
| `benchmarks/environments/seta_env_retry/` | delete | 121 个受跟踪条目全为两次历史运行生成的软链接；无普通文件、无代码/配置引用；已加 ignore 防止回归 |
| tools 一次性脚本 | 无删除 | 全库无引用者均是独立 CLI 或有 docs，不满足三条死代码证据 |

没有 legacy 目录、re-export shim、`DeprecationWarning` 或旧路径别名。

## 6. 最终工作树与提交检查

```bash
git status --short --branch
git log --oneline 64e2fb91..HEAD
```

最终文档提交后的期望状态为干净的 `refactor/structure`，提交列表见 `PLAN.md`。
