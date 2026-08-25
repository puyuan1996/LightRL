# SETA fixed12 评估集与协议

本文记录 `seta_fixed12_score_v1` 的样本构成、历史选择方法、当前运行参数和解释边界。它是 DAPO 与 DIVE-PO 长训期间的低成本趋势哨兵，不是用于探索性评估或统计显著性检验的完整 benchmark。

## 1. 权威文件

- 样本：`benchmarks/datasets/seta_env_convert/eval_fixed12.jsonl`
- 评估配置：`configs/evaluation/seta_fixed12_score_v1.yaml`
- 当前成对实验入口：`runs/refactor/launch_seta_fixed12_score_8g.sh`
- 当前训练集：`benchmarks/datasets/seta_env_convert/train_minus_eval12.filtered.jsonl`
- fixed12 SHA256：`5bba3ce8a02e7f116692a254958c2d4e6540f29dcbfdceb12122864384385042`
- 当前训练集 SHA256：`fe550642e1b4c2fed55ce048659d2a14f6929a721baa817629767cb4f5a13a52`

正式 launcher 在启动前校验这两个哈希。DAPO 与 DIVE-PO 必须使用同一份 fixed12，且训练期间不得修改样本或顺序。

## 2. 12 个样本

| 原始位置（0-based） | task id | 难度 | 类别 | 任务概要 |
| ---: | ---: | --- | --- | --- |
| 0 | 705 | hard | linux-administration | 修复 systemd 下 Python 脚本的虚拟环境、工作目录与环境变量 |
| 125 | 850 | hard | system-administration | 配置 `atd`、`at.allow`、定时任务及审计脚本 |
| 250 | 774 | medium | system-administration | 诊断 sudoers 多条匹配规则导致 `NOPASSWD` 不生效 |
| 375 | 881 | hard | system-administration | 配置 PAM 密码长度、重试、哈希和历史策略 |
| 500 | 300 | medium | software-engineering | 演示 GNU Screen 的启动、分离、重连与会话检查 |
| 625 | 1136 | medium | debugging | 诊断精简版 Vim 缺少命令，保持 `.vimrc` 不变并修复系统安装 |
| 750 | 448 | hard | software-engineering | 实现多进程作业注册、完成通知 daemon 和状态查询工具 |
| 875 | 622 | medium | system-administration | 配置活动记录隐私黑名单并清理 SQLite 历史数据 |
| 1000 | 842 | medium | system-administration | 分析 apt held packages、依赖阻塞并生成报告，不实际升级 |
| 1125 | 695 | hard | software-engineering | 升级 CMake/GCC 并构建使用 C++17 的旧项目 |
| 1250 | 823 | hard | system-administration | 修复中断升级后的 dpkg/APT 状态、仓库配置和 nginx hold |
| 1375 | 634 | hard | system-administration | 实现进程 watchdog、优雅终止、黑名单和恢复日志 |

分布为：

- 难度：7 hard、5 medium、0 easy；
- 类别：7 system-administration、3 software-engineering、1 linux-administration、1 debugging。

作为参照，原始1376题中 hard/medium/easy 为722/629/25（约52.5%/45.7%/1.8%）；fixed12 为58.3%/41.7%/0%。难度比例大致接近，但类别覆盖明显偏向系统管理，因此不能称为严格代表总体的分层样本。

## 3. 如何选取

fixed12 在提交 `b998a985`（2026-08-07）中首次加入。源集合是当时的 `benchmarks/datasets/seta_env_convert/train.jsonl`，共 1376 个互异任务。

选择方法是按该 JSONL 的既有行序做确定性的系统抽样：

```text
position_i = 125 × i,  i = 0, 1, ..., 11
```

因此选中位置是 `0, 125, 250, …, 1375`，task id 依次为：

```text
705, 850, 774, 881, 300, 1136, 448, 622, 842, 695, 823, 634
```

这里的“等间隔”指原始 JSONL 行号等间隔，不是 task id 数值等间隔；该过程没有随机 seed，也没有按难度或类别主动配额。它利用源数据的既有顺序，把12个点分散到整个序列，避免只截取文件头部或连续任务。

选中后，12题被从训练集合中留出：原始补集 `train_minus_eval12.jsonl` 有1364题；当前正式实验进一步使用过滤后的 `train_minus_eval12.filtered.jsonl`，有1344题。当前训练文件与 fixed12 的 task-id 交集为空。

注意：`tools/evaluation/build_seta_fixed_eval.py` 是后来扩展 fixed48 的分层选择器。它保留 fixed12，再从未见任务中补36题；它不是 fixed12 的原始生成方法。

## 4. 当前评估协议

当前 DAPO/DIVE-PO 成对长训冻结如下参数：

- 点位：rollout-step `100, 200, …, 1000`；
- 每题生成数：`n=1`；
- 解码：`temperature=0.0`、`top_p=1.0`、`top_k=-1`；
- 最大响应：8192 tokens；
- seed：`20260809`；
- worker 并发：最多3个 fixed12 rollout；
- 指标：12题的任务级原始 terminal-test score/完成率及其均值；
- 两个算法使用完全相同的样本、顺序、解码配置、seed 和评估点。

评估结果只在12题全部结束后发布。Docker build/reset/allocate、grader parser/no-results 等基础设施失败不能当作普通0分；出现这类失败或缺题时，该点应标为无效并补做或停止归因。候选程序自身导致的测试失败或合法测试超时仍是任务结果0分。

## 5. 设计原理

### 固定而非在线随机

每个 checkpoint 都评估同一组任务，可以做 task-paired 的纵向比较，减少因每次抽到不同题目产生的方差。固定集合也使 DAPO 与 DIVE-PO 的同点位差值更容易归因。

### 与训练严格隔离

fixed12 从训练 JSONL 中删除，避免周期评估题直接进入 policy update。launcher 同时校验评估集和训练集哈希，防止运行间悄然漂移。

### 控制长训成本

SETA 每题需要创建 Docker 环境、多轮工具调用和真实 grader。12题、确定性单次解码的成本远低于多样本 pass@k，适合每100 rollout-step 重复执行。

### 覆盖序列而非局部截断

按源文件等间隔取样覆盖了完整序列范围，优于“前12题”或连续区间。实际结果也覆盖 hard/medium 以及系统管理、软件工程和调试任务。

## 6. 局限与正确解释

fixed12 只能作为趋势哨兵：

- 样本量很小。若把每题简化为独立二项结果，最坏情况下标准误约14.4个百分点，普通正态近似的95%半宽约28.3个百分点；
- 它不是随机样本。代表性依赖原始 JSONL 行序，若源文件排序带有结构，系统抽样会继承该结构；
- 类别并不按总体比例分层：system-administration 占7/12，software-engineering 仅3/12，且没有 easy 题；
- `n=1, temperature=0` 主要衡量当前策略在固定任务上的确定性任务完成能力，不能衡量 pass@k、生成多样性或探索覆盖；
- 训练 batch raw reward 与 fixed12 held-out score 含义不同，不能混成一条曲线；
- 12题均值的细小波动不应被描述为显著算法收益。应同时查看逐题配对差值，并在完整长训后结合更大评估集或多 seed 复验。

因此，当前实验中 fixed12 的职责是尽早发现明显退化、比较同点位方向和验证训练是否转化为 held-out 能力。DIVE-PO 的探索/利用结论应在1000步完成后，结合局内/局间内外在奖励、UCB 臂选择、策略熵及更强评估协议得出。

## 7. 变更规则

1. 当前 DAPO/DIVE-PO 成对实验运行期间不得替换 fixed12；否则已有点位不可比较。
2. 若需更强代表性，另建版本化协议和全新 run id，不覆盖 `seta_fixed12_score_v1`。
3. 新协议应生成 manifest，记录源数据哈希、候选排除条件、seed、分层目标、task id、顺序及评估配置。
4. 推荐复用 fixed48 的思路：按难度和类别分层、排除曾用于调试的任务，并保留逐任务原始结果；但不要在当前1000步实验中途切换。

## 8. 快速复核

```bash
sha256sum benchmarks/datasets/seta_env_convert/eval_fixed12.jsonl
wc -l benchmarks/datasets/seta_env_convert/eval_fixed12.jsonl \
      benchmarks/datasets/seta_env_convert/train_minus_eval12.filtered.jsonl
```

预期 fixed12 为12行，哈希为本文记录值；当前 filtered 训练集为1344行。正式实验的最终 `run_config.json` 还应记录相同的 `eval_protocol`、`eval_seed`、`eval_steps` 和 `eval_set_sha256`。
