# Hermes Cosmos

**面向AI大模型训练的全球统一调度与容错计算系统**

## 项目概述

Hermes 是一个全球跨Region、万卡级GPU、高容错、碳感知的AI训练与推理调度系统。

### 核心特性

- **全球调度器**: 跨Region资源抽象、基于MCTS的放置策略、抢占与弹性配额
- **分布式Checkpoint**: 分布式内存Checkpoint、异步多级存储、Delta状态差分
- **智能自愈系统**: 故障预测Agent、自动迁移与根因分析引擎
- **可观测性平台**: eBPF GPU指标采集、算子级Tracing、碳排放计量

## 项目结构

```
hermes-cosmos/
├── src/hermes/
│   ├── core/            # 核心类型和配置
│   ├── gateway/         # 统一API网关
│   ├── scheduler/       # 全球调度器
│   ├── checkpoint/      # 分布式Checkpoint服务
│   ├── agent/           # 本地集群Agent
│   ├── fault_prediction/# 故障预测模块
│   └── cli.py           # 命令行工具
├── config/              # 配置文件
├── docker/              # Docker配置
├── deploy/              # 部署配置
│   ├── alibaba/         # 阿里云部署脚本
│   ├── aws/             # AWS部署脚本
│   ├── kubernetes/      # Kubernetes部署配置
│   └── prometheus/      # Prometheus监控配置
├── examples/            # 使用示例
├── hermes-operator/     # Kubernetes Operator
├── grafana/             # Grafana仪表盘配置
└── tests/               # 测试文件
```

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/hermes-cosmos/hermes.git
cd hermes

# 安装依赖
pip install -e ".[dev]"
```

### 使用Docker Compose启动

```bash
cd docker
docker-compose up -d
```

### 使用CLI

```bash
# 检查系统状态
hermes-cli status

# 提交训练作业
hermes-cli jobs submit my-job \
  --tenant-id tenant-1 \
  --user-id user-1 \
  --image pytorch/pytorch:latest \
  --gpu-count 8

# 列出作业
hermes-cli jobs list

# 查看资源摘要
hermes-cli resources summary
```

### Self-Researcher：让系统撰写关于自身的论文

`autotestgen.py` 内置的 Self-Researcher 会读取系统自身的演化历史（`loop_output/` 下的快照、事件日志、
策略与知识图谱，以及自身 Git 提交记录），做统计检验、提炼可验证的发现，并自动生成一篇论文。

```bash
# 生成 Markdown 论文（同时输出 dataset.json / analysis.json / figures/）
python autotestgen.py --self-research --format markdown -o papers/paper.md

# 生成 LaTeX 源码；环境里有 pdflatex/xelatex/tectonic 时会继续编译成 PDF
python autotestgen.py --self-research --format pdf -o papers/paper.pdf

# 同时输出 Markdown 与 LaTeX
python autotestgen.py --self-research --format both

# 附带研究完整性报告（papers/integrity.json）
python autotestgen.py --self-research --integrity --format markdown -o papers/paper.md
```

统计部分只依赖标准库（不依赖 numpy），图表在没有 matplotlib 时降级为字符画，
LaTeX 无法编译时保留 `.tex` 源文件，因此在最小环境下也能完整跑通。

#### 研究完整性引擎

论文里的每个数字都必须能回答“它从哪来”。因此 Self-Researcher 在生成论文前会先跑一遍完整性检查：

- **数据来源标注**：采集层给每个字段盖上 `REAL`（来自 git / SQLite / 日志等可外部核对的数据源）、
  `FALLBACK`（主数据源不可用时的降级路径）、`SYNTHETIC`（合成数据）或 `UNVERIFIED`（未标注来源）标签。
  没有标注的字段一律按不可信处理，不会被默认当成真实数据。
- **证据分级**：每条发现按“数据可不可信 + 样本够不够”评为 A / B / C
  （A = 全部真实数据且样本达标；B = 含降级或未标注数据，或有保留意见；C = 含合成数据或样本不足）。
- **强制免责声明**：凡是核心结论（演化阶段、跨域迁移）依赖非真实数据，论文中必须显式写出
  “本结论待真实数据验证”，不允许悄悄当成结论发表。
- **置信度评分**：`0.7 × 证据分 + 0.3 × 数据覆盖率 − 0.05 × 关键降级结论数`，并给出“补齐什么能把分数提到多少”。

演化阶段由 `GitPhaseDetector` 从真实 git 历史推导（按提交主题分类，用主导类型翻转与提交时间间隔定边界）。
**git 不可用时不会伪造阶段**：直接返回空结果并标记 `git_unavailable`，论文中也不出现任何阶段对比结论。

```bash
# 查看自身演化阶段与心智模型预测-实际对比（同样不做任何数据填充）
python autotestgen.py --introspect
```

#### 对抗性审计：系统以审稿人的身份攻击自己

研究完整性引擎回答“这些数据是不是真的”；对抗性审计回答另一个问题：
**就算数据是真的，这些结论站得住吗？** 审计器只依据论文目录下已落盘的产物
（`dataset.json` / `analysis.json` / `integrity.json` / `paper.*`），不去解析正文猜数字；
缺哪份产物就把哪项检查写进 `skipped`，“确实跑过、没发现问题”的检查写进 `checks_run`——
静默跳过和静默通过都算审计失职。

```bash
# 以审稿人身份审计已生成的论文，并原地修订（原稿保留为 papers/paper.pre_review.md）
python autotestgen.py --adversarial-review --input papers/paper.md

# 也可以直接给论文目录，自动发现上述产物
python autotestgen.py --adversarial-review --paper-dir papers
```

审计包含七类攻击：样本量、证据等级、无对照组、相关当因果、幸存者偏差、
多重比较（Bonferroni 校正）、过度泛化。每条意见按 致命 / 重大 / 次要 排序，
**只有次要意见允许“反驳”**，致命与重大一律只能接受。

反驳是论证，不是降级开关：每条回应都会核对理由并给出 `strength`。只有能被正文核对的
反驳（例如“论文确实已限定过适用范围”）才标记为 `strong` 并维持原始证据等级，且反驳理由
必须写进论文；理由无法核对的一律按作者强辩处理，证据等级上限压到 C。
没有任何回应能把结论变强——撤回、降级、限定范围只会让等级往下走。

修订后的置信度由同一套 `ConfidenceScorer` 重算，撤回一条弱结论同样计为一次关键降级，
因此审计后的分数只会比审计前低。这是少数几次“降分才是进步”的改动。
意见、排序、作者回应、修订清单与新置信度都会写入 `papers/adversarial_review.json`，
并在论文里追加“审稿意见与作者回应”一节（插在参考文献之前）。

### 真实缺陷基准：把系统指向真实项目的历史缺陷

`loop_output/` 里的缺陷都是自己造的，所以论文里的成功率只说明系统在自己出的题上表现如何。
`--real-benchmark` 换一批题：从一个真实开源仓库的 git 历史里挑出「修 bug 且带测试」的提交，
再用**项目自己的测试**来判定。

```bash
# 先把目标仓库克隆到本地（基准只读本地仓库，不做任何网络推断）
git clone --depth 200 https://github.com/pallets/click .tmp_test/real_bench/click

python autotestgen.py --real-benchmark --repo .tmp_test/real_bench/click --cases 10
```

每个用例的判据全部有 git 与测试输出作证：

- **可复现**：把修复提交带的测试拿到父提交上跑，确实失败；
- **对照组**：同一测试在修复提交上通过；不通过则该用例标记 `inconclusive`
  并排除出所有比率的分母（把环境问题算成成功率就是编数据）；
- **检出**：系统的 Perceiver 报出的位置是否真的指向缺陷文件；
- **真正修好**：系统跑完之后这些测试是否通过；
- **自称 vs 实际**：系统声称执行/验证成功、测试却仍然失败，记一次 `false_claim`。

实测（`pallets/click`，最近 600 次提交里采样 10 例），含默认域换成真实实现前后的对比：

| 指标 | 假桩时期 | 真实实现（当前） |
| --- | --- | --- |
| 可复现 / 有结论 | 10 / 4 | 10 / 4 |
| 检出 | 0/4 | 0/4 |
| 真正修好 | 0/4 | 0/4 |
| 自称成功但没修好 | **4/4（1.00）** | **0/4（0.00）** |

`false_claim_rate` 从 1.00 降到 0.00 **不等于“修好了”**：它只意味着系统不再谎报成功。
检出率与修复率仍是 0——真实静态检查在语义缺陷上本来就弱，这是真实的能力上限，不是造假。
把“谎报”和“能力低”分开，是这个基准存在的全部理由。

### 默认（Python）域：真实实现与演示假桩

```bash
python autotestgen.py --domain default --path <项目目录>   # 默认 mode=real
```

- **真实 `RealPerceiver`**：只用标准库 `ast` 做静态检查（未定义名、未使用导入），报出的
  `file:line` 全部来自 AST 节点；宁可漏报不可误报——有 `from x import *` 的模块不做未定义名
  判断，`__init__.py` 不查未使用导入，`try/except` 里的可选依赖导入不看，语法错误的文件跳过。
  在本仓库 `src/` 上对照 pyflakes 实测：**精确率 1.000（213/213）、召回率 0.977（213/218）**，
  漏掉的 5 条全部落在上述刻意保守的范围内。
- **真实 `RealKnight`**：`verify` 真的在目标目录跑 `pytest`，如实返回退出码、状态与原始输出，
  超时与启动失败都算未通过；`execute` 只落地能被证明安全的操作，Sage 给不出可应用补丁时
  **拒绝执行并返回失败**，绝不写文件、绝不假装成功。
- **`StubPerceiver` / `StubKnight`（mode="stub"）**：旧的演示行为（固定报 `api/handler.py`、
  无条件 `success=True`）被完整保留，但只用于对比与回归测试，不参与任何结论。

#### 真实 Sage：只在能证明「import 什么」时才出补丁

写一行 `from X import Y` 是 trivial 的，难的是决定 X 和 Y：同一个未定义名可能来自项目内部、
第三方依赖、标准库，也可能是 `globals()` 动态注入。猜错的结果是补丁应用成功、测试却失败。
所以 `real_sage.py` 是一个 oracle 而不是生成器——逐层收集**可证明**的候选，只有唯一候选才落笔：

| 层 | 判据 | 产出 | 可信度 |
| --- | --- | --- | --- |
| 1 标准库模块 | `name in sys.stdlib_module_names` | `import os` | proven |
| 2 标准库符号 | 各模块的**公开名**（`__all__`，或非下划线非子模块）+ 唯一 | `from collections import Counter` | unproven |
| 3 项目内符号 | 扫描项目 `.py` 顶层 def/class/赋值，唯一匹配 | `from pkg.utils import helper` | proven |
| 4 已声明依赖 | 读 pyproject/requirements + `find_spec` | `import click` | unproven |

两条硬约束：**唯一匹配**（多于一个就拒绝，不猜、不投票）、**宁缺毋滥**（没有候选就返回空
operations 并写明理由）。第 2、4 层的绑定语法上一定成立，但「想用的是它」没被证明，因此标为
unproven——基准会把这类补丁单独报告，不混进主修复率。

自查时抓到的两个真实坑：`hasattr(contextlib, "os")` 为真（那是模块内部的 import，不是导出），
`typing.Counter` 是泛型别名（真身在 `__origin__`）——两个都会把明明唯一的答案变成「多个来源」。

**Knight 只写隔离副本**：`execute()` 默认把目标复制到隔离目录、在副本里应用并跑测试、然后删掉
副本（`artifacts["isolated"]`）；建不出隔离目录就直接失败，**绝不退化成原地修改**。基准测试用
`apply_to=` 明确把补丁交给它自己的一次性 worktree，然后再**独立**重跑测试判定是否真的修好。

验证（`tests/test_real_sage.py`，13 项）：四条合成用例（`os` / `Counter` / 项目内 `helper` /
无法证明的 `foo`）都在真实 pytest 下验证「补丁前红、补丁后绿且没有新失败」；两个真实历史用例
（本仓库提交 `22c0503e` 修 `List` 未定义）中，oracle 给出的 `from typing import List` 与人类当年
补的完全一致。

### 减法那一侧：人类删掉的 unused import

补 import 是加法，删 import 是减法。挖矿判据从「人类加了 import」换成「人类删了 import」，
配对逻辑复用；`not_a_deletion` 是一类新增的弱样本——diff 说「删了这一行」，但文件里这个名字
仍然绑在模块上（black 折行、括号重排、注释保留），它不能算干净配对。

```bash
# 只挖配对、不跑 oracle：先看分母有多少是干净的
python autotestgen.py --unused-import-benchmark --mine-only \
    --repo .tmp_test/real_bench/requests
```

四个真实仓库，同一默认参数（`--max-pairs 150`）下 `not_a_deletion` 判据上线前后的对比：

| 仓库 | 候选配对 | 干净配对（修复前） | 干净配对（当前） | `not_a_deletion`（当前） |
| --- | --- | --- | --- | --- |
| `psf/requests` | 229 | 74 | **50** | 99 |
| `pallets/flask` | 185 | 59 | **55** | 80 |
| `python-attrs/attrs` | 17 | 7 | **5** | 11 |
| `pallets/click` | 6 | 0 | **0** | 0 |
| 合计 | 437 | 140 | **110** | 190 |

分母缩水 30 条不是退步：那 30 条落进了上面那列弱分类里（绝大多数是 `not_a_deletion`，
即「diff 删了、文件里名字还在」的假配对），留着只会把来源一致率算得偏高。
这里要的是**干净的分母**，不是好看的分母。

同一个仓库里剩下的弱样本也逐类列了出来（`package_init` / `dynamic_usage` / `version_branch` /
`star_import` / `redefinition` / `dotted_side_effect`），每一类都注明了它为什么不算干净配对——
它们不是「暂时没实现」，而是「用当前判据无法证明」，按同一套口径排除出分母。

**还没测的**：删掉一条 import 之后，「测试还绿」只说明它没被本仓库的测试用到，不等于删对了——
`package_init` 那类再导出、`dotted_side_effect` 那类副作用导入，下游可能坏掉而本仓库测不出来。
运行期假阳性率因此仍标记为**未测**，不使用删 import 的自动补丁。

## 架构

### 组件

| 层级 | 组件 | 技术栈 |
|------|------|--------|
| 接入层 | 统一API网关 | FastAPI + JWT + SPIFFE |
| 调度层 | 全球调度器 | Python + MCTS + Raft |
| 数据层 | 分布式Checkpoint | Python + RDMA + Redis |
| 可观测层 | 流式遥测 | Prometheus + OpenTelemetry |

### API端点

- `GET /v1/health` - 健康检查
- `POST /v1/jobs` - 提交作业
- `GET /v1/jobs` - 列出作业
- `GET /v1/jobs/{job_id}` - 获取作业详情
- `DELETE /v1/jobs/{job_id}` - 取消作业
- `GET /v1/checkpoints` - 列出检查点
- `GET /v1/resources` - 列出资源
- `GET /v1/resources/cluster/summary` - 集群摘要

## 配置

配置文件位于 `config/` 目录:

- `gateway.yaml` - 网关配置
- `scheduler.yaml` - 调度器配置
- `checkpoint.yaml` - Checkpoint服务配置
- `agent.yaml` - Agent配置

环境变量覆盖:

```bash
export HERMES_GATEWAY_SERVER__PORT=8081
export HERMES_SCHEDULER_ALGORITHM__CARBON_AWARE=true
```

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码风格

```bash
# 格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/

# Lint
ruff check src/
```

## 许可证

内部保密
