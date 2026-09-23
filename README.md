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

实测（`pallets/click`，最近 600 次提交里采样 10 例）：可复现 10/10、有结论 4/10
（其余 6 例项目自身测试在本机环境跑不过，已排除）、**系统检出 0/4、真正修好 0/4、
自称成功但没修好 4/4**。

结论：在真实缺陷上系统目前检出率与修复率都是 0，却 100% 自称成功。原因是默认域的
Perceiver / Sage / Knight 仍是硬编码假桩（固定报 `api/handler.py`、Knight 无条件返回成功
并打印固定的测试结果）。基准把这个差距变成了可复现的数字，而不是靠读代码猜。

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
