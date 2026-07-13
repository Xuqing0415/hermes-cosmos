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
