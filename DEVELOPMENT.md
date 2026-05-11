# Hermes Cosmos 开发指南 - 阶段一：健康检查与验证

## 快速开始（第一天）

### 1. 环境准备

```bash
# 克隆项目（如果还没有）
cd hermes-cosmos

# 给脚本添加执行权限
chmod +x start.sh dev.sh

# 使用快速开发脚本（推荐）
./dev.sh help
```

### 2. 运行健康检查（阶段一：步骤1-2）

#### 方式一：使用 `dev.sh`（推荐，不需要Docker）

```bash
# 运行单元测试
./dev.sh unit

# 运行端到端测试
./dev.sh e2e

# 启动调度器进行手动测试
./dev.sh scheduler

# 在另一个终端提交测试作业
./dev.sh test-job
```

#### 方式二：使用 `start.sh`（需要Docker Compose）

```bash
# 启动所有服务
./start.sh

# 查看健康检查报告
cat HEALTH_CHECK_REPORT.md

# 查看实时日志
./start.sh logs

# 停止服务
./start.sh stop
```

### 3. 验证关键功能（第一天）

#### 验证一：作业提交与调度

```python
# 测试脚本
import httpx

response = httpx.post(
    "http://localhost:50051/jobs",
    json={
        "name": "first-test-job",
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "priority": "HIGH",
        "requirements": {
            "gpu_count": 8,
            "gpu_type": "nvidia-h100",
            "memory_gb": 128,
            "cpu_cores": 32
        },
        "image": "pytorch/pytorch:latest",
        "command": "python train.py"
    }
)

print(response.json())
```

#### 验证二：故障恢复测试（SLA 验证）

```bash
# 运行故障恢复测试
pytest tests/test_e2e.py::TestJobFailover -v -s
```

#### 验证三：集群状态

```bash
# 查看调度器状态
curl http://localhost:50051/status

# 查看集群摘要
curl http://localhost:50051/cluster/summary
```

---

## 项目结构说明

```
hermes-cosmos/
├── src/hermes/
│   ├── core/                    # 核心模块
│   │   ├── config.py           # 配置模型
│   │   ├── models.py           # 数据模型
│   │   ├── exceptions.py       # 异常定义
│   │   └── store.py            # 内存存储（开发用）
│   ├── gateway/                # API网关
│   ├── scheduler/              # 调度器
│   ├── checkpoint/             # Checkpoint服务
│   └── agent/                  # 节点Agent
├── tests/
│   ├── test_e2e.py             # 端到端测试（重要）
│   ├── test_scheduler.py       # 调度器测试
│   ├── test_checkpoint.py      # Checkpoint测试
│   └── test_core.py            # 核心模块测试
├── config/                     # 配置文件
├── docker/                     # Docker配置
├── deploy/                     # 部署配置
├── start.sh                    # 快速启动
└── dev.sh                      # 开发工具
```

---

## 验证清单（第一天）

- [ ] **环境检查**
  - [ ] Python 3.11+ 已安装
  - [ ] 虚拟环境已创建并激活
  - [ ] 依赖已安装 (`pip install -e ".[dev]"`)

- [ ] **单元测试**
  - [ ] 核心模型测试通过 (`test_core.py`)
  - [ ] 调度器测试通过 (`test_scheduler.py`)
  - [ ] Checkpoint测试通过 (`test_checkpoint.py`)

- [ ] **端到端测试**
  - [ ] 作业提交测试通过
  - [ ] 调度延迟测试通过 (< 500ms)
  - [ ] 故障恢复测试通过 (< 5s)

- [ ] **功能验证**
  - [ ] 调度器启动成功
  - [ ] 健康检查端点返回200
  - [ ] 集群资源可查询

---

## 常见问题

### Q: Docker Compose启动失败怎么办？

A: 先使用 `dev.sh` 进行本地开发测试，不依赖Docker。

### Q: 找不到模块？

A: 确保在虚拟环境中安装了项目：
```bash
source venv/bin/activate
pip install -e ".[dev]"
```

### Q: 端口被占用？

A: 修改相应的配置文件，或使用：
```bash
lsof -ti :50051 | xargs kill -9  # 杀占用端口的进程
```

---

## 下一步

完成第一天验证后，继续进行：

1. **阶段一剩余步骤**：
   - 依赖连通性检查
   - 代码静态分析
   - 完整的健康检查报告

2. **阶段二**：集成测试与端到端场景
3. **阶段三**：混沌工程与压力测试

详见 `PROJECT_PLAN.md`（如果有的话）
