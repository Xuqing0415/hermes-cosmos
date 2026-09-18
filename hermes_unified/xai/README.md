# FedXAI: Federated Explainable AI

联邦可解释性仪表盘，为联邦学习模型提供可解释性分析。

## 功能特性

### 1. 可解释性方法
- **KernelSHAP**: 基于 Shapley 值的模型无关解释
- **LIME**: 局部可解释模型无关解释
- **Permutation Importance**: 基于特征排列的重要性评估

### 2. 客户端本地解释
- 每个客户端在本地生成模型解释
- 只共享聚合后的特征重要性向量（不泄露原始数据）
- 支持多种解释方法切换

### 3. 服务器端聚合
- 全局平均重要性
- 跨客户端方差（发现特征使用分歧）
- 按类别重要性聚合

### 4. 可视化仪表盘
- 特征重要性条状图
- 客户端-特征热图
- 时间序列曲线
- 跨客户端方差分析

### 5. 异常检测
- 基于特征重要性的异常检测
- 当客户端重要性模式突然改变时触发告警
- 可用于检测数据分布漂移或恶意攻击

## 快速开始

### 运行 XAI 仪表盘

```bash
python -m hermes_unified.xai.dashboard
```

访问 http://localhost:8502 查看仪表盘。

### 运行完整实验

```bash
python -m hermes_unified.xai.run_xai
```

### 在现有联邦学习中集成 XAI

```python
from hermes_unified.xai import FederatedXAIClient, FederatedXAIServer

# 在服务器端
xai_server = FederatedXAIServer(feature_names=['feature_0', 'feature_1', ...])

# 在客户端
xai_client = FederatedXAIClient(
    client_id='client_1',
    model=trained_model,
    feature_names=['feature_0', 'feature_1', ...]
)

# 注册客户端
xai_server.register_client(xai_client)

# 每轮结束后生成解释
for round in range(num_rounds):
    # ... 联邦训练代码 ...
    
    # 生成并收集 XAI 解释
    xai_client.generate_local_explanation(X_test)
    
    # 服务器聚合
    result = xai_server.run_xai_round()
    
    # 检测异常
    if result['anomalies']:
        print(f"检测到 {len(result['anomalies'])} 个异常客户端")
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    FedXAI Server                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Aggregator  │  │   Anomaly    │  │  Dashboard   │     │
│  │              │  │  Detector    │  │   Server     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
│         │                 │                                 │
└─────────┼─────────────────┼─────────────────────────────────┘
          │                 │
          ▼                 ▼
    ┌───────────┐     ┌───────────┐
    │  Client   │     │  Client   │
    │  1 XAI    │     │  2 XAI    │
    └─────┬─────┘     └─────┬─────┘
          │                 │
          ▼                 ▼
    ┌───────────┐     ┌───────────┐
    │  Local    │     │  Local    │
    │  Model    │     │  Model    │
    └───────────┘     └───────────┘
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | XAI 仪表盘 |
| `/api/xai/data` | GET | 获取所有 XAI 数据 |
| `/api/xai/update` | POST | 更新 XAI 数据 |
| `/api/xai/anomalies` | GET | 获取异常告警 |

## 异常检测

异常检测基于特征重要性偏差：

```python
# 当重要性偏差超过阈值时触发告警
is_anomalous, deviation = client.is_anomalous(threshold=0.3)

# 告警包含：
# - client_id: 异常客户端 ID
# - deviation: 偏差分数
# - severity: 严重程度 (high/medium/low)
# - timestamp: 检测时间
```

## 与 Prometheus 集成

添加以下指标到 Prometheus 配置：

```yaml
- job_name: 'hermes-xai'
  static_configs:
    - targets: ['xai-server:8502']
  metrics_path: '/api/xai/metrics'
```

## 应用场景

### 医疗联邦学习
- 解释模型对患者诊断的决策依据
- 确保符合 HIPAA 等隐私法规

### 金融风控
- 解释信用评分模型的特征贡献
- 检测模型偏差和歧视

### 跨组织协作
- 在不共享原始数据的情况下验证模型
- 发现数据分布差异

## 依赖

- numpy
- flask
- flask-cors
- plotly (可选，用于仪表盘图表)

## 许可

MIT License
