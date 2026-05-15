# Meta-FL Controller

元联邦学习控制器，自动选择最适合当前任务的联邦学习算法。

## 功能特性

### 1. 元特征提取
- **统计特征**: 客户端数量、样本数、类别数、特征维度
- **分布特征**: Non-IID程度（基尼系数、Jensen-Shannon散度、熵）
- **资源特征**: 带宽、计算能力、网络延迟

### 2. 性能预测
- 基于历史实验数据训练预测模型
- 预测每种算法的准确率、收敛轮数、通信量
- 支持多种算法: FedAvg、Ditto、FedRep、Krum、Trimmed Mean

### 3. 智能推荐
- 多目标优化（准确率、通信量、收敛速度）
- Pareto前沿算法选择
- 置信度评估

### 4. 解释生成
- 自然语言解释推荐理由
- 特征重要性分析
- 任务特性分析

## 快速开始

### 命令行模式

```bash
# 运行元学习推荐
python -m hermes_unified.meta_fl.run_meta --mode cli

# 指定任务参数
python -m hermes_unified.meta_fl.run_meta --mode cli --clients 20 --non-iid 0.7
```

### 服务器模式

```bash
# 启动推荐API服务器
python -m hermes_unified.meta_fl.run_meta --mode server --port 8503
```

### API 使用

```bash
# 获取推荐
curl -X POST http://localhost:8503/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"num_clients": 20, "non_iid_level": 0.7}'

# 分析任务
curl -X POST http://localhost:8503/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"num_clients": 15, "non_iid_level": 0.5}'
```

### Python API

```python
from hermes_unified.meta_fl import TaskFeatureExtractor, AlgorithmRecommender

# 提取任务特征
extractor = TaskFeatureExtractor()
features = extractor.extract(client_data, resources)

# 获取推荐
recommender = AlgorithmRecommender()
recommendation = recommender.recommend(features)

print(f"推荐算法: {recommendation['algorithm_name']}")
print(f"置信度: {recommendation['confidence']}%")
print(f"解释: {recommendation['explanation']}")
```

## 核心组件

### TaskFeatureExtractor

```python
extractor = TaskFeatureExtractor()

# 提取特征
features = extractor.extract(client_data, resources)

# 特征包括:
# - num_clients: 客户端数量
# - non_iid_gini: Non-IID程度（基尼系数）
# - non_iid_jsd: Jensen-Shannon散度
# - avg_bandwidth_mbps: 平均带宽
# - avg_compute_score: 平均计算能力
```

### PerformancePredictor

```python
predictor = PerformancePredictor()

# 训练预测模型（使用合成数据或真实实验数据）
X, y, metadata = predictor.generate_synthetic_dataset(n_samples=1000)
predictor.train(X, y, metadata['feature_names'])

# 预测性能
predictions = predictor.predict(features)
# predictions['fedavg'] = {'accuracy': 0.85, 'rounds': 45, 'communication': 1.0}
```

### AlgorithmRecommender

```python
recommender = AlgorithmRecommender()

# 获取推荐
recommendation = recommender.recommend(features)

# 分析任务
analysis = recommender.analyze_task(features)

# 获取Pareto前沿
frontier = recommender.get_pareto_frontier(features)

# 比较算法
comparison = recommender.compare_algorithms(features)
```

## 推荐算法

| 算法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **FedAvg** | IID数据 | 简单、低开销 | Non-IID表现差 |
| **Ditto** | 高Non-IID | 个性化、公平 | 通信量大 |
| **FedRep** | 资源受限 | 低通信、异构模型 | 个性化弱 |
| **Krum** | 存在攻击 | 抗攻击 | 计算开销大 |
| **Trimmed Mean** | 中等攻击 | 简单鲁棒 | 信息损失 |

## Non-IID 评估

基于以下指标评估数据分布的Non-IID程度：

| 指标 | 说明 | 范围 |
|------|------|------|
| **Gini系数** | 标签分布不均匀程度 | 0-1（越高越Non-IID） |
| **JSD** | 客户端间分布差异 | 0-1 |
| **熵** | 全局标签分布的不确定性 | 越高越均匀 |

**解释**:
- Non-IID < 0.3: 低Non-IID，推荐FedAvg
- 0.3 < Non-IID < 0.6: 中等Non-IID，推荐Ditto或FedRep
- Non-IID > 0.6: 高Non-IID，强烈推荐Ditto

## 与现有模块集成

### 在个性化FL中使用

```python
# run_pfl.py
if args.auto_mode:
    from hermes_unified.meta_fl import AlgorithmRecommender
    
    features = extract_features(client_data)
    recommender = AlgorithmRecommender()
    recommendation = recommender.recommend(features)
    
    print(f"自动选择算法: {recommendation['algorithm_name']}")
    print(f"理由: {recommendation['explanation']}")
    
    # 使用推荐的算法运行
    run_algorithm(recommendation['recommended_algorithm'], client_data)
```

## 扩展支持新算法

1. 在 `recommender.py` 的 `algorithm_descriptions` 中添加算法描述
2. 在 `performance_predictor.py` 中添加算法的性能预测逻辑
3. 更新推荐逻辑中的算法比较

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                   Meta-FL Controller                    │
│  ┌──────────────────┐  ┌──────────────────┐          │
│  │ FeatureExtractor │→ │ PerformancePred. │→          │
│  │   (元特征提取)    │  │   (性能预测)      │          │
│  └──────────────────┘  └────────┬─────────┘          │
│                                 │                     │
│                                 ▼                     │
│  ┌──────────────────────────────────────────┐          │
│  │           AlgorithmRecommender           │          │
│  │  (推荐引擎: 多目标优化 + 解释生成)        │          │
│  └──────────────────────────────────────────┘          │
│                          │                             │
│                          ▼                             │
│              ┌───────────────────┐                     │
│              │    Web Dashboard  │                     │
│              └───────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/recommend` | POST | 获取算法推荐 |
| `/api/analyze` | GET | 分析任务特性 |
| `/` | GET | API说明 |

## 依赖

- numpy
- scipy
- scikit-learn
- flask
- flask-cors

## 许可

MIT License
