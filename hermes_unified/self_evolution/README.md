# Self-Evolving Federated Learning

自进化联邦学习 - Hermes 自动运行实验、分析结果、持续改进。

## 核心理念

Hermes 不再需要人工选择算法或调参。它自动探索参数空间，不断进化，形成无限循环的自进化体。

## 功能特性

### 1. 自动化实验循环
- 随机探索参数空间
- UCB算法平衡探索与利用
- 在线学习更新预测模型

### 2. 智能配置选择
- Upper Confidence Bound (UCB) 选择策略
- 贝叶斯优化（待实现）
- epsilon-greedy 随机探索

### 3. 自动部署
- 自动更新最佳配置
- GitOps 集成（模拟）
- 部署配置自动保存

### 4. 进化可视化
- 进化树视图
- 性能趋势图
- 里程碑高亮
- Web 仪表盘

## 快速开始

### 运行自进化

```bash
# 运行 10 代，最多 1 小时
python -m hermes_unified.self_evolution.run_self_evolution \
    --generations 10 \
    --hours 1.0
```

### 只运行服务器（查看已有数据）

```bash
# 直接启动仪表盘（需要先运行过自进化）
python -m hermes_unified.self_evolution.run_self_evolution \
    --generations 0 \
    --hours 0.1 \
    --port 8504
```

### Python API

```python
from hermes_unified.self_evolution import SelfEvolvingFederatedLearning

# 创建自进化系统
sef = SelfEvolvingFederatedLearning(
    db_path="self_evolution_db.json",
    output_dir="results"
)

# 运行进化
sef.evolve(num_generations=20, max_time_hours=2.0)

# 获取进化报告
report = sef.generate_evolution_report()
print(f"Best Accuracy: {report['summary']['best_accuracy']}")

# 获取进化树
tree = sef.get_evolution_tree()
```

## 参数配置

```python
config = {
    'num_clients': 20,          # 客户端数量 (5-50)
    'non_iid_level': 0.6,       # Non-IID 程度 (0.1-0.95)
    'algorithm': 'ditto',       # 算法选择
    'learning_rate': 0.05,      # 学习率
    'num_rounds': 30,           # 训练轮数
    'bandwidth': 50.0,          # 带宽 Mbps
    'compute_power': 1.0        # 计算能力
}
```

## 探索策略

### UCB (Upper Confidence Bound)

```
UCB_score = mean_accuracy + c * sqrt(ln(n) / n_i)

where:
- mean_accuracy: 平均准确率
- c: 探索常数 (默认 1.0)
- n: 总实验次数
- n_i: 当前配置尝试次数
```

### epsilon-greedy

- `epsilon=0.1`: 10% 概率随机探索
- 90% 概率选择最佳配置

## 输出文件

```
self_evolution_results/
├── evolution_progress.json     # 进度信息
├── best_deployment.json       # 最佳部署配置
└── self_evolution_db.json      # 完整实验数据库
```

## 仪表盘

访问 `http://localhost:8504` 查看：

- 实验总数、代数
- 最佳准确率
- 里程碑数量
- 进化树可视化
- 算法性能对比
- 准确率趋势

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                Self-Evolving FL System                    │
│  ┌───────────────────────────────────────────────────┐   │
│  │ Configuration Selector (UCB/epsilon-greedy)         │   │
│  └──────────────────────┬────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼────────────────────────────┐   │
│  │         Experiment Runner (simulated)              │   │
│  └──────────────────────┬────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼────────────────────────────┐   │
│  │ Online Learning (Performance Predictor update)    │   │
│  └──────────────────────┬────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼────────────────────────────┐   │
│  │ Milestone Check & Auto-Deploy                     │   │
│  └──────────────────────┬────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼────────────────────────────┐   │
│  │ Database & Visualization                          │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 扩展点

### 添加新探索策略

```python
def select_config_custom(self) -> Dict:
    # 实现自定义选择逻辑
    pass
```

### 添加真实实验运行器

替换 `run_single_experiment` 中的模拟部分。

### 集成真实部署系统

实现 `auto_deploy_best_config` 中的实际 GitOps/ArgoCD 集成。

## 下一步

- 集成真实 FL 运行器而非模拟
- 实现贝叶斯优化
- 添加超参数优化
- 集成 GitOps/ArgoCD 真实部署
- 更多可视化（SHAP 解释等）

## 依赖

- numpy
- flask
- flask-cors
- plotly (可选)

## 许可

MIT License
