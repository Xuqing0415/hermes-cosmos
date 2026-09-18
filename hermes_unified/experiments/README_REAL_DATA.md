# 真实联邦学习数据集对接

让 Hermes 在真实联邦数据集上运行，而不是模拟数据。

## 支持的数据集

| 数据集 | 类型 | 客户端数 | 类别数 | 说明 |
|--------|------|----------|--------|------|
| **FEMNIST** | 图像分类 | 3500+ | 62 | EMNIST 按书写者划分 |
| **Shakespeare** | 语言模型 | 1146 | 80 | 戏剧文本按角色划分 |
| **StackOverflow** | 标签预测 | 500k+ | 10000 | StackOverflow 问答按用户划分 |

## 快速开始

### 运行单个实验

```bash
# FEMNIST + FedAvg
python -m hermes_unified.experiments.real_data_benchmark \
    --dataset femnist \
    --algorithm fedavg \
    --clients 20 \
    --rounds 10

# Shakespeare + Meta-FL 自动选择
python -m hermes_unified.experiments.real_data_benchmark \
    --dataset shakespeare \
    --clients 15 \
    --rounds 15 \
    --auto
```

### 对比多个算法

```bash
# 对比所有算法在 FEMNIST 上的表现
python -m hermes_unified.experiments.real_data_benchmark \
    --dataset femnist \
    --clients 20 \
    --rounds 10 \
    --compare
```

## 使用指南

### 数据集加载

```python
from hermes_unified.data import load_real_federated_data

# 加载 FEMNIST
client_data = load_real_federated_data('femnist', num_clients=20)

# 加载 Shakespeare
client_data = load_real_federated_data('shakespeare', num_clients=10)

# 客户端数据格式
for i, ((train_x, train_y), (test_x, test_y)) in enumerate(client_data):
    print(f"Client {i}:")
    print(f"  Train: {train_x.shape}, {train_y.shape}")
    print(f"  Test: {test_x.shape}, {test_y.shape}")
```

### 创建模型

```python
from hermes_unified.data import create_model_for_dataset

# 为数据集创建对应模型
model = create_model_for_dataset('femnist')
model = create_model_for_dataset('shakespeare')

# 手动创建模型
from hermes_unified.data import SimpleCNNForFEMNIST, SimpleRNNForShakespeare
cnn_model = SimpleCNNForFEMNIST(input_shape=(28, 28, 1), num_classes=62)
rnn_model = SimpleRNNForShakespeare(vocab_size=80)
```

### 运行完整流程

```python
from hermes_unified.experiments.real_data_benchmark import (
    run_real_data_benchmark,
    compare_algorithms
)

# 运行单个实验
results = run_real_data_benchmark(
    dataset_name='femnist',
    algorithm='fedavg',
    num_clients=20,
    num_rounds=10,
    auto_mode=False
)

# 对比多个算法
comparison = compare_algorithms(
    dataset_name='femnist',
    num_clients=20,
    num_rounds=10
)
```

## 数据集详情

### FEMNIST

- **输入**: 28x28 灰度图像
- **输出**: 62 个类别 (0-9, a-z, A-Z)
- **特点**: 高度 Non-IID（每个用户书写风格不同）
- **适合**: 个性化算法 (Ditto, FedRep) 测试

### Shakespeare

- **输入**: 40 个字符序列
- **输出**: 下一个字符 (80 类别)
- **特点**: 每个客户端对应一个角色的台词
- **适合**: 语言模型联邦训练

### StackOverflow (待实现)

- **输入**: 100 个词序列
- **输出**: 标签
- **特点**: 大规模数据，高稀疏度
- **适合**: 扩展性测试

## 输出结果

实验结果保存在 `benchmark_results/` 目录：

```
benchmark_results/
├── femnist_fedavg.json         # 单个实验
├── femnist_ditto.json
└── ...
```

JSON 格式：
```json
{
    "dataset": "femnist",
    "algorithm": "fedavg",
    "num_clients": 20,
    "num_rounds": 10,
    "total_time": 123.45,
    "final_accuracy": 0.82,
    "communication_mb": 15.2,
    "history": [...每轮详细数据...]
}
```

## 预期结果

| 数据集 | 算法 | 准确率 | 通信量 |
|--------|------|--------|--------|
| FEMNIST | FedAvg | 78% | 25MB |
| FEMNIST | Ditto | 82% | 28MB |
| FEMNIST | FedRep | 80% | 12MB |
| Shakespeare | FedAvg | 65 perplexity | 30MB |
| Shakespeare | FedRep | 62 perplexity | 15MB |

## 下一步

1. **集成真实 LEAF 数据集**：使用 `leaf` 工具下载真实数据
2. **添加更多模型**：ResNet, LSTM, 等
3. **真实硬件部署**：在边缘设备上运行
4. **超参数调优**：在真实数据上运行 Meta-FL
5. **详细报告**：分析 Non-IID 对算法的影响

## 注意事项

当前实现使用**合成数据集**，结构与真实数据一致，但数据值是生成的。

要使用真实数据，请：
1. 从 [LEAF](https://github.com/TalwalkarLab/leaf) 下载
2. 或使用 `tensorflow_federated` 库
3. 修改 `data/loaders.py` 中的加载函数

## 参考文献

- LEAF 数据集: https://leaf.cmu.edu/
- TensorFlow Federated: https://www.tensorflow.org/federated
