# 

 Hermes 

## 

|  |  |  |  |  |
|--------|------|----------|--------|------|
| **FEMNIST** |  | 3500+ | 62 | EMNIST  |
| **Shakespeare** |  | 1146 | 80 |  |
| **StackOverflow** |  | 500k+ | 10000 | StackOverflow  |

## 

### 

```bash
# FEMNIST + FedAvg
python -m hermes_unified.experiments.real_data_benchmark \
    --dataset femnist \
    --algorithm fedavg \
    --clients 20 \
    --rounds 10

# Shakespeare + Meta-FL 
python -m hermes_unified.experiments.real_data_benchmark \
    --dataset shakespeare \
    --clients 15 \
    --rounds 15 \
    --auto
```

### 

```bash
#  FEMNIST 
python -m hermes_unified.experiments.real_data_benchmark \
    --dataset femnist \
    --clients 20 \
    --rounds 10 \
    --compare
```

## 

### 

```python
from hermes_unified.data import load_real_federated_data

#  FEMNIST
client_data = load_real_federated_data('femnist', num_clients=20)

#  Shakespeare
client_data = load_real_federated_data('shakespeare', num_clients=10)

# 
for i, ((train_x, train_y), (test_x, test_y)) in enumerate(client_data):
    print(f"Client {i}:")
    print(f"  Train: {train_x.shape}, {train_y.shape}")
    print(f"  Test: {test_x.shape}, {test_y.shape}")
```

### 

```python
from hermes_unified.data import create_model_for_dataset

# 
model = create_model_for_dataset('femnist')
model = create_model_for_dataset('shakespeare')

# 
from hermes_unified.data import SimpleCNNForFEMNIST, SimpleRNNForShakespeare
cnn_model = SimpleCNNForFEMNIST(input_shape=(28, 28, 1), num_classes=62)
rnn_model = SimpleRNNForShakespeare(vocab_size=80)
```

### 

```python
from hermes_unified.experiments.real_data_benchmark import (
    run_real_data_benchmark,
    compare_algorithms
)

# 
results = run_real_data_benchmark(
    dataset_name='femnist',
    algorithm='fedavg',
    num_clients=20,
    num_rounds=10,
    auto_mode=False
)

# 
comparison = compare_algorithms(
    dataset_name='femnist',
    num_clients=20,
    num_rounds=10
)
```

## 

### FEMNIST

- ****: 28x28 
- ****: 62  (0-9, a-z, A-Z)
- ****:  Non-IID
- ****:  (Ditto, FedRep) 

### Shakespeare

- ****: 40 
- ****:  (80 )
- ****: 
- ****: 

### StackOverflow ()

- ****: 100 
- ****: 
- ****: 
- ****: 

## 

 `benchmark_results/` 

```
benchmark_results/
 femnist_fedavg.json         # 
 femnist_ditto.json
 ...
```

JSON 
```json
{
    "dataset": "femnist",
    "algorithm": "fedavg",
    "num_clients": 20,
    "num_rounds": 10,
    "total_time": 123.45,
    "final_accuracy": 0.82,
    "communication_mb": 15.2,
    "history": [......]
}
```

## 

|  |  |  |  |
|--------|------|--------|--------|
| FEMNIST | FedAvg | 78% | 25MB |
| FEMNIST | Ditto | 82% | 28MB |
| FEMNIST | FedRep | 80% | 12MB |
| Shakespeare | FedAvg | 65 perplexity | 30MB |
| Shakespeare | FedRep | 62 perplexity | 15MB |

## 

1. ** LEAF ** `leaf` 
2. ****ResNet, LSTM, 
3. ****
4. **** Meta-FL
5. **** Non-IID 

## 

****


1.  [LEAF](https://github.com/TalwalkarLab/leaf) 
2.  `tensorflow_federated` 
3.  `data/loaders.py` 

## 

- LEAF : https://leaf.cmu.edu/
- TensorFlow Federated: https://www.tensorflow.org/federated
