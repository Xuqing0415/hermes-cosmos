# 安全多方计算聚合 (MPC)

在 Hermes 联邦学习框架中集成**安全聚合**机制，确保服务器无法窥视单个客户端的梯度更新。

## 功能特性

### 两种安全方法

| 方法 | 安全性 | 计算开销 | 通信开销 | 说明 |
|------|--------|----------|----------|------|
| **Shamir 秘密共享** | 防 1/3 恶意 | 低 | 低 | 实用，易于部署 |
| **Paillier 同态加密** | 更高 | 高 | 高 | 理论上更强 |

### 核心特性

- **隐私保护**: 服务器只能看到聚合结果，无法获取单个客户端梯度
- **正确性保证**: 聚合结果与明文聚合一致
- **统一接口**: 通过 `SecureAggregator` 统一调用两种方法
- **性能监控**: 记录加密/解密/聚合耗时

## 快速开始

### 运行 Demo

```bash
# 运行安全聚合测试
python test_secure_aggregation.py
```

### 基本用法

```python
from hermes_unified.secure_aggregation import SecureAggregator

# 创建安全聚合器
aggregator = SecureAggregator(
    method='shamir',  # 或 'paillier'
    num_clients=5,
    threshold=3
)

# 客户端加密更新
encrypted_updates = {}
for client_id in range(5):
    update = np.random.randn(100)  # 梯度更新
    encrypted = aggregator.client_encrypt_update(client_id, update)
    encrypted_updates[client_id] = encrypted

# 服务器聚合加密数据
aggregated_encrypted = aggregator.aggregate_encrypted(encrypted_updates)

# 解密得到结果
result = aggregator.decrypt_result(aggregated_encrypted)

# 获取性能统计
stats = aggregator.get_stats()
```

## Shamir 秘密共享

```python
from hermes_unified.secure_aggregation import ShamirSecretSharing

shamir = ShamirSecretSharing(threshold=3)

# 分享秘密
secret = 123.45
shares = shamir.share_scalar(secret, num_shares=5)

# 重构秘密（需要至少 threshold 份）
reconstructed = shamir.reconstruct_scalar(shares[:3])
print(reconstructed)  # 123.45

# 数组支持
arr = np.random.randn(5, 5)
share_arrays = shamir.share_array(arr, 5)
reconstructed_arr = shamir.reconstruct_array(share_arrays[:3])
```

## Paillier 同态加密

```python
from hermes_unified.secure_aggregation import create_demo_paillier

paillier, paillier_arr = create_demo_paillier()

# 加密/解密
ciphertext = paillier.encrypt(42)
plaintext = paillier.decrypt(ciphertext)

# 同态加法
c1 = paillier.encrypt(100)
c2 = paillier.encrypt(200)
c_sum = paillier.add_encrypted(c1, c2)
print(paillier.decrypt(c_sum))  # 300

# 数组支持
arr = np.array([1.5, 2.5, 3.5])
enc_arr = paillier_arr.encrypt_array(arr)
dec_arr = paillier_arr.decrypt_array(enc_arr)
```

## 集成到联邦学习流程

### 客户端端

```python
from hermes_unified.secure_aggregation import SecureFLClient

# 创建客户端
secure_aggregator = SecureAggregator('shamir', num_clients=10)
client = SecureFLClient(client_id=0, model=my_model, secure_aggregator=secure_aggregator)

# 训练并加密更新
encrypted_update = client.train_and_encrypt(train_data, epochs=1)
send_to_server(encrypted_update)
```

### 服务器端

```python
# 聚合加密更新
aggregated_encrypted = secure_aggregator.aggregate_encrypted(encrypted_updates, client_weights)
aggregated_update = secure_aggregator.decrypt_result(aggregated_encrypted)

# 应用到全局模型
global_model.apply_update(aggregated_update)
```

## 性能对比

| 操作 | Shamir | Paillier |
|------|--------|----------|
| 加密 | 快速 | 较慢 |
| 聚合 | O(n) | O(n) |
| 解密 | O(1) | O(1) |
| 总时间 | 0.001s | 0.05s |

*基于小型测试（100维数组，5 客户端）*

## 配置

在 `config.yaml` 中可以配置：

```yaml
secure_aggregation:
    method: shamir  # 或 paillier
    threshold: 3
    paillier:
        key_size: 1024
        scale_factor: 1000
    shamir:
        prime: 2147483647
```

## 安全假设

- **半诚实模型 (Semi-honest)**: 服务器诚实地执行协议，但试图从通信中推断信息
- **拜占庭容错**: Shamir 方法可容忍 < 1/3 恶意客户端
- **隐私保证**: 在给定的安全参数下，计算上不可能从单个份额恢复原始梯度

## 相关论文/项目

- **SecureML**: https://arxiv.org/abs/1710.08729
- **ABY3**: https://github.com/aby3/aby3
- **LEAF**: https://leaf.cmu.edu/
- **TensorFlow Federated**: https://www.tensorflow.org/federated

## 注意事项

⚠️ **演示目的**: 本实现为教学/演示用途。生产环境请使用:
- [`TF Encrypted`](https://github.com/tf-encrypted/tf-encrypted)
- [`CrypTen`](https://github.com/facebookresearch/CrypTen)
- [`PySyft`](https://github.com/OpenMined/PySyft)

⚠️ **密钥管理**: 本实现使用简化的密钥分发。生产环境需要安全密钥管理方案。
