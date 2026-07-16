#  (MPC)

 Hermes ****

## 

### 

|  |  |  |  |  |
|------|--------|----------|----------|------|
| **Shamir ** |  1/3  |  |  |  |
| **Paillier ** |  |  |  |  |

### 

- ****: 
- ****: 
- ****:  `SecureAggregator` 
- ****: //

## 

###  Demo

```bash
# 
python test_secure_aggregation.py
```

### 

```python
from hermes_unified.secure_aggregation import SecureAggregator

# 
aggregator = SecureAggregator(
    method='shamir',  #  'paillier'
    num_clients=5,
    threshold=3
)

# 
encrypted_updates = {}
for client_id in range(5):
    update = np.random.randn(100)  # 
    encrypted = aggregator.client_encrypt_update(client_id, update)
    encrypted_updates[client_id] = encrypted

# 
aggregated_encrypted = aggregator.aggregate_encrypted(encrypted_updates)

# 
result = aggregator.decrypt_result(aggregated_encrypted)

# 
stats = aggregator.get_stats()
```

## Shamir 

```python
from hermes_unified.secure_aggregation import ShamirSecretSharing

shamir = ShamirSecretSharing(threshold=3)

# 
secret = 123.45
shares = shamir.share_scalar(secret, num_shares=5)

#  threshold 
reconstructed = shamir.reconstruct_scalar(shares[:3])
print(reconstructed)  # 123.45

# 
arr = np.random.randn(5, 5)
share_arrays = shamir.share_array(arr, 5)
reconstructed_arr = shamir.reconstruct_array(share_arrays[:3])
```

## Paillier 

```python
from hermes_unified.secure_aggregation import create_demo_paillier

paillier, paillier_arr = create_demo_paillier()

# /
ciphertext = paillier.encrypt(42)
plaintext = paillier.decrypt(ciphertext)

# 
c1 = paillier.encrypt(100)
c2 = paillier.encrypt(200)
c_sum = paillier.add_encrypted(c1, c2)
print(paillier.decrypt(c_sum))  # 300

# 
arr = np.array([1.5, 2.5, 3.5])
enc_arr = paillier_arr.encrypt_array(arr)
dec_arr = paillier_arr.decrypt_array(enc_arr)
```

## 

### 

```python
from hermes_unified.secure_aggregation import SecureFLClient

# 
secure_aggregator = SecureAggregator('shamir', num_clients=10)
client = SecureFLClient(client_id=0, model=my_model, secure_aggregator=secure_aggregator)

# 
encrypted_update = client.train_and_encrypt(train_data, epochs=1)
send_to_server(encrypted_update)
```

### 

```python
# 
aggregated_encrypted = secure_aggregator.aggregate_encrypted(encrypted_updates, client_weights)
aggregated_update = secure_aggregator.decrypt_result(aggregated_encrypted)

# 
global_model.apply_update(aggregated_update)
```

## 

|  | Shamir | Paillier |
|------|--------|----------|
|  |  |  |
|  | O(n) | O(n) |
|  | O(1) | O(1) |
|  | 0.001s | 0.05s |

*1005 *

## 

 `config.yaml` 

```yaml
secure_aggregation:
    method: shamir  #  paillier
    threshold: 3
    paillier:
        key_size: 1024
        scale_factor: 1000
    shamir:
        prime: 2147483647
```

## 

- ** (Semi-honest)**: 
- ****: Shamir  < 1/3 
- ****: 

## /

- **SecureML**: https://arxiv.org/abs/1710.08729
- **ABY3**: https://github.com/aby3/aby3
- **LEAF**: https://leaf.cmu.edu/
- **TensorFlow Federated**: https://www.tensorflow.org/federated

## 

 ****: /:
- [`TF Encrypted`](https://github.com/tf-encrypted/tf-encrypted)
- [`CrypTen`](https://github.com/facebookresearch/CrypTen)
- [`PySyft`](https://github.com/OpenMined/PySyft)

 ****: 
