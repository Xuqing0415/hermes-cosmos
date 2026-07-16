# Hermes 

****: AI  
****: 5  
****: 20265

---

##  

### 1. CLI

```bash
pip install hermes-cosmos
```

### 2. 

```bash
export HERMES_API_URL=http://hermes.example.com/v1
export HERMES_TENANT_ID=your-tenant-id
```

### 3. 

```bash
# 
hermes job submit \
  --name "my-training-job" \
  --gpus 8 \
  --image "pytorch/pytorch:2.2.0" \
  --command "python train.py"

# 
hermes service create \
  --name "my-inference-service" \
  --model "llama-2-13b" \
  --replicas 2
```

---

##  

### 

|  |  |  |
|------|------|------|
| --gpus | GPU | 8 |
| --gpu-type | GPU | nvidia-h100 |
| --memory | (GB) | 512 |
| --cpu | CPU | 32 |

### 

```bash
--priority CRITICAL  # 
--priority HIGH
--priority NORMAL    # 
--priority LOW       # 
```

### 

```bash
--regions us-east    # Region
--carbon-aware       # 
--tee-enabled        # TEE
```

---

##  

```bash
# 
hermes job list

# 
hermes job get <job-id>

# 
hermes job logs <job-id> -f

# 
hermes cluster status
```

---

##  Checkpoint

```bash
# Checkpoint
hermes checkpoint save <job-id>

# Checkpoint
hermes checkpoint restore <checkpoint-id>
```

---

##  

```bash
# 
hermes job cost <job-id>

# 
hermes tenant cost
```

---

##  

### Q: 
A: 

### Q: GPU
A:  `hermes job metrics <job-id>` Grafana

### Q: 
A:  `hermes job logs <job-id>` 

---

##  

|  |  |
|------|----------|
| Slack: #hermes-support | 2 |
| : hermes-support@example.com | 4 |
| : PagerDuty | 30 |

---

****: docs.hermes.example.com
