# Hermes 

****: 1.0  
****: AI  
****: 30

---

##  

### 1. 

```bash
# 
export HERMES_API_URL=http://hermes.example.com/v1
export HERMES_TENANT_ID=your-tenant-id

# CLI
pip install hermes-cosmos
```

### 2. 

```python
import httpx

response = httpx.post(
    f"{HERMES_API_URL}/jobs",
    json={
        "name": "my-first-training-job",
        "tenant_id": HERMES_TENANT_ID,
        "user_id": "your-user-id",
        "priority": "NORMAL",
        "requirements": {
            "gpu_count": 8,
            "gpu_type": "nvidia-h100",
            "memory_gb": 512,
            "cpu_cores": 32,
        },
        "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        "command": "python train.py --epochs 10",
    }
)

job = response.json()
print(f"ID: {job['job']['id']}")
```

### 3. 

```bash
# 
curl "$HERMES_API_URL/jobs/{job-id}"

# 
curl "$HERMES_API_URL/resources/cluster/summary"
```

---

##  

### 

```python
# 
"priority": "CRITICAL"  # 
"priority": "HIGH"      # 
"priority": "NORMAL"    # 
"priority": "LOW"       # 
```

### 

```python
"requirements": {
    "gpu_count": 8,          # GPU
    "gpu_type": "nvidia-h100", # GPU
    "memory_gb": 512,        # 
    "cpu_cores": 32,         # CPU
    "storage_gb": 1000,      # 
    "network_bandwidth_gbps": 100,  # 
    "max_duration_hours": 72, # 
    "checkpoint_interval_seconds": 300, # Checkpoint
}
```

### 

```python
"constraints": {
    "regions": ["eu-west"],  # 
    "data_locality": true,   # 
    "carbon_aware": true,    # 
    "max_carbon_intensity": 200,  # 
}
```

---

##  Checkpoint 

### Checkpoint

```python
"checkpoint_enabled": true,
"requirements": {
    "checkpoint_interval_seconds": 300,  # 5
}
```

### Checkpoint

```bash
curl -X POST "$HERMES_API_URL/jobs/{job-id}/checkpoint"
```

### Checkpoint

```bash
curl -X POST "$HERMES_API_URL/checkpoints/{checkpoint-id}/restore"
```

### 

```python
import torch
from hermes.checkpoint import HermesCheckpointer

checkpointer = HermesCheckpointer(job_id="your-job-id")

# 
for epoch in range(epochs):
    # Checkpoint
    try:
        state = checkpointer.load()
        start_epoch = state["epoch"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    except Exception:
        start_epoch = 0
    
    for batch in dataloader:
        # 
        loss = train_step(batch)
        
        # CheckpointHermes
        if batch_idx % 100 == 0:
            checkpointer.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss.item(),
            })
```

---

##  

### 

```bash
# 
curl "$HERMES_API_URL/jobs/{job-id}/cost-estimate"

# 
curl "$HERMES_API_URL/tenants/{tenant-id}/cost-summary"
```

### 

|  |  |  |
|------|------|----------|
|  |  | GPU+15% |
|  |  | -30% |
|  | Spot | -40% |

### 

```python
"constraints": {
    "carbon_aware": true,           # 
    "max_carbon_intensity": 200,    # 
}
```

---

##  

### 

|  |  |  |
|------|------|----------|
|  | status=QUEUED |  |
|  | status=FAILED |  |
| Checkpoint |  |  |
| GPU |  | <5 |

### 

```bash
# 
curl "$HERMES_API_URL/agents/{agent-id}/predictions"

# 
curl -X POST "$HERMES_API_URL/jobs/{job-id}/migrate?target_region=us-west"
```

### 

- ****: #hermes-support Slack
- ****: hermes-support@example.com
- ****: docs.hermes.example.com

---

##  

### 

|  |  |  |
|------|------|--------|
|  |  | < 500ms |
| GPU | GPU | > 80% |
| Checkpoint | Checkpoint | < 1 |
|  |  | < 5 |

### 

```bash
# Prometheus
# P99
rate(hermes_scheduler_scheduling_latency_seconds[5m])

# GPU
avg(hermes_agent_gpu_utilization)

# Checkpoint
sum(hermes_checkpoint_total{status="success"}) / sum(hermes_checkpoint_total)
```

---

##  

### 

-  `data_locality` 
- 
- TEE

### 

- 
- 
- 

---

##  

```python
import httpx

# BERT
response = httpx.post(
    "http://hermes.example.com/v1/jobs",
    json={
        "name": "bert-fine-tuning",
        "tenant_id": "nlp-team",
        "user_id": "john-doe",
        "priority": "HIGH",
        "requirements": {
            "gpu_count": 16,
            "gpu_type": "nvidia-h100",
            "memory_gb": 1024,
            "cpu_cores": 64,
            "storage_gb": 2000,
            "checkpoint_interval_seconds": 180,
        },
        "constraints": {
            "regions": ["us-east"],
            "carbon_aware": true,
        },
        "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        "command": """
            python run_pretraining.py \
                --model_type bert \
                --model_name_or_path bert-base-uncased \
                --do_train \
                --train_file train.txt \
                --num_train_epochs 3 \
                --per_device_train_batch_size 32
        """,
        "environment": {
            "WANDB_PROJECT": "nlp-bert",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "volumes": [
            "nlp-datasets:/data",
        ],
        "checkpoint_enabled": true,
        "metadata": {
            "experiment_name": "bert-v2-fine-tune",
            "team": "nlp",
        },
    }
)

print("")
print(f"ID: {response.json()['job']['id']}")
```

---

##  

|  |  |
|------|------|
| API | https://hermes.example.com/docs |
|  | https://github.com/hermes-cosmos/examples |
|  | https://docs.hermes.example.com/best-practices |
|  | https://docs.hermes.example.com/troubleshooting |
