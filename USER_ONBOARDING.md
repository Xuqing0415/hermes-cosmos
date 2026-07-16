# Hermes 

****: 1.0  
****: 1

---

##  

### 
> “Hi []K8sPod3-4Checkpoint‘’”

### 
> “3”

---

##  

### 
- [ ] K8s
- [ ] Docker

### 

#### 1. 10
```bash
# Hermes Checkpoint
docker build -t hermes-user-job --build-arg BASE_IMAGE=user-image .
```

#### 2. 5
```python
# Checkpoint
from hermes.checkpoint import HermesCheckpointer

checkpointer = HermesCheckpointer(job_id="my-job")

for epoch in range(epochs):
    for batch in dataloader:
        # ...
        
        # 100batchCheckpoint
        if batch_idx % 100 == 0:
            checkpointer.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            })
```

#### 3. Deployment5
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-training-job
spec:
  replicas: 1
  selector:
    matchLabels:
      app: user-training-job
  template:
    metadata:
      labels:
        app: user-training-job
    spec:
      restartPolicy: Never  # Hermes
      containers:
      - name: training
        image: hermes-user-job
        command: ["python", "train.py"]
        resources:
          limits:
            nvidia.com/gpu: 1
```

#### 4. Hermes5
```bash
# Hermes CLI
hermes job submit \
  --name "user-training-job" \
  --gpus 1 \
  --image hermes-user-job \
  --checkpoint-interval 300
```

#### 5. 10
```bash
# Pod
kubectl delete pod <pod-name>

# Pod3
watch kubectl get pods

# 
kubectl logs <new-pod-name> -f
```

---

##  

|  |  |
|------|--------|
|  | 3-4 |
| Checkpoint | < 1 |
|  | < 100batch |

---

##  

### 
> “”

### 
- “”
- “5”
- “”
