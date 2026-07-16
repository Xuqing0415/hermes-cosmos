# Hermes Cosmos  - 

## 

### 1. 

```bash
# 
cd hermes-cosmos

# 
chmod +x start.sh dev.sh

# 
./dev.sh help
```

### 2. 1-2

####  `dev.sh`Docker

```bash
# 
./dev.sh unit

# 
./dev.sh e2e

# 
./dev.sh scheduler

# 
./dev.sh test-job
```

####  `start.sh`Docker Compose

```bash
# 
./start.sh

# 
cat HEALTH_CHECK_REPORT.md

# 
./start.sh logs

# 
./start.sh stop
```

### 3. 

#### 

```python
# 
import httpx

response = httpx.post(
    "http://localhost:50051/jobs",
    json={
        "name": "first-test-job",
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "priority": "HIGH",
        "requirements": {
            "gpu_count": 8,
            "gpu_type": "nvidia-h100",
            "memory_gb": 128,
            "cpu_cores": 32
        },
        "image": "pytorch/pytorch:latest",
        "command": "python train.py"
    }
)

print(response.json())
```

#### SLA 

```bash
# 
pytest tests/test_e2e.py::TestJobFailover -v -s
```

#### 

```bash
# 
curl http://localhost:50051/status

# 
curl http://localhost:50051/cluster/summary
```

---

## 

```
hermes-cosmos/
 src/hermes/
    core/                    # 
       config.py           # 
       models.py           # 
       exceptions.py       # 
       store.py            # 
    gateway/                # API
    scheduler/              # 
    checkpoint/             # Checkpoint
    agent/                  # Agent
 tests/
    test_e2e.py             # 
    test_scheduler.py       # 
    test_checkpoint.py      # Checkpoint
    test_core.py            # 
 config/                     # 
 docker/                     # Docker
 deploy/                     # 
 start.sh                    # 
 dev.sh                      # 
```

---

## 

- [ ] ****
  - [ ] Python 3.11+ 
  - [ ] 
  - [ ]  (`pip install -e ".[dev]"`)

- [ ] ****
  - [ ]  (`test_core.py`)
  - [ ]  (`test_scheduler.py`)
  - [ ] Checkpoint (`test_checkpoint.py`)

- [ ] ****
  - [ ] 
  - [ ]  (< 500ms)
  - [ ]  (< 5s)

- [ ] ****
  - [ ] 
  - [ ] 200
  - [ ] 

---

## 

### Q: Docker Compose

A:  `dev.sh` Docker

### Q: 

A: 
```bash
source venv/bin/activate
pip install -e ".[dev]"
```

### Q: 

A: 
```bash
lsof -ti :50051 | xargs kill -9  # 
```

---

## 



1. ****
   - 
   - 
   - 

2. ****
3. ****

 `PROJECT_PLAN.md`
