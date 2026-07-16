# Hermes Cosmos

**AI**

## 

Hermes RegionGPUAI

### 

- ****: RegionMCTS
- **Checkpoint**: CheckpointDelta
- ****: Agent
- ****: eBPF GPUTracing

## 

```
hermes-cosmos/
 src/hermes/
    core/            # 
    gateway/         # API
    scheduler/       # 
    checkpoint/      # Checkpoint
    agent/           # Agent
    fault_prediction/# 
    cli.py           # 
 config/              # 
 docker/              # Docker
 deploy/              # 
    alibaba/         # 
    aws/             # AWS
    kubernetes/      # Kubernetes
    prometheus/      # Prometheus
 examples/            # 
 hermes-operator/     # Kubernetes Operator
 grafana/             # Grafana
 tests/               # 
```

## 

### 

```bash
# 
git clone https://github.com/hermes-cosmos/hermes.git
cd hermes

# 
pip install -e ".[dev]"
```

### Docker Compose

```bash
cd docker
docker-compose up -d
```

### CLI

```bash
# 
hermes-cli status

# 
hermes-cli jobs submit my-job \
  --tenant-id tenant-1 \
  --user-id user-1 \
  --image pytorch/pytorch:latest \
  --gpu-count 8

# 
hermes-cli jobs list

# 
hermes-cli resources summary
```

## 

### 

|  |  |  |
|------|------|--------|
|  | API | FastAPI + JWT + SPIFFE |
|  |  | Python + MCTS + Raft |
|  | Checkpoint | Python + RDMA + Redis |
|  |  | Prometheus + OpenTelemetry |

### API

- `GET /v1/health` - 
- `POST /v1/jobs` - 
- `GET /v1/jobs` - 
- `GET /v1/jobs/{job_id}` - 
- `DELETE /v1/jobs/{job_id}` - 
- `GET /v1/checkpoints` - 
- `GET /v1/resources` - 
- `GET /v1/resources/cluster/summary` - 

## 

 `config/` :

- `gateway.yaml` - 
- `scheduler.yaml` - 
- `checkpoint.yaml` - Checkpoint
- `agent.yaml` - Agent

:

```bash
export HERMES_GATEWAY_SERVER__PORT=8081
export HERMES_SCHEDULER_ALGORITHM__CARBON_AWARE=true
```

## 

### 

```bash
pytest tests/ -v
```

### 

```bash
# 
black src/ tests/
isort src/ tests/

# 
mypy src/

# Lint
ruff check src/
```

## 


