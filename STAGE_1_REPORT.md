#  - 

**** 2026-05-11 ()  
**** P0 -  128  +  Checkpoint 

---

##  

### 1. 

|  |  |  |
|------|------|------|
| Python 3.11+ |   | `pyproject.toml`  |
|  |   | `dev.sh`  |
|  |   | `pyproject.toml`  `setuptools` |
|  |   |  |

### 2. 

|  |  |  |  |
|------|------|----------|----------|
| **Core** |  | `src/hermes/core/` |  |
| **Scheduler** |  | `src/hermes/scheduler/` |  |
| **Gateway** |  | `src/hermes/gateway/` | REST API |
| **Checkpoint** |  | `src/hermes/checkpoint/` | Delta  |
| **Agent** |  | `src/hermes/agent/` |  |

### 3. API 

|  |  |  |
|------|------|------|
| Gateway | `GET /health` |   |
| Scheduler | `GET /health` |   |
| Scheduler | `GET /status` |   |
| Scheduler | `POST /jobs` |   |
| Scheduler | `GET /jobs/{id}` |   |
| Scheduler | `DELETE /jobs/{id}` |   |
| Scheduler | `GET /cluster/summary` |   |

### 4. 

|  |  |  |
|----------|------|----------|
|  |   | `tests/test_core.py` |
|  |   | `tests/test_scheduler.py` |
| Checkpoint  |   | `tests/test_checkpoint.py` |
|  |   | `tests/test_e2e.py` |

---

##  

### 

|  |  |
|------|------|
| `dev.sh` | ****  |
| `quick_test.py` | ****  |
| `DEVELOPMENT.md` |  |
| `pytest.ini` ( pyproject.toml ) |  |

### 

|  |  |
|------|------|
| `src/hermes/core/store.py` | **** |
| `src/hermes/scheduler/main.py` | **** API |
| `src/hermes/core/models.py` | JobCheckpointResource |
| `src/hermes/core/config.py` | Pydantic Settings |

### 

|  |  |
|------|----------|
| `tests/test_e2e.py` | **** |
| `tests/test_scheduler.py` |  |
| `tests/test_checkpoint.py` | Checkpoint  |

---

##  

### 

```bash
cd hermes-cosmos

# 1. 
python quick_test.py

# 2. 
python -m pytest tests/test_e2e.py -v -s
```

### 

```bash
cd hermes-cosmos

# Windows Linux/Mac 
# chmod +x dev.sh start.sh

# 
./dev.sh unit

# 
./dev.sh e2e

# 
./dev.sh scheduler
```

###  Docker Compose

```bash
cd hermes-cosmos/docker

# 
docker-compose up -d

# 
docker-compose ps

# 
docker-compose logs -f

# 
docker-compose down
```

---

##  

 `tests/test_e2e.py` 

```
======================================================================
HERMES COSMOS - END-TO-END TEST SUITE
======================================================================

Test 1: Job Submission and Scheduling
----------------------------------------------------------------------
 Job submitted, ID: <uuid>
 Scheduling latency: XXms (< 500ms target)
 Job status: QUEUED

Test 2: Cluster Summary
----------------------------------------------------------------------
 Total GPUs: 1500
 Available GPUs: 1500
 us-east: 500/500 GPUs available
 us-west: 500/500 GPUs available
 eu-west: 500/500 GPUs available

Test 3: Scheduler Status
----------------------------------------------------------------------
 Health status: healthy
 Is leader: True
 Term: 1

Test 4: Job Failover Recovery
----------------------------------------------------------------------
 Job submitted for failover test: <uuid>
 Recovery time: X.XXXs (< 5.0s SLA)
 Recovery SLA met!

Test 5: Data Sovereignty Enforcement
----------------------------------------------------------------------
 EU-only job submitted: <uuid>
 Data sovereignty policy enforced
```

---

##  SLA 

| KPI |  |  |
|-----|------|----------|
|  P95 | < 500ms |  |
| Checkpoint  | < 0.1% |   |
|  | < 5s |   |

---

##  



1. **** (`src/hermes/scheduler/main.py:89-102`)
   - 
   - 
   - 

2. **** (`src/hermes/core/store.py`)
   - 
   - 
   - Checkpoint 

3. **** (`tests/test_e2e.py`)
   - 
   - 
   - SLA 

---

##  

### 

1. ****`python quick_test.py`
2. ****`./dev.sh e2e`
3. **** `HEALTH_CHECK_REPORT.md` 
4. ****

### 

1. `ruff``mypy`
2. 
3. 
4.  (ADR)

### 

1.  Redis/etcd 
2. Raft
3. gRPC
4. 

---

##  

- `README.md` - 
- `DEVELOPMENT.md` - 
- `config/*.yaml` - 
- `deploy/kubernetes/hermes.yaml` - K8s 
- `docker/docker-compose.yml` - 

---

**** 


-   FastAPI  REST API
-  
-   MCTS 
-  
-  
