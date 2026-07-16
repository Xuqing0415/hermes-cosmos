# Hermes 

****: 20265  
****: 30  
****: AI

---

##  

1. **** (5)
2. **** (15)
   - 
   - 
   - 
3. **** (5)
4. **Q&A** (5)

---

## 1⃣ 

###  Hermes?

> **AI**

### 

|  |  |
|------|------|
|  | GPU |
|  | <5 |
|  |  |
|  | GDPR |

### 

```

                    Gateway ()                       

                              
              
                                            
  
   Scheduler         Checkpoint          Agent       
  ()      ()      ()     
  
                                            
              
                              
                   
                      GPU     
                     (1000+ GPUs)    
                   
```

---

## 2⃣ 

###  1: 

```bash
#  Hermes CLI 
hermes job submit \
  --name "llama-2-finetune" \
  --gpus 32 \
  --image "pytorch/pytorch:2.2.0" \
  --command "python train.py" \
  --region us-east

#  ( < 500ms)
{
  "job_id": "abc-123",
  "status": "RUNNING",
  "scheduled_region": "us-east",
  "allocated_gpus": 32
}
```

****:
- 
- Region
- 

---

###  2: 

****: Pod

```bash
# 
kubectl delete pod llama-2-finetune-0

#  (< 5)
watch hermes job status abc-123
```

****:
```
0s: Pod
1s: Agent
2s: 
3s: PodCheckpoint
4s: 
```

****:
- ****: AgentGPU
- **Checkpoint**: Delta
- ****: MCTS

---

###  3: 

 Grafana Dashboard: http://grafana.hermes.example.com

****:

|  |  |  |
|------|------|------|
|  | P95 < 500ms |  |
| GPU | > 85% |  |
| Checkpoint | 99.9% |  |
|  | < 5s |  |

****:
1. 
2. 
3. 

---

## 3⃣ 

### 

|  |  | Hermes |  |
|------|--------|--------|------|
|  | ~10s | ~350ms | **28x** |
|  | ~120s | ~3s | **40x** |
| GPU | ~60% | ~87% | **+45%** |
|  | 72h | 48h | **-33%** |
|  | 100% | 62% | **-38%** |

### 

```
: LLaMA 2 13B Fine-tuning
: 32 x H100

:
- : 72
- : 3
- : 6
- : $12,000

Hermes:
- : 48 (-33%)
- : 0 ()
- : 3
- : $7,440 (-38%)
```

---

## 4⃣ 

### 

```python
#  SDK
pip install hermes-cosmos

# 
from hermes import Client

client = Client()
job = client.jobs.submit(
    name="my-training-job",
    gpu_count=8,
    image="pytorch/pytorch:latest",
    command="python train.py"
)

print(f"ID: {job.id}")
```

### 

|  |  |
|------|------|
|  | docs.hermes.example.com |
| API | api.hermes.example.com |
|  | github.com/hermes/examples |
|  | #hermes-support (Slack) |

---

##  Q&A



---

****:
- [ ] 
- [ ] 
- [ ] Grafana
- [ ] 
- [ ] 
