# 

****: Hermes   
****: Hermes   
****: 20265  
****: v1.0

---

## 1. 

****

****:
- 
- 
- 
- 

****: 

---

## 2. 

### 2.1 


- 
- 
- 

### 2.2 

****
1. 
2. 
3. 

### 2.3 

- 
- 
- 

---

## 3. 

### 3.1 

```mermaid
flowchart TD
    A[] --> B[]
    B --> C[]
    C --> D[]
    E[] --> F[]
    F --> D
    D --> G[]
    
    subgraph 
        H[()]
        I[()]
    end
    
    B --> H
    F --> I
    G --> I
```

### 3.2 

#### 3.2.1 

****: DFS

```python
def find_critical_paths(start_node):
    paths = []
    dfs(start_node, [start_node], 0)
    paths.sort(key=lambda x: x[1], reverse=True)
    return paths[:max_count]
```

****: 

```python
def calculate_betweenness_centrality():
    centrality = {}
    for source, target in all_node_pairs:
        for path in shortest_paths(source, target):
            for node in path:
                centrality[node] += 1
    normalize(centrality)
    return centrality
```

#### 3.2.2 

****:

```
risk_score(node) = centrality * (2 if critical_path else 1)
```

****:

```
fitness(experiment) = total_impact - cost * 0.3
```


- `total_impact`: 
- `cost`: 

#### 3.2.3 

****:

|  |  |  |
|------|------|----------|
| P99 | > 500ms | 3 |
|  | > 5% | 5 |
|  | < 95% | 5 |

****:

```python
def check_stop_conditions(metrics):
    for condition in stop_conditions:
        if violate_threshold(metrics[condition.name]):
            record_violation(condition.name)
            if violation_duration >= condition.duration:
                return True, condition.description
    return False, None
```

---

## 4. 

### 4.1 

****:
- 16
- 15
- api-gateway

****:
|  |  |  |
|------|------|------|
| POD_DELETE | Pod | 0-30 |
| NETWORK_LATENCY |  | 100-5000ms |
| DISK_FULL |  | 80-99% |
| CPU_HIGH | CPU | 80-100% |

****:
- API P99 (ms)
-  (%)
-  (%)
- CPU (%)
-  (%)

### 4.2 

|  |  |  |
|------|--------|----------|
|  | 10 |  |
|  | 5 |  |

****:
1. 
2. 
3. 
4. 45
5. 
6. 

### 4.3 

|  |  |
|------|------|
|  |  |
|  | P99 |
|  |  |
|  |  |

---

## 5. 

### 5.1 

|  |  |  |
|------|--------|--------|
|  | 80.0% | 0.0% |
|  (ms) | 423.56 ± 156.23 | 148.92 ± 12.34 |
|  | 0.08 ± 0.03 | 0.01 ± 0.00 |
|  (%) | 91.56 ± 3.23 | 99.45 ± 0.12 |

### 5.2 

![](stop_rate_comparison.png)

### 5.3 

![](latency_change.png)

### 5.4 

![](error_rate_curve.png)

### 5.5 

**t**:
- t(13) = 5.82, p < 0.001
- t(13) = 6.15, p < 0.001
- t(13) = -7.32, p < 0.001

p < 0.001

---

## 6. 

### 6.1 

****: schedulercheckpoint

****: 

****: 

### 6.2 

- ****: 
- ****: 
- ****: 

### 6.3 

1. ****: 
2. **ML**: 
3. ****: 
4. ****: K8sPrometheus

---

## 7. 



****:
- K8s
- 
- 

---

## 8. 

[1] Netflix. (2011). Chaos Monkey. Retrieved from https://netflix.github.io/chaosmonkey/

[2] Fowler, M. (2018). Chaos Engineering. Retrieved from https://martinfowler.com/bliki/ChaosEngineering.html

[3] Google. (2020). Site Reliability Engineering. O'Reilly Media.

[4] Kubernetes. (2023). Chaos Mesh. Retrieved from https://chaos-mesh.org/

---

## 

### A. 

#### A.1 

```python
class ServiceGraphAnalyzer:
    def analyze(self, start_node='api-gateway'):
        critical_paths = self.find_critical_paths(start_node)
        centrality = self.calculate_betweenness_centrality()
        return AnalysisResult(
            critical_paths=critical_paths,
            centrality_scores=centrality
        )
```

#### A.2 

```python
def _fitness(self, faults, services):
    total_impact = sum(
        self.calculate_risk_score(services.get(f.target_service)) * 
        f.intensity * f.expected_impact
        for f in faults
    )
    total_cost = sum(f.duration_sec * f.intensity for f in faults)
    return total_impact - total_cost * 0.3
```

### B. 

```yaml
services:
  - name: api-gateway
    dependencies: [scheduler, auth, checkout]
  - name: scheduler
    dependencies: [checkpoint, fault-prediction]
  - name: checkpoint
    dependencies: [redis, minio]
  # ... 
```

### C. 

 `experiment_results.csv`

---

****
