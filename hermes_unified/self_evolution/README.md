# Self-Evolving Federated Learning

 - Hermes 

## 

Hermes 

## 

### 1. 
- 
- UCB
- 

### 2. 
- Upper Confidence Bound (UCB) 
- 
- epsilon-greedy 

### 3. 
- 
- GitOps 
- 

### 4. 
- 
- 
- 
- Web 

## 

### 

```bash
#  10  1 
python -m hermes_unified.self_evolution.run_self_evolution \
    --generations 10 \
    --hours 1.0
```

### 

```bash
# 
python -m hermes_unified.self_evolution.run_self_evolution \
    --generations 0 \
    --hours 0.1 \
    --port 8504
```

### Python API

```python
from hermes_unified.self_evolution import SelfEvolvingFederatedLearning

# 
sef = SelfEvolvingFederatedLearning(
    db_path="self_evolution_db.json",
    output_dir="results"
)

# 
sef.evolve(num_generations=20, max_time_hours=2.0)

# 
report = sef.generate_evolution_report()
print(f"Best Accuracy: {report['summary']['best_accuracy']}")

# 
tree = sef.get_evolution_tree()
```

## 

```python
config = {
    'num_clients': 20,          #  (5-50)
    'non_iid_level': 0.6,       # Non-IID  (0.1-0.95)
    'algorithm': 'ditto',       # 
    'learning_rate': 0.05,      # 
    'num_rounds': 30,           # 
    'bandwidth': 50.0,          #  Mbps
    'compute_power': 1.0        # 
}
```

## 

### UCB (Upper Confidence Bound)

```
UCB_score = mean_accuracy + c * sqrt(ln(n) / n_i)

where:
- mean_accuracy: 
- c:  ( 1.0)
- n: 
- n_i: 
```

### epsilon-greedy

- `epsilon=0.1`: 10% 
- 90% 

## 

```
self_evolution_results/
 evolution_progress.json     # 
 best_deployment.json       # 
 self_evolution_db.json      # 
```

## 

 `http://localhost:8504` 

- 
- 
- 
- 
- 
- 

## 

```

                Self-Evolving FL System                    
     
   Configuration Selector (UCB/epsilon-greedy)            
     
                                                         
     
           Experiment Runner (simulated)                 
     
                                                         
     
   Online Learning (Performance Predictor update)       
     
                                                         
     
   Milestone Check & Auto-Deploy                        
     
                                                         
     
   Database & Visualization                             
     

```

## 

### 

```python
def select_config_custom(self) -> Dict:
    # 
    pass
```

### 

 `run_single_experiment` 

### 

 `auto_deploy_best_config`  GitOps/ArgoCD 

## 

-  FL 
- 
- 
-  GitOps/ArgoCD 
- SHAP 

## 

- numpy
- flask
- flask-cors
- plotly ()

## 

MIT License
