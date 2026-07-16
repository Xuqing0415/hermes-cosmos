# Meta-FL Controller



## 

### 1. 
- ****: 
- ****: Non-IIDJensen-Shannon
- ****: 

### 2. 
- 
- 
- : FedAvgDittoFedRepKrumTrimmed Mean

### 3. 
- 
- Pareto
- 

### 4. 
- 
- 
- 

## 

### 

```bash
# 
python -m hermes_unified.meta_fl.run_meta --mode cli

# 
python -m hermes_unified.meta_fl.run_meta --mode cli --clients 20 --non-iid 0.7
```

### 

```bash
# API
python -m hermes_unified.meta_fl.run_meta --mode server --port 8503
```

### API 

```bash
# 
curl -X POST http://localhost:8503/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"num_clients": 20, "non_iid_level": 0.7}'

# 
curl -X POST http://localhost:8503/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"num_clients": 15, "non_iid_level": 0.5}'
```

### Python API

```python
from hermes_unified.meta_fl import TaskFeatureExtractor, AlgorithmRecommender

# 
extractor = TaskFeatureExtractor()
features = extractor.extract(client_data, resources)

# 
recommender = AlgorithmRecommender()
recommendation = recommender.recommend(features)

print(f": {recommendation['algorithm_name']}")
print(f": {recommendation['confidence']}%")
print(f": {recommendation['explanation']}")
```

## 

### TaskFeatureExtractor

```python
extractor = TaskFeatureExtractor()

# 
features = extractor.extract(client_data, resources)

# :
# - num_clients: 
# - non_iid_gini: Non-IID
# - non_iid_jsd: Jensen-Shannon
# - avg_bandwidth_mbps: 
# - avg_compute_score: 
```

### PerformancePredictor

```python
predictor = PerformancePredictor()

# 
X, y, metadata = predictor.generate_synthetic_dataset(n_samples=1000)
predictor.train(X, y, metadata['feature_names'])

# 
predictions = predictor.predict(features)
# predictions['fedavg'] = {'accuracy': 0.85, 'rounds': 45, 'communication': 1.0}
```

### AlgorithmRecommender

```python
recommender = AlgorithmRecommender()

# 
recommendation = recommender.recommend(features)

# 
analysis = recommender.analyze_task(features)

# Pareto
frontier = recommender.get_pareto_frontier(features)

# 
comparison = recommender.compare_algorithms(features)
```

## 

|  |  |  |  |
|------|----------|------|------|
| **FedAvg** | IID |  | Non-IID |
| **Ditto** | Non-IID |  |  |
| **FedRep** |  |  |  |
| **Krum** |  |  |  |
| **Trimmed Mean** |  |  |  |

## Non-IID 

Non-IID

|  |  |  |
|------|------|------|
| **Gini** |  | 0-1Non-IID |
| **JSD** |  | 0-1 |
| **** |  |  |

****:
- Non-IID < 0.3: Non-IIDFedAvg
- 0.3 < Non-IID < 0.6: Non-IIDDittoFedRep
- Non-IID > 0.6: Non-IIDDitto

## 

### FL

```python
# run_pfl.py
if args.auto_mode:
    from hermes_unified.meta_fl import AlgorithmRecommender
    
    features = extract_features(client_data)
    recommender = AlgorithmRecommender()
    recommendation = recommender.recommend(features)
    
    print(f": {recommendation['algorithm_name']}")
    print(f": {recommendation['explanation']}")
    
    # 
    run_algorithm(recommendation['recommended_algorithm'], client_data)
```

## 

1.  `recommender.py`  `algorithm_descriptions` 
2.  `performance_predictor.py` 
3. 

## 

```

                   Meta-FL Controller                    
              
   FeatureExtractor →  PerformancePred. →          
     ()         ()                
              
                                                      
                                                      
            
             AlgorithmRecommender                     
    (:  + )                  
            
                                                       
                                                       
                                   
                  Web Dashboard                       
                                   

```

## API 

|  |  |  |
|------|------|------|
| `/api/recommend` | POST |  |
| `/api/analyze` | GET |  |
| `/` | GET | API |

## 

- numpy
- scipy
- scikit-learn
- flask
- flask-cors

## 

MIT License
