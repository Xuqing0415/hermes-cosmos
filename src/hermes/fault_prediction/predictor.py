"""
Hermes Fault Predictor - LSTM
GPU
"""

import numpy as np
import onnxruntime as ort
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI(title="Hermes Fault Predictor", version="2.0")

class GPUMetrics(BaseModel):
    """GPU"""
    node_id: str
    gpu_id: str
    temperature: float  #  (C)
    power: float        #  (W)
    utilization: float  #  (%)
    memory_usage: float #  (%)
    ecc_errors: int     # ECC
    fan_speed: float    #  (%)

class PredictionRequest(BaseModel):
    """"""
    metrics: List[GPUMetrics]

class PredictionResponse(BaseModel):
    """"""
    node_id: str
    gpu_id: str
    fault_probability: float
    risk_level: str  # low, medium, high, critical

class FaultPredictor:
    """"""
    
    def __init__(self, model_path: str = "model.onnx"):
        self._model_path = model_path
        self._session = None
        self._input_name = None
        self._output_name = None
        
        # 
        self.mean = np.array([70.0, 150.0, 80.0, 75.0, 0.5, 60.0], dtype=np.float32)
        self.std = np.array([10.0, 50.0, 20.0, 20.0, 1.0, 20.0], dtype=np.float32)
        
        self._load_model()
    
    def _load_model(self) -> None:
        import os
        if os.path.exists(self._model_path):
            try:
                self._session = ort.InferenceSession(self._model_path)
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                print(f"Loaded model from {self._model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}, using mock predictions")
                self._session = None
        else:
            print(f"Model file {self._model_path} not found, using mock predictions")
            self._session = None
    
    def normalize(self, metrics: np.ndarray) -> np.ndarray:
        """"""
        return (metrics - self.mean) / self.std
    
    def predict(self, metrics: Dict[str, float]) -> float:
        """"""
        if self._session is not None:
            feature_order = ['temperature', 'power', 'utilization', 'memory_usage', 'ecc_errors', 'fan_speed']
            input_data = np.array([metrics.get(f, 0.0) for f in feature_order], dtype=np.float32)
            
            input_data = self.normalize(input_data)
            input_data = input_data.reshape(1, 1, 6)
            
            result = self._session.run([self._output_name], {self._input_name: input_data})
            probability = float(result[0][0][0])
            
            return min(max(probability, 0.0), 1.0)
        else:
            ecc_errors = metrics.get('ecc_errors', 0)
            temperature = metrics.get('temperature', 0)
            base_probability = min(ecc_errors * 0.1, 0.9)
            temp_factor = max(min((temperature - 70) / 50, 0.3), -0.5)
            return max(min(base_probability + temp_factor, 1.0), 0.0)
    
    def get_risk_level(self, probability: float) -> str:
        """"""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"

# 
predictor = FaultPredictor()

# 
node_metrics_cache = {}

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_fault(request: PredictionRequest):
    """GPU"""
    results = []
    
    for metrics in request.metrics:
        # 
        feature_dict = {
            'temperature': metrics.temperature,
            'power': metrics.power,
            'utilization': metrics.utilization,
            'memory_usage': metrics.memory_usage,
            'ecc_errors': metrics.ecc_errors,
            'fan_speed': metrics.fan_speed
        }
        
        # 
        probability = predictor.predict(feature_dict)
        risk_level = predictor.get_risk_level(probability)
        
        # 
        node_metrics_cache[metrics.node_id] = {
            'last_update': time.time(),
            'probability': probability,
            'risk_level': risk_level
        }
        
        results.append(PredictionResponse(
            node_id=metrics.node_id,
            gpu_id=metrics.gpu_id,
            fault_probability=probability,
            risk_level=risk_level
        ))
    
    return results

@app.get("/nodes/{node_id}/risk")
async def get_node_risk(node_id: str):
    """"""
    if node_id not in node_metrics_cache:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return node_metrics_cache[node_id]

@app.get("/nodes")
async def get_all_nodes():
    """"""
    return {"nodes": node_metrics_cache}

@app.post("/inject_fault")
async def inject_fault(node_id: str, probability: float = 0.9):
    """"""
    node_metrics_cache[node_id] = {
        'last_update': time.time(),
        'probability': probability,
        'risk_level': predictor.get_risk_level(probability)
    }
    return {"message": f"Fault probability set to {probability} for node {node_id}"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "fault-predictor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
