"""
Hermes Fault Predictor - 基于LSTM的故障预测服务
实时预测GPU节点故障概率，支持主动迁移
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
    """GPU指标数据"""
    node_id: str
    gpu_id: str
    temperature: float  # 温度 (C)
    power: float        # 功耗 (W)
    utilization: float  # 利用率 (%)
    memory_usage: float # 内存使用 (%)
    ecc_errors: int     # ECC错误数
    fan_speed: float    # 风扇转速 (%)

class PredictionRequest(BaseModel):
    """预测请求"""
    metrics: List[GPUMetrics]

class PredictionResponse(BaseModel):
    """预测响应"""
    node_id: str
    gpu_id: str
    fault_probability: float
    risk_level: str  # low, medium, high, critical

class FaultPredictor:
    """故障预测器"""
    
    def __init__(self, model_path: str = "model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 归一化参数（训练数据的均值和标准差）
        self.mean = np.array([70.0, 150.0, 80.0, 75.0, 0.5, 60.0], dtype=np.float32)
        self.std = np.array([10.0, 50.0, 20.0, 20.0, 1.0, 20.0], dtype=np.float32)
    
    def normalize(self, metrics: np.ndarray) -> np.ndarray:
        """归一化输入数据"""
        return (metrics - self.mean) / self.std
    
    def predict(self, metrics: Dict[str, float]) -> float:
        """预测故障概率"""
        # 提取特征
        feature_order = ['temperature', 'power', 'utilization', 'memory_usage', 'ecc_errors', 'fan_speed']
        input_data = np.array([metrics.get(f, 0.0) for f in feature_order], dtype=np.float32)
        
        # 归一化
        input_data = self.normalize(input_data)
        
        # 添加时间步维度
        input_data = input_data.reshape(1, 1, 6)
        
        # 推理
        result = self.session.run([self.output_name], {self.input_name: input_data})
        probability = float(result[0][0][0])
        
        return min(max(probability, 0.0), 1.0)
    
    def get_risk_level(self, probability: float) -> str:
        """根据概率返回风险等级"""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"

# 全局预测器实例
predictor = FaultPredictor()

# 节点状态缓存（模拟实时指标）
node_metrics_cache = {}

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_fault(request: PredictionRequest):
    """预测GPU故障概率"""
    results = []
    
    for metrics in request.metrics:
        # 构建特征字典
        feature_dict = {
            'temperature': metrics.temperature,
            'power': metrics.power,
            'utilization': metrics.utilization,
            'memory_usage': metrics.memory_usage,
            'ecc_errors': metrics.ecc_errors,
            'fan_speed': metrics.fan_speed
        }
        
        # 预测
        probability = predictor.predict(feature_dict)
        risk_level = predictor.get_risk_level(probability)
        
        # 更新缓存
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
    """获取节点风险状态"""
    if node_id not in node_metrics_cache:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return node_metrics_cache[node_id]

@app.get("/nodes")
async def get_all_nodes():
    """获取所有节点状态"""
    return {"nodes": node_metrics_cache}

@app.post("/inject_fault")
async def inject_fault(node_id: str, probability: float = 0.9):
    """手动注入故障概率（用于测试）"""
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
