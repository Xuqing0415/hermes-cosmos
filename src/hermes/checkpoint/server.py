"""
Hermes Checkpoint Service - FastAPI版
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from uuid import uuid4
import pickle
import zlib
import os

app = FastAPI(title="Hermes Checkpoint Service", version="2.0")

# 内存存储（演示用）
checkpoints: Dict[str, dict] = {}
job_checkpoints: Dict[str, List[str]] = {}

# 本地存储路径
STORAGE_PATH = "/tmp/hermes/checkpoints"
os.makedirs(STORAGE_PATH, exist_ok=True)

class CheckpointRequest(BaseModel):
    job_id: str
    step: int
    data: Dict[str, Any]
    delta_from: Optional[str] = None

class CheckpointInfo(BaseModel):
    checkpoint_id: str
    job_id: str
    step: int
    timestamp: float
    size_bytes: int
    delta_from: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "checkpoint"}

@app.post("/checkpoints", response_model=CheckpointInfo)
async def save_checkpoint(request: CheckpointRequest):
    checkpoint_id = f"chk_{request.job_id[:8]}_{request.step}_{uuid4().hex[:8]}"
    
    # 序列化并压缩
    serialized = pickle.dumps(request.data)
    compressed = zlib.compress(serialized)
    
    # 保存到本地文件
    file_path = os.path.join(STORAGE_PATH, f"{checkpoint_id}.chk")
    with open(file_path, 'wb') as f:
        f.write(compressed)
    
    # 保存元数据
    checkpoint_info = {
        "checkpoint_id": checkpoint_id,
        "job_id": request.job_id,
        "step": request.step,
        "timestamp": 0,
        "size_bytes": len(compressed),
        "delta_from": request.delta_from
    }
    checkpoints[checkpoint_id] = checkpoint_info
    
    # 更新作业的checkpoint列表
    if request.job_id not in job_checkpoints:
        job_checkpoints[request.job_id] = []
    job_checkpoints[request.job_id].append(checkpoint_id)
    
    return checkpoint_info

@app.get("/checkpoints/{checkpoint_id}")
async def load_checkpoint(checkpoint_id: str):
    if checkpoint_id not in checkpoints:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    # 读取本地文件
    file_path = os.path.join(STORAGE_PATH, f"{checkpoint_id}.chk")
    try:
        with open(file_path, 'rb') as f:
            compressed = f.read()
        
        # 解压并反序列化
        serialized = zlib.decompress(compressed)
        data = pickle.loads(serialized)
        
        return {
            "checkpoint_id": checkpoint_id,
            "data": data
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Checkpoint file not found")

@app.get("/checkpoints/latest/{job_id}")
async def get_latest_checkpoint(job_id: str):
    if job_id not in job_checkpoints or not job_checkpoints[job_id]:
        raise HTTPException(status_code=404, detail="No checkpoints found for job")
    
    latest_id = job_checkpoints[job_id][-1]
    return {"checkpoint_id": latest_id}

@app.get("/checkpoints/list/{job_id}")
async def list_checkpoints(job_id: str):
    if job_id not in job_checkpoints:
        return {"checkpoints": []}
    
    return {"checkpoints": job_checkpoints[job_id]}

@app.delete("/checkpoints/{checkpoint_id}")
async def delete_checkpoint(checkpoint_id: str):
    if checkpoint_id not in checkpoints:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    job_id = checkpoints[checkpoint_id]["job_id"]
    
    # 删除文件
    file_path = os.path.join(STORAGE_PATH, f"{checkpoint_id}.chk")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 删除元数据
    del checkpoints[checkpoint_id]
    
    # 从作业列表中移除
    if job_id in job_checkpoints and checkpoint_id in job_checkpoints[job_id]:
        job_checkpoints[job_id].remove(checkpoint_id)
    
    return {"message": "Checkpoint deleted"}

@app.delete("/checkpoints/cleanup/{job_id}")
async def cleanup_checkpoints(job_id: str, keep_count: int = 3):
    if job_id not in job_checkpoints:
        return {"deleted": 0}
    
    checkpoints_list = job_checkpoints[job_id]
    if len(checkpoints_list) <= keep_count:
        return {"deleted": 0}
    
    to_delete = checkpoints_list[:-keep_count]
    for cp_id in to_delete:
        await delete_checkpoint(cp_id)
    
    return {"deleted": len(to_delete)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
