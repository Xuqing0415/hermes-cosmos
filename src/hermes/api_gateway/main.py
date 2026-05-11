"""
Hermes API Gateway - FastAPI版
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI(title="Hermes API Gateway", version="2.0")

# 后端服务地址
SCHEDULER_URL = "http://localhost:8001"
CHECKPOINT_URL = "http://localhost:8002"

class JobSubmitRequest(BaseModel):
    job_id: str
    model_name: str
    batch_size: int = 16
    epochs: int = 2
    gpu_type: str = "CPU"
    num_gpus: int = 1
    checkpoint_interval: int = 1
    priority: str = "normal"

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gateway"}

@app.get("/api/v1/health")
async def api_health():
    return {"status": "ok"}

@app.post("/api/v1/jobs")
async def submit_job(request: JobSubmitRequest):
    """提交训练作业"""
    
    # 转换为调度器格式
    scheduler_request = {
        "name": f"{request.model_name}-{request.job_id}",
        "tenant_id": "default",
        "user_id": "api-user",
        "gpu_count": request.num_gpus,
        "priority": request.priority.upper(),
        "region_preferences": None,
        "carbon_aware": True
    }
    
    # 调用调度器
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SCHEDULER_URL}/jobs",
                json=scheduler_request
            )
            
            if response.status_code == 200:
                job = response.json()
                return {
                    "job_id": request.job_id,
                    "status": "accepted",
                    "scheduler_job_id": job["job_id"],
                    "region": job.get("region")
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Scheduler error: {response.text}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit job: {str(e)}"
            )

@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """获取作业状态"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SCHEDULER_URL}/jobs/{job_id}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="Job not found")
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Scheduler error: {response.text}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get job: {str(e)}"
            )

@app.get("/api/v1/jobs")
async def list_jobs(status: Optional[str] = None):
    """列出作业"""
    params = {}
    if status:
        params["status"] = status
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SCHEDULER_URL}/jobs", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Scheduler error: {response.text}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list jobs: {str(e)}"
            )

@app.post("/api/v1/jobs/{job_id}/fail")
async def simulate_failure(job_id: str):
    """模拟作业故障"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{SCHEDULER_URL}/simulate_failure/{job_id}")
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "job_id": job_id,
                    "status": "recovered" if "recovered" in result.get("message", "").lower() else "failed",
                    "details": result
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Scheduler error: {response.text}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to simulate failure: {str(e)}"
            )

@app.get("/api/v1/cluster")
async def cluster_summary():
    """获取集群状态"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SCHEDULER_URL}/cluster/summary")
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Scheduler error: {response.text}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get cluster summary: {str(e)}"
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
