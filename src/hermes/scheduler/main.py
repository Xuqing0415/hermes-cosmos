"""
Hermes Scheduler - 

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import time
import threading
from uuid import uuid4
import requests

# K8s
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

app = FastAPI(title="Hermes Scheduler - Proactive Migration", version="3.0")

v1 = None

def _init_k8s():
    global v1
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
    except Exception:
        try:
            config.load_incluster_config()
            v1 = client.CoreV1Api()
        except Exception:
            print("[SCHEDULER] Kubernetes config not found, running in standalone mode")
            v1 = None

# 
jobs: Dict[str, dict] = {}

# 
FAULT_PREDICTOR_URL = "http://localhost:8003"

# 
MIGRATION_THRESHOLD = 0.8  # 

class JobRequest(BaseModel):
    name: str
    tenant_id: str
    user_id: str
    gpu_count: int = 1
    priority: str = "NORMAL"

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "scheduler", "feature": "proactive-migration"}

@app.post("/jobs")
async def submit_job(job: JobRequest):
    job_id = str(uuid4())[:8]
    pod_name = f"hermes-job-{job_id}"
    
    if v1 is None:
        job_data = {
            "job_id": job_id,
            "name": job.name,
            "tenant_id": job.tenant_id,
            "user_id": job.user_id,
            "status": "RUNNING",
            "pod_name": pod_name,
            "gpu_count": job.gpu_count,
            "node_id": "local",
            "created_at": time.time(),
            "updated_at": time.time(),
            "migrations": 0,
            "last_migration_time": None,
            "k8s_status": "standalone"
        }
        jobs[job_id] = job_data
        return job_data
    
    try:
        create_training_pod(pod_name, job.name, job.gpu_count, job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pod: {str(e)}")
    
    job_data = {
        "job_id": job_id,
        "name": job.name,
        "tenant_id": job.tenant_id,
        "user_id": job.user_id,
        "status": "RUNNING",
        "pod_name": pod_name,
        "gpu_count": job.gpu_count,
        "node_id": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "migrations": 0,
        "last_migration_time": None,
        "k8s_status": "connected"
    }
    jobs[job_id] = job_data
    
    return job_data

def create_training_pod(pod_name, model_name, num_gpus, job_id):
    """Pod"""
    training_script = """
import os
import sys
import signal
import time
import torch
import torch.nn as nn
import torch.optim as optim

should_exit = False

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def handle_migration(sig, frame):
    global should_exit
    print('[MIGRATION] Received SIGUSR1, preparing to migrate...')
    should_exit = True

signal.signal(signal.SIGUSR1, handle_migration)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X = torch.randn(32, 10)
y = torch.randn(32, 10)

step = 0
while True:
    if should_exit:
        print('[MIGRATION] Saving checkpoint before migration...')
        torch.save({'step': step}, '/shared/checkpoint.pt')
        print('[MIGRATION] Ready for migration')
        sys.exit(0)
    
    optimizer.zero_grad()
    output = model(X)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    optimizer.step()
    
    step += 1
    if step % 10 == 0:
        print(f'Step {step}, Loss: {loss.item():.4f}')
    time.sleep(0.1)
"""
    
    container = client.V1Container(
        name="trainer",
        image="python:3.10-slim",
        command=["python", "-c", training_script],
        env=[
            client.V1EnvVar(name="JOB_ID", value=job_id),
            client.V1EnvVar(name="MODEL_NAME", value=model_name)
        ],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"}
        ),
        volume_mounts=[client.V1VolumeMount(
            name="shared",
            mount_path="/shared"
        )]
    )
    
    pod_spec = client.V1PodSpec(
        containers=[container],
        volumes=[client.V1Volume(
            name="shared",
            empty_dir=client.V1EmptyDirVolumeSource()
        )],
        restart_policy="Never"
    )
    
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=pod_name,
            labels={"hermes-job": job_id}
        ),
        spec=pod_spec
    )
    
    v1.create_namespaced_pod(namespace="default", body=pod)
    return pod_name

@app.post("/jobs/{job_id}/migrate")
async def migrate_job(job_id: str, target_node: Optional[str] = None):
    """"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    old_pod_name = job["pod_name"]
    
    start_time = time.time()
    
    try:
        # 1. PodSIGUSR1
        print(f"[MIGRATION] Sending SIGUSR1 to {old_pod_name}")
        send_signal_to_pod(old_pod_name, "SIGUSR1")
        
        # 2. Checkpoint30
        await asyncio.sleep(5)
        
        # 3. Pod
        new_pod_name = f"hermes-job-{job_id}-migrated"
        create_training_pod(new_pod_name, job["name"], job["gpu_count"], job_id)
        
        # 4. Pod
        await asyncio.sleep(10)
        
        # 5. Pod
        try:
            v1.delete_namespaced_pod(name=old_pod_name, namespace="default")
        except ApiException as e:
            if e.status != 404:
                raise
        
        migration_time = (time.time() - start_time) * 1000
        
        job["pod_name"] = new_pod_name
        job["migrations"] += 1
        job["last_migration_time"] = migration_time
        job["updated_at"] = time.time()
        
        print(f"[MIGRATION] Job {job_id} migrated successfully in {migration_time:.0f}ms")
        
        return {
            "message": "Job migrated successfully",
            "job_id": job_id,
            "old_pod": old_pod_name,
            "new_pod": new_pod_name,
            "migration_time_ms": migration_time,
            "total_migrations": job["migrations"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

def send_signal_to_pod(pod_name, signal_name):
    """Pod"""
    import subprocess
    subprocess.run([
        "kubectl", "exec", pod_name, "-n", "default", "--",
        "kill", f"-{signal_name}", "1"
    ], capture_output=True)

async def check_and_migrate():
    """"""
    while True:
        try:
            # 
            response = requests.get(f"{FAULT_PREDICTOR_URL}/nodes")
            if response.status_code == 200:
                nodes = response.json().get("nodes", {})
                
                for node_id, status in nodes.items():
                    probability = status.get("probability", 0.0)
                    
                    if probability >= MIGRATION_THRESHOLD:
                        print(f"[PROACTIVE] Node {node_id} has fault probability {probability:.2f}, triggering migration")
                        
                        # 
                        jobs_on_node = [
                            job for job in jobs.values()
                            if job["node_id"] == node_id and job["status"] == "RUNNING"
                        ]
                        
                        for job in jobs_on_node:
                            await migrate_job(job["job_id"])
            
            await asyncio.sleep(30)  # 30
        
        except Exception as e:
            print(f"[PROACTIVE] Error checking nodes: {e}")
            await asyncio.sleep(60)

@app.post("/trigger_proactive_check")
async def trigger_proactive_check():
    """"""
    await check_and_migrate()
    return {"message": "Proactive check completed"}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if v1 is not None:
        try:
            pod = v1.read_namespaced_pod(name=job["pod_name"], namespace="default")
            job["pod_status"] = pod.status.phase
            job["node_id"] = pod.spec.node_name
        except Exception:
            job["pod_status"] = "UNKNOWN"
    else:
        job["pod_status"] = "RUNNING"
    
    return job

@app.get("/jobs")
async def list_jobs():
    return {"total": len(jobs), "jobs": list(jobs.values())}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if v1 is not None:
        try:
            v1.delete_namespaced_pod(name=job["pod_name"], namespace="default")
        except Exception:
            pass
    
    del jobs[job_id]
    return {"message": "Job deleted", "job_id": job_id}

@app.on_event("startup")
async def startup_event():
    """"""
    print("[SCHEDULER] Starting Hermes Scheduler with Proactive Migration")
    
    # 
    asyncio.create_task(check_and_migrate())
    print("[SCHEDULER] Proactive migration checker started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
