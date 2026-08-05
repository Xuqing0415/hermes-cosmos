"""
Hermes Distributed Scheduler - 
PyTorch DDPPod
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import time
import threading
import json
from uuid import uuid4

# K8s
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

app = FastAPI(title="Hermes Distributed Scheduler", version="2.0")

# K8s
try:
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
except Exception:
    print("[DDP-SCHEDULER] Kubernetes config not found, running in standalone mode")
    v1 = None
    apps_v1 = None

# 
jobs: Dict[str, dict] = {}

class DDPJobRequest(BaseModel):
    name: str
    tenant_id: str
    user_id: str
    num_replicas: int = 4
    image: str = "hermes-ddp:latest"
    checkpoint_interval: int = 10

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "distributed-scheduler"}

@app.post("/ddp/jobs")
async def submit_ddp_job(job: DDPJobRequest):
    """"""
    job_id = str(uuid4())[:8]
    
    # StatefulSet
    try:
        create_ddp_statefulset(job_id, job.name, job.num_replicas, job.image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create StatefulSet: {str(e)}")
    
    job_data = {
        "job_id": job_id,
        "name": job.name,
        "tenant_id": job.tenant_id,
        "user_id": job.user_id,
        "status": "RUNNING",
        "num_replicas": job.num_replicas,
        "statefulset_name": f"ddp-{job_id}",
        "checkpoint_interval": job.checkpoint_interval,
        "created_at": time.time(),
        "updated_at": time.time(),
        "recoveries": 0,
        "last_recovery_time": None
    }
    jobs[job_id] = job_data
    
    print(f"[DDP]  {job_id} : {job.num_replicas}")
    return job_data

def create_ddp_statefulset(job_id, name, num_replicas, image):
    """DDPStatefulSet"""
    statefulset_name = f"ddp-{job_id}"
    
    # 
    env = [
        client.V1EnvVar(name="RANK", valueFrom=client.V1EnvVarSource(
            field_ref=client.V1ObjectFieldSelector(field_path="metadata.name")
        )),
        client.V1EnvVar(name="WORLD_SIZE", value=str(num_replicas)),
        client.V1EnvVar(name="MASTER_ADDR", 
                       value=f"{statefulset_name}-0.{statefulset_name}.default.svc.cluster.local"),
        client.V1EnvVar(name="MASTER_PORT", value="29500"),
        client.V1EnvVar(name="REDIS_HOST", value="redis-service"),
        client.V1EnvVar(name="REDIS_PORT", value="6379")
    ]
    
    # 
    container = client.V1Container(
        name="trainer",
        image=image,
        command=["python", "/app/ddp_train.py"],
        env=env,
        volume_mounts=[client.V1VolumeMount(
            name="shared-storage",
            mount_path="/shared/checkpoints"
        )],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"}
        )
    )
    
    # Pod
    pod_spec = client.V1PodSpec(
        containers=[container],
        volumes=[client.V1Volume(
            name="shared-storage",
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name="shared-pvc"
            )
        )],
        restart_policy="OnFailure"
    )
    
    # StatefulSet
    statefulset = client.V1StatefulSet(
        metadata=client.V1ObjectMeta(
            name=statefulset_name,
            labels={
                "app": statefulset_name,
                "hermes-job": job_id,
                "hermes-type": "ddp"
            }
        ),
        spec=client.V1StatefulSetSpec(
            replicas=num_replicas,
            service_name=statefulset_name,
            selector=client.V1LabelSelector(
                match_labels={"app": statefulset_name}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": statefulset_name}),
                spec=pod_spec
            ),
            volume_claim_templates=[
                client.V1PersistentVolumeClaim(
                    metadata=client.V1ObjectMeta(name="shared-storage"),
                    spec=client.V1PersistentVolumeClaimSpec(
                        access_modes=["ReadWriteMany"],
                        resources=client.V1ResourceRequirements(
                            requests={"storage": "10Gi"}
                        )
                    )
                )
            ]
        )
    )
    
    # Headless Service
    service = client.V1Service(
        metadata=client.V1ObjectMeta(name=statefulset_name),
        spec=client.V1ServiceSpec(
            cluster_ip=None,
            selector={"app": statefulset_name},
            ports=[client.V1ServicePort(port=29500, name="ddp")]
        )
    )
    
    if v1 is None or apps_v1 is None:
        print(f"[DDP] Running in standalone mode, skipping StatefulSet creation for {statefulset_name}")
        return statefulset_name

    v1.create_namespaced_service(namespace="default", body=service)
    apps_v1.create_namespaced_stateful_set(namespace="default", body=statefulset)
    
    return statefulset_name

@app.post("/ddp/jobs/{job_id}/recover")
async def recover_ddp_job(job_id: str):
    """"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    statefulset_name = job["statefulset_name"]
    
    start_time = time.time()
    
    try:
        # 1. 
        pause_training(job_id)
        
        # 2. StatefulSetPod
        if apps_v1 is not None:
            apps_v1.delete_namespaced_stateful_set(
                name=statefulset_name,
                namespace="default",
                body=client.V1DeleteOptions(grace_period_seconds=0)
            )
        
        # 3. Pod
        await asyncio.sleep(5)
        
        # 4. StatefulSet
        create_ddp_statefulset(
            job_id, 
            job["name"], 
            job["num_replicas"], 
            "hermes-ddp:latest"
        )
        
        # 5. 
        resume_training(job_id)
        
        recovery_time = (time.time() - start_time) * 1000
        
        job["status"] = "RECOVERED"
        job["recoveries"] += 1
        job["last_recovery_time"] = recovery_time
        job["updated_at"] = time.time()
        
        print(f"[DDP]  {job_id} : {recovery_time:.0f}ms")
        
        return {
            "message": "DDP job recovered",
            "job_id": job_id,
            "recovery_time_ms": recovery_time,
            "total_recoveries": job["recoveries"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recovery failed: {str(e)}")

def pause_training(job_id):
    """"""
    import redis
    r = redis.Redis(host='redis-service', port=6379)
    try:
        r.set("training_paused", "true")
        print(f"[DDP] ")
    finally:
        r.close()

def resume_training(job_id):
    """"""
    import redis
    r = redis.Redis(host='redis-service', port=6379)
    try:
        r.set("training_paused", "false")
        print(f"[DDP] ")
    finally:
        r.close()

@app.get("/ddp/jobs/{job_id}")
async def get_ddp_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Pod
    try:
        pods = v1.list_namespaced_pod(
            namespace="default",
            label_selector=f"hermes-job={job_id}"
        )
        job["pod_status"] = [
            {"name": p.metadata.name, "phase": p.status.phase}
            for p in pods.items
        ]
    except Exception:
        job["pod_status"] = []
    
    return job

@app.get("/ddp/jobs")
async def list_ddp_jobs():
    return {"total": len(jobs), "jobs": list(jobs.values())}

@app.delete("/ddp/jobs/{job_id}")
async def delete_ddp_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    try:
        # StatefulSet
        apps_v1.delete_namespaced_stateful_set(
            name=job["statefulset_name"],
            namespace="default"
        )
        # Service
        v1.delete_namespaced_service(
            name=job["statefulset_name"],
            namespace="default"
        )
    except Exception:
        pass
    
    del jobs[job_id]
    return {"message": "DDP job deleted", "job_id": job_id}

# DDP
def start_ddp_fault_watcher():
    """DDPPod"""
    w = watch.Watch()
    try:
        for event in w.stream(v1.list_namespaced_pod, namespace="default", label_selector="hermes-type=ddp"):
            if event['type'] == 'DELETED':
                pod = event['object']
                job_id = pod.metadata.labels.get('hermes-job')
                if job_id and job_id in jobs:
                    print(f"[DDP FAULT] Pod {pod.metadata.name} deleted, recovering job {job_id}")
                    # 5
                    async def _delayed_recovery():
                        await asyncio.sleep(5)
                        await recover_ddp_job(job_id)
                    loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        _delayed_recovery(),
                        loop
                    )
    except Exception as e:
        print(f"[DDP ERROR] Fault watcher error: {e}")

@app.on_event("startup")
async def startup_event():
    """"""
    print("[DDP Scheduler] Starting Hermes Distributed Scheduler")
    # 
    threading.Thread(target=start_ddp_fault_watcher, daemon=True).start()
    print("[DDP Scheduler] DDP fault watcher started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
