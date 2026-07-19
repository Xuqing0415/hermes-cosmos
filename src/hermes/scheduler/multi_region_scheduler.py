"""
Hermes Multi-Region Scheduler - Region
K8s
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

app = FastAPI(title="Hermes Multi-Region Scheduler", version="3.0")

# 
clusters = {
    "us-east": {
        "name": "us-east",
        "region": "us-east",
        "carbon_intensity": 45,
        "gpu_capacity": 200,
        "gpu_available": 180,
        "client": None,
        "api_client": None
    },
    "eu-west": {
        "name": "eu-west",
        "region": "eu-west",
        "carbon_intensity": 25,
        "gpu_capacity": 180,
        "gpu_available": 160,
        "client": None,
        "api_client": None
    },
    "asia-east": {
        "name": "asia-east",
        "region": "asia-east",
        "carbon_intensity": 60,
        "gpu_capacity": 150,
        "gpu_available": 130,
        "client": None,
        "api_client": None
    }
}

# 
jobs: Dict[str, dict] = {}

class MultiRegionJobRequest(BaseModel):
    name: str
    tenant_id: str
    user_id: str
    total_replicas: int = 4
    regions: List[str] = ["us-east", "eu-west"]
    data_residency: Optional[str] = None  # GDPR
    image: str = "hermes-ddp:latest"
    checkpoint_interval: int = 10
    priority: str = "normal"

@app.on_event("startup")
async def startup_event():
    """"""
    print("[Multi-Region] ...")
    
    # kubeconfig
    for cluster_name, cluster_info in clusters.items():
        try:
            # kubeconfig
            kubeconfig_path = f"/etc/hermes/kubeconfig/{cluster_name}"
            config.load_kube_config(config_file=kubeconfig_path)
            
            cluster_info["client"] = client.CoreV1Api()
            cluster_info["api_client"] = client.ApiClient()
            
            print(f"[Multi-Region]  {cluster_name} ")
        except Exception as e:
            print(f"[Multi-Region]  {cluster_name} : {e}")
    
    # 
    threading.Thread(target=start_global_fault_watcher, daemon=True).start()
    print("[Multi-Region] ")

@app.get("/health")
async def health_check():
    """"""
    cluster_status = {}
    for name, info in clusters.items():
        cluster_status[name] = "connected" if info["client"] else "disconnected"
    
    return {
        "status": "ok",
        "service": "multi-region-scheduler",
        "clusters": cluster_status
    }

@app.post("/multi-region/jobs")
async def submit_multi_region_job(job: MultiRegionJobRequest):
    """Region"""
    job_id = str(uuid4())[:8]
    
    # 
    if job.data_residency:
        if job.data_residency not in job.regions:
            raise HTTPException(
                status_code=400,
                detail=f"Data residency violation: data must stay in {job.data_residency}"
            )
    
    # RegionPod
    replicas_per_region = job.total_replicas // len(job.regions)
    region_assignments = {}
    
    for region in job.regions:
        if region not in clusters or not clusters[region]["client"]:
            raise HTTPException(status_code=400, detail=f"Region {region} not available")
        
        region_assignments[region] = replicas_per_region
    
    # 
    if job.priority == "carbon-aware":
        sorted_regions = sorted(job.regions, key=lambda r: clusters[r]["carbon_intensity"])
        # 
        for i, region in enumerate(sorted_regions):
            if i == 0:
                region_assignments[region] += job.total_replicas % len(job.regions)
    
    # Region Pod
    pod_addresses = []
    rank = 0
    
    for region, replica_count in region_assignments.items():
        for i in range(replica_count):
            pod_name = f"hermes-{job_id}-{region}-{i}"
            pod_ip = await create_cross_region_pod(
                job_id, pod_name, region, rank, 
                job.total_replicas, job.image
            )
            pod_addresses.append(f"{rank}:{pod_ip}")
            rank += 1
    
    job_data = {
        "job_id": job_id,
        "name": job.name,
        "tenant_id": job.tenant_id,
        "user_id": job.user_id,
        "status": "RUNNING",
        "total_replicas": job.total_replicas,
        "regions": job.regions,
        "region_assignments": region_assignments,
        "pod_addresses": pod_addresses,
        "data_residency": job.data_residency,
        "created_at": time.time(),
        "updated_at": time.time(),
        "recoveries": 0,
        "region_failures": {}
    }
    jobs[job_id] = job_data
    
    print(f"[Multi-Region]  {job_id} ")
    print(f"  Regions: {job.regions}")
    print(f"  Assignments: {region_assignments}")
    
    return job_data

async def create_cross_region_pod(job_id, pod_name, region, rank, world_size, image):
    """RegionPod"""
    cluster_info = clusters[region]
    v1 = cluster_info["client"]
    
    # rankRegion
    rank_addrs = ",".join([f"{r}:pending" for r in range(world_size)])
    
    container = client.V1Container(
        name="trainer",
        image=image,
        command=["python", "/app/cross_region_train.py"],
        env=[
            client.V1EnvVar(name="JOB_ID", value=job_id),
            client.V1EnvVar(name="RANK", value=str(rank)),
            client.V1EnvVar(name="WORLD_SIZE", value=str(world_size)),
            client.V1EnvVar(name="REGION", value=region),
            client.V1EnvVar(name="RANK_ADDRS", value=rank_addrs),
            client.V1EnvVar(name="MASTER_ADDR", value="global-redis-service"),
            client.V1EnvVar(name="REDIS_HOST", value="global-redis-service"),
        ],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"}
        )
    )
    
    pod_spec = client.V1PodSpec(
        containers=[container],
        restart_policy="Never"
    )
    
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=pod_name,
            labels={
                "hermes-job": job_id,
                "hermes-region": region,
                "hermes-rank": str(rank)
            }
        ),
        spec=pod_spec
    )
    
    v1.create_namespaced_pod(namespace="default", body=pod)
    
    # PodIP
    await asyncio.sleep(10)
    
    pod_obj = v1.read_namespaced_pod(name=pod_name, namespace="default")
    pod_ip = pod_obj.status.pod_ip or "pending"
    
    return pod_ip

@app.post("/multi-region/jobs/{job_id}/region-failure")
async def handle_region_failure(job_id: str, failed_region: str):
    """Region"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if failed_region not in job["regions"]:
        raise HTTPException(status_code=400, detail=f"Region {failed_region} not in job")
    
    start_time = time.time()
    
    print(f"[Multi-Region] Region {failed_region} ...")
    
    # 1. Region
    job["region_failures"][failed_region] = {
        "timestamp": time.time(),
        "status": "failed"
    }
    
    # 2. RedisCheckpoint
    import redis
    r = redis.Redis(host='global-redis-service', port=6379)
    try:
        latest_step = int(r.get(f"global_checkpoint_{job_id}_latest") or 0)
    finally:
        r.close()
    
    # 3. RegionRegion
    healthy_regions = [r for r in job["regions"] if r != failed_region]
    
    if not healthy_regions:
        raise HTTPException(status_code=500, detail="No healthy regions available")
    
    # 4. RegionPod
    failed_replicas = job["region_assignments"][failed_region]
    new_assignments = job["region_assignments"].copy()
    del new_assignments[failed_region]
    
    # Region
    replicas_per_region = failed_replicas // len(healthy_regions)
    for region in healthy_regions:
        new_assignments[region] += replicas_per_region
    
    # 5. Pod
    new_pod_addresses = []
    rank = 0
    
    for region, replica_count in new_assignments.items():
        for i in range(replica_count):
            pod_name = f"hermes-{job_id}-{region}-recovered-{i}"
            pod_ip = await create_cross_region_pod(
                job_id, pod_name, region, rank,
                sum(new_assignments.values()), "hermes-ddp:latest"
            )
            new_pod_addresses.append(f"{rank}:{pod_ip}")
            rank += 1
    
    # 6. 
    job["region_assignments"] = new_assignments
    job["pod_addresses"] = new_pod_addresses
    job["regions"] = healthy_regions
    job["status"] = "RECOVERED"
    job["recoveries"] += 1
    job["updated_at"] = time.time()
    
    recovery_time = (time.time() - start_time) * 1000
    
    print(f"[Multi-Region] Region: {recovery_time:.0f}ms")
    
    return {
        "message": f"Region {failed_region} failure recovered",
        "job_id": job_id,
        "recovery_time_ms": recovery_time,
        "new_regions": healthy_regions,
        "new_assignments": new_assignments
    }

@app.get("/multi-region/jobs/{job_id}")
async def get_multi_region_job(job_id: str):
    """Region"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # RegionPod
    region_pod_status = {}
    for region in job["regions"]:
        if clusters[region]["client"]:
            pods = clusters[region]["client"].list_namespaced_pod(
                namespace="default",
                label_selector=f"hermes-job={job_id},hermes-region={region}"
            )
            region_pod_status[region] = [
                {"name": p.metadata.name, "phase": p.status.phase, "ip": p.status.pod_ip}
                for p in pods.items
            ]
    
    job["region_pod_status"] = region_pod_status
    return job

@app.get("/multi-region/clusters")
async def list_clusters():
    """"""
    result = []
    for name, info in clusters.items():
        result.append({
            "name": name,
            "region": info["region"],
            "carbon_intensity": info["carbon_intensity"],
            "gpu_capacity": info["gpu_capacity"],
            "gpu_available": info["gpu_available"],
            "status": "connected" if info["client"] else "disconnected"
        })
    return {"clusters": result}

def start_global_fault_watcher():
    """"""
    print("[Multi-Region] ...")
    
    while True:
        try:
            for region, info in clusters.items():
                if info["client"]:
                    # 
                    try:
                        nodes = info["client"].list_node()
                        if len(nodes.items) == 0:
                            print(f"[Multi-Region] Region {region} ")
                    except Exception as e:
                        print(f"[Multi-Region] Region {region} : {e}")
            
            time.sleep(30)  # 30
        except Exception as e:
            print(f"[Multi-Region] : {e}")
            time.sleep(60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
