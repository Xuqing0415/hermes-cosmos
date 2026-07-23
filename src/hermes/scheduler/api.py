"""
Scheduler API endpoints
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel

from hermes.core.config import Region
from hermes.core.models import Job, JobStatus

router = APIRouter()


class SchedulerStatusResponse(BaseModel):
    status: str
    is_leader: bool
    term: int
    committed_index: int
    last_heartbeat: Optional[datetime]


class ClusterStateResponse(BaseModel):
    total_gpus: int
    available_gpus: int
    running_jobs: int
    pending_jobs: int
    regions: dict[str, dict[str, int]]


class PlacementResponse(BaseModel):
    job_id: UUID
    region: Region
    gpu_count: int
    node_ids: list[str]
    scheduled_at: datetime


@router.get("/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status() -> SchedulerStatusResponse:
    return SchedulerStatusResponse(
        status="running",
        is_leader=True,
        term=1,
        committed_index=100,
        last_heartbeat=datetime.now(timezone.utc),
    )


@router.get("/cluster", response_model=ClusterStateResponse)
async def get_cluster_state() -> ClusterStateResponse:
    return ClusterStateResponse(
        total_gpus=10000,
        available_gpus=6000,
        running_jobs=100,
        pending_jobs=50,
        regions={
            "us-east": {"total": 4000, "available": 2500, "jobs": 40},
            "us-west": {"total": 3000, "available": 2000, "jobs": 30},
            "eu-west": {"total": 2000, "available": 1000, "jobs": 20},
            "asia-east": {"total": 1000, "available": 500, "jobs": 10},
        },
    )


@router.post("/jobs/{job_id}/preempt")
async def preempt_job(job_id: UUID, reason: str = Query(...)) -> dict:
    return {
        "job_id": str(job_id),
        "status": "preempted",
        "reason": reason,
        "preempted_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/jobs/{job_id}/migrate")
async def migrate_job(
    job_id: UUID,
    target_region: Region = Query(...),
) -> dict:
    return {
        "job_id": str(job_id),
        "status": "migrating",
        "target_region": target_region.value,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/jobs/{job_id}/placement", response_model=PlacementResponse)
async def get_job_placement(job_id: UUID) -> PlacementResponse:
    return PlacementResponse(
        job_id=job_id,
        region=Region.US_EAST,
        gpu_count=8,
        node_ids=["node-1", "node-2", "node-3", "node-4"],
        scheduled_at=datetime.now(timezone.utc),
    )


class InferencePlacementResponse(BaseModel):
    service_id: UUID
    region: Region
    gpu_count: int
    replicas: int
    node_ids: list[str]
    endpoint: str
    scheduled_at: datetime


@router.post("/inference/{service_id}/schedule")
async def schedule_inference_service(service_id: UUID) -> InferencePlacementResponse:
    return InferencePlacementResponse(
        service_id=service_id,
        region=Region.US_EAST,
        gpu_count=4,
        replicas=2,
        node_ids=["node-10", "node-11"],
        endpoint=f"http://inference-{service_id}.hermes.svc.cluster.local:8000",
        scheduled_at=datetime.now(timezone.utc),
    )


@router.get("/inference/{service_id}/status")
async def get_inference_service_status(service_id: UUID) -> dict:
    return {
        "service_id": str(service_id),
        "status": "running",
        "replicas": 2,
        "available_replicas": 2,
        "region": Region.US_EAST.value,
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/inference/{service_id}/scale")
async def scale_inference_service(
    service_id: UUID,
    replicas: int = Query(..., ge=1),
) -> dict:
    return {
        "service_id": str(service_id),
        "status": "scaling",
        "target_replicas": replicas,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
