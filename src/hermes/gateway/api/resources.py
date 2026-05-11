"""
Resources API endpoints
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel

from hermes.core.config import Region, GPUType
from hermes.core.models import Resource, ResourceType, GPUResource

router = APIRouter()


class ResourceListResponse(BaseModel):
    resources: list[Resource]
    total: int
    page: int
    page_size: int


class ResourceResponse(BaseModel):
    resource: Resource


class ResourceUtilizationResponse(BaseModel):
    resource_id: UUID
    resource_name: str
    resource_type: ResourceType
    total_capacity: int
    allocated_capacity: int
    available_capacity: int
    utilization: float


class ClusterSummaryResponse(BaseModel):
    total_gpus: int
    available_gpus: int
    total_jobs: int
    running_jobs: int
    pending_jobs: int
    regions: dict[str, int]


@router.get(
    "/",
    response_model=ResourceListResponse,
    summary="List resources",
    description="Retrieve a paginated list of available resources",
)
async def list_resources(
    region: Optional[Region] = Query(None, description="Filter by region"),
    gpu_type: Optional[GPUType] = Query(None, description="Filter by GPU type"),
    resource_type: Optional[ResourceType] = Query(None, description="Filter by resource type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
) -> ResourceListResponse:
    resources: list[Resource] = []
    total = 0

    return ResourceListResponse(
        resources=resources,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{resource_id}",
    response_model=ResourceResponse,
    summary="Get resource details",
    description="Retrieve details of a specific resource",
)
async def get_resource(resource_id: UUID) -> ResourceResponse:
    resource = GPUResource(
        id=resource_id,
        name="gpu-cluster-1",
        region=Region.US_EAST,
        gpu_type=GPUType.NVIDIA_H100,
        total_capacity=1000,
        available_capacity=500,
        allocated_capacity=500,
        cost_per_hour=3.50,
    )

    return ResourceResponse(resource=resource)


@router.get(
    "/{resource_id}/utilization",
    response_model=ResourceUtilizationResponse,
    summary="Get resource utilization",
    description="Retrieve utilization metrics for a resource",
)
async def get_resource_utilization(resource_id: UUID) -> ResourceUtilizationResponse:
    return ResourceUtilizationResponse(
        resource_id=resource_id,
        resource_name="gpu-cluster-1",
        resource_type=ResourceType.GPU,
        total_capacity=1000,
        allocated_capacity=500,
        available_capacity=500,
        utilization=0.5,
    )


@router.get(
    "/cluster/summary",
    response_model=ClusterSummaryResponse,
    summary="Get cluster summary",
    description="Retrieve overall cluster resource summary",
)
async def get_cluster_summary() -> ClusterSummaryResponse:
    return ClusterSummaryResponse(
        total_gpus=10000,
        available_gpus=6000,
        total_jobs=150,
        running_jobs=100,
        pending_jobs=50,
        regions={
            "us-east": 4000,
            "us-west": 3000,
            "eu-west": 2000,
            "asia-east": 1000,
        },
    )
