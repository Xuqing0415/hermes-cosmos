"""
Inference Service API endpoints
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from hermes.core.models import (
    InferenceService,
    InferenceRequest,
    InferenceResponse,
    InferenceMetrics,
    InferenceConfig,
    HealthCheckConfig,
    AutoscalingConfig,
    JobRequirements,
    PlacementConstraints,
    JobStatus,
    JobPriority,
)

router = APIRouter()


class CreateInferenceServiceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    priority: JobPriority = Field(default=JobPriority.NORMAL)
    requirements: JobRequirements
    constraints: PlacementConstraints = Field(default_factory=PlacementConstraints)
    inference_config: InferenceConfig
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    image: str = Field(..., min_length=1)
    command: Optional[str] = None
    environment: dict[str, str] = Field(default_factory=dict)
    volumes: list[str] = Field(default_factory=list)
    replicas: int = Field(default=1, ge=1)
    sla_availability: float = Field(default=99.9, ge=0.0, le=100.0)
    sla_latency_p99_ms: float = Field(default=100.0, ge=1.0)


class UpdateInferenceServiceRequest(BaseModel):
    priority: Optional[JobPriority] = None
    requirements: Optional[JobRequirements] = None
    constraints: Optional[PlacementConstraints] = None
    inference_config: Optional[InferenceConfig] = None
    autoscaling: Optional[AutoscalingConfig] = None
    replicas: Optional[int] = Field(None, ge=1)


class InferenceServiceListResponse(BaseModel):
    services: list[InferenceService]
    total: int
    page: int
    page_size: int


class InferenceServiceResponse(BaseModel):
    service: InferenceService


class ScaleRequest(BaseModel):
    replicas: int = Field(..., ge=1, description="Target number of replicas")


# In-memory storage for demonstration
inference_services: dict[UUID, InferenceService] = {}


@router.post(
    "/",
    response_model=InferenceServiceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new inference service",
    description="Deploy a new LLM inference service",
)
async def create_inference_service(
    request: CreateInferenceServiceRequest,
) -> InferenceServiceResponse:
    service = InferenceService(
        name=request.name,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        priority=request.priority,
        requirements=request.requirements,
        constraints=request.constraints,
        inference_config=request.inference_config,
        health_check=request.health_check,
        autoscaling=request.autoscaling,
        image=request.image,
        command=request.command,
        environment=request.environment,
        volumes=request.volumes,
        replicas=request.replicas,
        sla_availability=request.sla_availability,
        sla_latency_p99_ms=request.sla_latency_p99_ms,
    )
    inference_services[service.id] = service
    return InferenceServiceResponse(service=service)


@router.get(
    "/",
    response_model=InferenceServiceListResponse,
    summary="List all inference services",
    description="Retrieve a paginated list of inference services",
)
async def list_inference_services(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
) -> InferenceServiceListResponse:
    services = list(inference_services.values())
    
    if tenant_id:
        services = [s for s in services if s.tenant_id == tenant_id]
    if status:
        services = [s for s in services if s.status == status]
    
    total = len(services)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_services = services[start:end]
    
    return InferenceServiceListResponse(
        services=paginated_services,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{service_id}",
    response_model=InferenceServiceResponse,
    summary="Get inference service details",
    description="Retrieve details of a specific inference service",
)
async def get_inference_service(service_id: UUID) -> InferenceServiceResponse:
    service = inference_services.get(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference service {service_id} not found",
        )
    return InferenceServiceResponse(service=service)


@router.put(
    "/{service_id}",
    response_model=InferenceServiceResponse,
    summary="Update inference service",
    description="Update inference service configuration",
)
async def update_inference_service(
    service_id: UUID,
    request: UpdateInferenceServiceRequest,
) -> InferenceServiceResponse:
    service = inference_services.get(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference service {service_id} not found",
        )
    
    if request.priority:
        service.priority = request.priority
    if request.requirements:
        service.requirements = request.requirements
    if request.constraints:
        service.constraints = request.constraints
    if request.inference_config:
        service.inference_config = request.inference_config
    if request.autoscaling:
        service.autoscaling = request.autoscaling
    if request.replicas:
        service.replicas = request.replicas
    
    service.updated_at = datetime.now(timezone.utc)
    return InferenceServiceResponse(service=service)


@router.delete(
    "/{service_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete inference service",
    description="Undeploy an inference service",
)
async def delete_inference_service(service_id: UUID) -> None:
    if service_id not in inference_services:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference service {service_id} not found",
        )
    del inference_services[service_id]


@router.post(
    "/{service_id}/scale",
    response_model=InferenceServiceResponse,
    summary="Scale inference service",
    description="Scale the number of replicas for an inference service",
)
async def scale_inference_service(
    service_id: UUID,
    request: ScaleRequest,
) -> InferenceServiceResponse:
    service = inference_services.get(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference service {service_id} not found",
        )
    
    service.replicas = request.replicas
    service.updated_at = datetime.now(timezone.utc)
    return InferenceServiceResponse(service=service)


@router.post(
    "/{service_id}/predict",
    response_model=InferenceResponse,
    summary="Run inference",
    description="Send a prediction request to the inference service",
)
async def predict(
    service_id: UUID,
    request: InferenceRequest,
) -> InferenceResponse:
    service = inference_services.get(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference service {service_id} not found",
        )
    
    if service.status != JobStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference service is not running (status: {service.status})",
        )
    
    # This is a placeholder - in a real implementation, this would proxy to the actual inference service
    start_time = datetime.now(timezone.utc)
    response_text = f"Generated response for: {request.prompt[:50]}..."
    latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    
    return InferenceResponse(
        text=response_text,
        finish_reason="stop",
        prompt_tokens=len(request.prompt.split()),
        completion_tokens=len(response_text.split()),
        total_tokens=len(request.prompt.split()) + len(response_text.split()),
        latency_ms=latency_ms,
    )


@router.get(
    "/{service_id}/metrics",
    response_model=InferenceMetrics,
    summary="Get inference service metrics",
    description="Retrieve performance metrics for an inference service",
)
async def get_inference_metrics(service_id: UUID) -> InferenceMetrics:
    service = inference_services.get(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference service {service_id} not found",
        )
    
    # This is a placeholder - in a real implementation, this would fetch from Prometheus or similar
    return InferenceMetrics(
        requests_total=1000,
        requests_per_second=50.0,
        latency_p50_ms=50.0,
        latency_p95_ms=80.0,
        latency_p99_ms=95.0,
        gpu_utilization=75.0,
        error_rate=0.1,
    )
