"""
Jobs API endpoints
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from hermes.core.models import Job, JobStatus, JobPriority, JobRequirements, PlacementConstraints

router = APIRouter()


class CreateJobRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    priority: JobPriority = Field(default=JobPriority.NORMAL)
    requirements: JobRequirements
    constraints: PlacementConstraints = Field(default_factory=PlacementConstraints)
    framework: str = Field(default="pytorch")
    command: Optional[str] = None
    image: str = Field(..., min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)
    volumes: list[str] = Field(default_factory=list)
    checkpoint_enabled: bool = Field(default=True)
    metadata: dict[str, str] = Field(default_factory=dict)


class UpdateJobRequest(BaseModel):
    priority: Optional[JobPriority] = None
    requirements: Optional[JobRequirements] = None
    constraints: Optional[PlacementConstraints] = None
    metadata: Optional[dict[str, str]] = None


class JobListResponse(BaseModel):
    jobs: list[Job]
    total: int
    page: int
    page_size: int


class JobResponse(BaseModel):
    job: Job


@router.post(
    "/",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new training job",
    description="Submit a new AI training job to the scheduler",
)
async def create_job(request: CreateJobRequest) -> JobResponse:
    job = Job(
        name=request.name,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        priority=request.priority,
        requirements=request.requirements,
        constraints=request.constraints,
        framework=request.framework,
        command=request.command,
        image=request.image,
        environment=request.environment,
        volumes=request.volumes,
        checkpoint_enabled=request.checkpoint_enabled,
        metadata=request.metadata,
    )

    return JobResponse(job=job)


@router.get(
    "/",
    response_model=JobListResponse,
    summary="List all jobs",
    description="Retrieve a paginated list of training jobs",
)
async def list_jobs(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
) -> JobListResponse:
    jobs: list[Job] = []
    total = 0

    return JobListResponse(
        jobs=jobs,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job details",
    description="Retrieve details of a specific training job",
)
async def get_job(job_id: UUID) -> JobResponse:
    job = Job(
        id=job_id,
        name="example-job",
        tenant_id="tenant-1",
        user_id="user-1",
        status=JobStatus.RUNNING,
        requirements=JobRequirements(),
    )

    return JobResponse(job=job)


@router.put(
    "/{job_id}",
    response_model=JobResponse,
    summary="Update job",
    description="Update job configuration",
)
async def update_job(job_id: UUID, request: UpdateJobRequest) -> JobResponse:
    job = Job(
        id=job_id,
        name="example-job",
        tenant_id="tenant-1",
        user_id="user-1",
        status=JobStatus.RUNNING,
        requirements=JobRequirements(),
    )

    if request.priority:
        job.priority = request.priority
    if request.requirements:
        job.requirements = request.requirements
    if request.constraints:
        job.constraints = request.constraints
    if request.metadata:
        job.metadata.update(request.metadata)

    job.updated_at = datetime.utcnow()

    return JobResponse(job=job)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel job",
    description="Cancel a running or pending job",
)
async def cancel_job(job_id: UUID) -> None:
    pass


@router.post(
    "/{job_id}/checkpoint",
    response_model=JobResponse,
    summary="Trigger checkpoint",
    description="Manually trigger a checkpoint for a running job",
)
async def trigger_checkpoint(job_id: UUID) -> JobResponse:
    job = Job(
        id=job_id,
        name="example-job",
        tenant_id="tenant-1",
        user_id="user-1",
        status=JobStatus.CHECKPOINTING,
        requirements=JobRequirements(),
    )

    return JobResponse(job=job)


@router.post(
    "/{job_id}/migrate",
    response_model=JobResponse,
    summary="Migrate job",
    description="Migrate a running job to a different region or resource pool",
)
async def migrate_job(
    job_id: UUID,
    target_region: str = Query(..., description="Target region for migration"),
) -> JobResponse:
    job = Job(
        id=job_id,
        name="example-job",
        tenant_id="tenant-1",
        user_id="user-1",
        status=JobStatus.MIGRATING,
        requirements=JobRequirements(),
    )

    return JobResponse(job=job)
