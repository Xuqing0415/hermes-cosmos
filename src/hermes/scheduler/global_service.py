"""
Global scheduler service - in-memory scheduler for development and testing
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional
from uuid import UUID

import structlog
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from hermes.core.config import Region, SchedulerConfig
from hermes.core.models import Job, JobPriority, JobRequirements, JobStatus, PlacementConstraints
from hermes.core.store import get_job_queue, get_resource_manager
from hermes.scheduler.algorithms.placement import PlacementEngine

logger = structlog.get_logger()


class CreateJobRequest(BaseModel):
    name: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    priority: JobPriority = JobPriority.NORMAL
    requirements: JobRequirements = Field(default_factory=JobRequirements)
    constraints: PlacementConstraints = Field(default_factory=PlacementConstraints)
    image: str = Field(..., min_length=1)
    command: Optional[str] = None


class JobResponse(BaseModel):
    job: Job


class JobListResponse(BaseModel):
    jobs: list[Job]
    total: int


class HealthResponse(BaseModel):
    status: str


class SchedulerStatusResponse(BaseModel):
    status: str
    is_leader: bool
    term: int
    pending_jobs: int
    running_jobs: int


class GlobalSchedulerService:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.job_queue = get_job_queue()
        self.resource_manager = get_resource_manager()
        self.placement_engine = PlacementEngine(
            carbon_aware=config.algorithm.carbon_aware,
        )

    async def submit_job(self, request: CreateJobRequest) -> Job:
        job = Job(
            name=request.name,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            priority=request.priority,
            requirements=request.requirements,
            constraints=request.constraints,
            image=request.image,
            command=request.command,
            status=JobStatus.PENDING,
        )

        region = await self.resource_manager.find_available_region(
            gpu_count=job.requirements.gpu_count,
            preferred_regions=job.constraints.regions,
            carbon_aware=job.constraints.carbon_aware,
        )

        if region:
            allocated = await self.resource_manager.allocate(
                job, region, job.requirements.gpu_count
            )
            if allocated:
                job.region = region
                job.status = JobStatus.SCHEDULING

        await self.job_queue.enqueue(job)
        return job

    async def get_job(self, job_id: UUID) -> Optional[Job]:
        return await self.job_queue.get_job(job_id)

    async def list_jobs(self) -> list[Job]:
        return await self.job_queue.get_all_jobs()

    async def get_cluster_summary(self) -> dict[str, Any]:
        return await self.resource_manager.get_cluster_state()

    async def get_status(self) -> dict[str, Any]:
        all_jobs = await self.job_queue.get_all_jobs()
        pending = len([j for j in all_jobs if j.status == JobStatus.PENDING])
        running = len([j for j in all_jobs if j.status in (JobStatus.RUNNING, JobStatus.SCHEDULING)])
        return {
            "status": "running",
            "is_leader": True,
            "term": 1,
            "pending_jobs": pending,
            "running_jobs": running,
        }


def create_app(config: Optional[SchedulerConfig] = None) -> FastAPI:
    scheduler_config = config or SchedulerConfig()
    service = GlobalSchedulerService(scheduler_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Global scheduler service started")
        yield
        logger.info("Global scheduler service stopped")

    app = FastAPI(
        title="Hermes Global Scheduler",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="healthy")

    @app.get("/status", response_model=SchedulerStatusResponse)
    async def status_endpoint() -> SchedulerStatusResponse:
        data = await service.get_status()
        return SchedulerStatusResponse(**data)

    @app.get("/cluster/summary")
    async def cluster_summary() -> dict[str, Any]:
        return await service.get_cluster_summary()

    @app.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
    async def create_job(request: CreateJobRequest) -> JobResponse:
        job = await service.submit_job(request)
        return JobResponse(job=job)

    @app.get("/jobs", response_model=JobListResponse)
    async def list_jobs() -> JobListResponse:
        jobs = await service.list_jobs()
        return JobListResponse(jobs=jobs, total=len(jobs))

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: UUID) -> JobResponse:
        job = await service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobResponse(job=job)

    app.state.scheduler_service = service
    return app


def main() -> None:
    import uvicorn

    config = SchedulerConfig()
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)
