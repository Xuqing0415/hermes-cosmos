"""
Checkpoints API endpoints
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from hermes.core.models import Checkpoint, CheckpointState

router = APIRouter()


class CheckpointListResponse(BaseModel):
    checkpoints: list[Checkpoint]
    total: int
    page: int
    page_size: int


class CheckpointResponse(BaseModel):
    checkpoint: Checkpoint


class CreateCheckpointRequest(BaseModel):
    job_id: UUID
    tenant_id: str
    is_delta: bool = Field(default=False, description="Create delta checkpoint")


@router.get(
    "/",
    response_model=CheckpointListResponse,
    summary="List checkpoints",
    description="Retrieve a paginated list of checkpoints",
)
async def list_checkpoints(
    job_id: Optional[UUID] = Query(None, description="Filter by job ID"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    state: Optional[CheckpointState] = Query(None, description="Filter by state"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
) -> CheckpointListResponse:
    checkpoints: list[Checkpoint] = []
    total = 0

    return CheckpointListResponse(
        checkpoints=checkpoints,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{checkpoint_id}",
    response_model=CheckpointResponse,
    summary="Get checkpoint details",
    description="Retrieve details of a specific checkpoint",
)
async def get_checkpoint(checkpoint_id: UUID) -> CheckpointResponse:
    checkpoint = Checkpoint(
        id=checkpoint_id,
        job_id=UUID("00000000-0000-0000-0000-000000000001"),
        tenant_id="tenant-1",
        state=CheckpointState.COMPLETED,
        size_bytes=1024 * 1024 * 1024,
    )

    return CheckpointResponse(checkpoint=checkpoint)


@router.post(
    "/",
    response_model=CheckpointResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create checkpoint",
    description="Create a new checkpoint for a job",
)
async def create_checkpoint(request: CreateCheckpointRequest) -> CheckpointResponse:
    checkpoint = Checkpoint(
        job_id=request.job_id,
        tenant_id=request.tenant_id,
        state=CheckpointState.CREATING,
        is_delta=request.is_delta,
    )

    return CheckpointResponse(checkpoint=checkpoint)


@router.delete(
    "/{checkpoint_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete checkpoint",
    description="Delete a checkpoint",
)
async def delete_checkpoint(checkpoint_id: UUID) -> None:
    pass


@router.post(
    "/{checkpoint_id}/restore",
    response_model=CheckpointResponse,
    summary="Restore checkpoint",
    description="Restore a job from a checkpoint",
)
async def restore_checkpoint(checkpoint_id: UUID) -> CheckpointResponse:
    checkpoint = Checkpoint(
        id=checkpoint_id,
        job_id=UUID("00000000-0000-0000-0000-000000000001"),
        tenant_id="tenant-1",
        state=CheckpointState.COMPLETED,
    )

    return CheckpointResponse(checkpoint=checkpoint)
