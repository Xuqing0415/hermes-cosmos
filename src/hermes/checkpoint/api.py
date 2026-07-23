"""
Checkpoint API endpoints
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from hermes.core.models import Checkpoint, CheckpointState

router = APIRouter()


class CreateCheckpointRequest(BaseModel):
    job_id: UUID
    tenant_id: str
    is_delta: bool = Field(default=False, description="Create delta checkpoint")
    metadata: dict[str, str] = Field(default_factory=dict)


class CheckpointResponse(BaseModel):
    checkpoint: Checkpoint


class CheckpointListResponse(BaseModel):
    checkpoints: list[Checkpoint]
    total: int
    page: int
    page_size: int


class RestoreRequest(BaseModel):
    target_job_id: Optional[UUID] = Field(default=None, description="Target job ID for restore")


@router.post(
    "/",
    response_model=CheckpointResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create checkpoint",
)
async def create_checkpoint(request: CreateCheckpointRequest) -> CheckpointResponse:
    checkpoint = Checkpoint(
        job_id=request.job_id,
        tenant_id=request.tenant_id,
        state=CheckpointState.CREATING,
        is_delta=request.is_delta,
        metadata=request.metadata,
    )

    return CheckpointResponse(checkpoint=checkpoint)


@router.get(
    "/",
    response_model=CheckpointListResponse,
    summary="List checkpoints",
)
async def list_checkpoints(
    job_id: Optional[UUID] = Query(None),
    tenant_id: Optional[str] = Query(None),
    state: Optional[CheckpointState] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> CheckpointListResponse:
    return CheckpointListResponse(
        checkpoints=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{checkpoint_id}",
    response_model=CheckpointResponse,
    summary="Get checkpoint",
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


@router.delete(
    "/{checkpoint_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete checkpoint",
)
async def delete_checkpoint(checkpoint_id: UUID) -> None:
    pass


@router.post(
    "/{checkpoint_id}/restore",
    response_model=CheckpointResponse,
    summary="Restore from checkpoint",
)
async def restore_checkpoint(
    checkpoint_id: UUID,
    request: RestoreRequest,
) -> CheckpointResponse:
    checkpoint = Checkpoint(
        id=checkpoint_id,
        job_id=request.target_job_id or UUID("00000000-0000-0000-0000-000000000001"),
        tenant_id="tenant-1",
        state=CheckpointState.COMPLETED,
    )

    return CheckpointResponse(checkpoint=checkpoint)


@router.get(
    "/{checkpoint_id}/metadata",
    summary="Get checkpoint metadata",
)
async def get_checkpoint_metadata(checkpoint_id: UUID) -> dict:
    return {
        "checkpoint_id": str(checkpoint_id),
        "size_bytes": 1024 * 1024 * 1024,
        "compression_ratio": 0.7,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_delta": False,
    }
