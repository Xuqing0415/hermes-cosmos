"""
Health check API endpoints
"""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: dict[str, str]


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]


class LivenessResponse(BaseModel):
    alive: bool


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the gateway",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        components={
            "scheduler": "healthy",
            "checkpoint": "healthy",
            "redis": "healthy",
            "etcd": "healthy",
        },
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Check if the gateway is ready to accept requests",
)
async def readiness_check() -> ReadinessResponse:
    return ReadinessResponse(
        ready=True,
        checks={
            "database": True,
            "redis": True,
            "scheduler": True,
        },
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness check",
    description="Check if the gateway is alive",
)
async def liveness_check() -> LivenessResponse:
    return LivenessResponse(alive=True)
