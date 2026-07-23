"""
Agent API endpoints
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel

from hermes.core.config import Region
from hermes.core.models import FaultPrediction

router = APIRouter()


class AgentStatusResponse(BaseModel):
    status: str
    region: Region
    uptime_seconds: int
    predictions_made: int
    healing_actions: int
    last_prediction: Optional[datetime]


class GPUMetricsResponse(BaseModel):
    gpu_id: str
    utilization: float
    memory_used: int
    memory_total: int
    temperature: float
    power_draw: float
    power_limit: float


class NodeMetricsResponse(BaseModel):
    cpu_utilization: float
    memory_used: int
    memory_total: int
    network_rx_bytes: int
    network_tx_bytes: int
    disk_used: int
    disk_total: int


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status() -> AgentStatusResponse:
    return AgentStatusResponse(
        status="running",
        region=Region.US_EAST,
        uptime_seconds=3600,
        predictions_made=10,
        healing_actions=2,
        last_prediction=datetime.now(timezone.utc),
    )


@router.get("/metrics/gpu", response_model=list[GPUMetricsResponse])
async def get_gpu_metrics() -> list[GPUMetricsResponse]:
    return [
        GPUMetricsResponse(
            gpu_id="gpu-0",
            utilization=85.5,
            memory_used=70 * 1024 * 1024 * 1024,
            memory_total=80 * 1024 * 1024 * 1024,
            temperature=75.0,
            power_draw=350.0,
            power_limit=400.0,
        ),
        GPUMetricsResponse(
            gpu_id="gpu-1",
            utilization=82.3,
            memory_used=68 * 1024 * 1024 * 1024,
            memory_total=80 * 1024 * 1024 * 1024,
            temperature=72.0,
            power_draw=340.0,
            power_limit=400.0,
        ),
    ]


@router.get("/metrics/node", response_model=NodeMetricsResponse)
async def get_node_metrics() -> NodeMetricsResponse:
    return NodeMetricsResponse(
        cpu_utilization=45.0,
        memory_used=128 * 1024 * 1024 * 1024,
        memory_total=256 * 1024 * 1024 * 1024,
        network_rx_bytes=10 * 1024 * 1024 * 1024,
        network_tx_bytes=5 * 1024 * 1024 * 1024,
        disk_used=500 * 1024 * 1024 * 1024,
        disk_total=2000 * 1024 * 1024 * 1024,
    )


@router.get("/predictions", response_model=list[FaultPrediction])
async def get_predictions(limit: int = 10) -> list[FaultPrediction]:
    return []


@router.post("/healing/trigger")
async def trigger_healing(
    prediction_id: UUID,
    action: str = "migrate",
) -> dict:
    return {
        "prediction_id": str(prediction_id),
        "action": action,
        "status": "initiated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
