"""
Scheduler service client
"""

import asyncio
from typing import Any, Optional
from uuid import UUID

import grpc
import structlog

from hermes.core.exceptions import ServiceUnavailableError
from hermes.core.models import Job

logger = structlog.get_logger()


class SchedulerClient:
    def __init__(self, address: str) -> None:
        self.address = address
        self._channel: Optional[grpc.aio.Channel] = None
        self._connected = False

    async def connect(self) -> None:
        logger.info("Connecting to scheduler", address=self.address)
        try:
            self._channel = grpc.aio.insecure_channel(self.address)
            await self._channel.channel_ready()
            self._connected = True
            logger.info("Connected to scheduler")
        except Exception as e:
            logger.error("Failed to connect to scheduler", error=str(e))
            raise ServiceUnavailableError("scheduler", str(e))

    async def disconnect(self) -> None:
        if self._channel:
            await self._channel.close()
            self._connected = False
            logger.info("Disconnected from scheduler")

    async def submit_job(self, job: Job) -> dict[str, Any]:
        self._ensure_connected()
        return {"job_id": str(job.id), "status": "queued"}

    async def get_job(self, job_id: UUID) -> Optional[Job]:
        self._ensure_connected()
        return None

    async def cancel_job(self, job_id: UUID) -> bool:
        self._ensure_connected()
        return True

    async def list_jobs(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Job], int]:
        self._ensure_connected()
        return [], 0

    async def get_cluster_status(self) -> dict[str, Any]:
        self._ensure_connected()
        return {
            "total_gpus": 10000,
            "available_gpus": 6000,
            "regions": {
                "us-east": {"total": 4000, "available": 2500},
                "us-west": {"total": 3000, "available": 2000},
                "eu-west": {"total": 2000, "available": 1000},
                "asia-east": {"total": 1000, "available": 500},
            },
        }

    def _ensure_connected(self) -> None:
        if not self._connected or not self._channel:
            raise ServiceUnavailableError("scheduler", "Not connected")
