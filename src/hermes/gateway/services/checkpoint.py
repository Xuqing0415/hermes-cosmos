"""
Checkpoint service client
"""

from typing import Any, Optional
from uuid import UUID

import grpc
import structlog

from hermes.core.exceptions import ServiceUnavailableError
from hermes.core.models import Checkpoint

logger = structlog.get_logger()


class CheckpointClient:
    def __init__(self, address: str) -> None:
        self.address = address
        self._channel: Optional[grpc.aio.Channel] = None
        self._connected = False

    async def connect(self) -> None:
        logger.info("Connecting to checkpoint service", address=self.address)
        try:
            self._channel = grpc.aio.insecure_channel(self.address)
            await self._channel.channel_ready()
            self._connected = True
            logger.info("Connected to checkpoint service")
        except Exception as e:
            logger.error("Failed to connect to checkpoint service", error=str(e))
            raise ServiceUnavailableError("checkpoint", str(e))

    async def disconnect(self) -> None:
        if self._channel:
            await self._channel.close()
            self._connected = False
            logger.info("Disconnected from checkpoint service")

    async def create_checkpoint(
        self,
        job_id: UUID,
        tenant_id: str,
        is_delta: bool = False,
    ) -> Checkpoint:
        self._ensure_connected()
        return Checkpoint(
            job_id=job_id,
            tenant_id=tenant_id,
            is_delta=is_delta,
        )

    async def get_checkpoint(self, checkpoint_id: UUID) -> Optional[Checkpoint]:
        self._ensure_connected()
        return None

    async def list_checkpoints(
        self,
        job_id: Optional[UUID] = None,
        tenant_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Checkpoint], int]:
        self._ensure_connected()
        return [], 0

    async def restore_checkpoint(self, checkpoint_id: UUID) -> dict[str, Any]:
        self._ensure_connected()
        return {"status": "restoring", "checkpoint_id": str(checkpoint_id)}

    async def delete_checkpoint(self, checkpoint_id: UUID) -> bool:
        self._ensure_connected()
        return True

    def _ensure_connected(self) -> None:
        if not self._connected or not self._channel:
            raise ServiceUnavailableError("checkpoint", "Not connected")
