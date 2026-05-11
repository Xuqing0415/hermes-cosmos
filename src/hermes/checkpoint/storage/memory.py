"""
In-memory storage backend for testing
"""

from typing import Optional
from uuid import UUID

import structlog

from hermes.checkpoint.storage.base import StorageBackend
from hermes.core.models import Checkpoint

logger = structlog.get_logger()


class MemoryStorage(StorageBackend):
    def __init__(self, max_size: int = 1024 * 1024 * 1024) -> None:
        self.max_size = max_size
        self._data: dict[UUID, bytes] = {}
        self._metadata: dict[UUID, Checkpoint] = {}
        self._current_size = 0

    async def save(self, checkpoint: Checkpoint, data: bytes) -> str:
        size = len(data)

        if self._current_size + size > self.max_size:
            await self._evict_oldest()

        self._data[checkpoint.id] = data
        self._metadata[checkpoint.id] = checkpoint
        self._current_size += size

        checkpoint.size_bytes = size
        checkpoint.storage_path = f"memory://{checkpoint.id}"

        logger.info(
            "Checkpoint saved to memory",
            checkpoint_id=str(checkpoint.id),
            size=size,
        )

        return checkpoint.storage_path

    async def load(self, checkpoint_id: UUID) -> Optional[bytes]:
        return self._data.get(checkpoint_id)

    async def delete(self, checkpoint_id: UUID) -> bool:
        if checkpoint_id not in self._data:
            return False

        size = len(self._data[checkpoint_id])
        del self._data[checkpoint_id]
        del self._metadata[checkpoint_id]
        self._current_size -= size

        logger.info(
            "Checkpoint deleted from memory",
            checkpoint_id=str(checkpoint_id),
        )

        return True

    async def exists(self, checkpoint_id: UUID) -> bool:
        return checkpoint_id in self._data

    async def get_size(self, checkpoint_id: UUID) -> Optional[int]:
        data = self._data.get(checkpoint_id)
        return len(data) if data else None

    async def list_checkpoints(
        self,
        job_id: Optional[UUID] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Checkpoint]:
        checkpoints = list(self._metadata.values())

        if job_id:
            checkpoints = [c for c in checkpoints if c.job_id == job_id]
        if tenant_id:
            checkpoints = [c for c in checkpoints if c.tenant_id == tenant_id]

        return checkpoints[:limit]

    async def close(self) -> None:
        self._data.clear()
        self._metadata.clear()
        self._current_size = 0

    async def _evict_oldest(self) -> None:
        if not self._metadata:
            return

        oldest_id = min(
            self._metadata.keys(),
            key=lambda x: self._metadata[x].created_at,
        )

        await self.delete(oldest_id)
        logger.info("Evicted oldest checkpoint", checkpoint_id=str(oldest_id))
