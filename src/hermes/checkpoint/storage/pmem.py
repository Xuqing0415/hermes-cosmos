"""
Persistent Memory (PMem) storage backend
"""

import os
from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog

from hermes.checkpoint.storage.base import StorageBackend
from hermes.core.models import Checkpoint

logger = structlog.get_logger()


class PMemStorage(StorageBackend):
    def __init__(self, path: str = "/mnt/pmem") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[UUID, Checkpoint] = {}

    def _get_checkpoint_path(self, checkpoint_id: UUID) -> Path:
        return self.path / f"{checkpoint_id}.ckpt"

    async def save(self, checkpoint: Checkpoint, data: bytes) -> str:
        checkpoint_path = self._get_checkpoint_path(checkpoint.id)

        with open(checkpoint_path, "wb") as f:
            f.write(data)
            os.fsync(f.fileno())

        checkpoint.size_bytes = len(data)
        checkpoint.storage_path = str(checkpoint_path)
        self._metadata[checkpoint.id] = checkpoint

        logger.info(
            "Checkpoint saved to PMem",
            checkpoint_id=str(checkpoint.id),
            size=len(data),
            path=str(checkpoint_path),
        )

        return checkpoint.storage_path

    async def load(self, checkpoint_id: UUID) -> Optional[bytes]:
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "rb") as f:
            return f.read()

    async def delete(self, checkpoint_id: UUID) -> bool:
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)

        if not checkpoint_path.exists():
            return False

        checkpoint_path.unlink()

        if checkpoint_id in self._metadata:
            del self._metadata[checkpoint_id]

        logger.info(
            "Checkpoint deleted from PMem",
            checkpoint_id=str(checkpoint_id),
        )

        return True

    async def exists(self, checkpoint_id: UUID) -> bool:
        return self._get_checkpoint_path(checkpoint_id).exists()

    async def get_size(self, checkpoint_id: UUID) -> Optional[int]:
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)

        if not checkpoint_path.exists():
            return None

        return checkpoint_path.stat().st_size

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
        self._metadata.clear()
