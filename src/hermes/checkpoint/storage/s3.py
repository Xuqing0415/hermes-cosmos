"""
S3 storage backend
"""

from typing import Optional
from uuid import UUID

import structlog

from hermes.checkpoint.storage.base import StorageBackend
from hermes.core.models import Checkpoint

logger = structlog.get_logger()


class S3Storage(StorageBackend):
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        prefix: str = "checkpoints/",
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self._metadata: dict[UUID, Checkpoint] = {}

    def _get_s3_key(self, checkpoint_id: UUID) -> str:
        return f"{self.prefix}{checkpoint_id}.ckpt"

    async def save(self, checkpoint: Checkpoint, data: bytes) -> str:
        key = self._get_s3_key(checkpoint.id)

        checkpoint.size_bytes = len(data)
        checkpoint.storage_path = f"s3://{self.bucket}/{key}"
        self._metadata[checkpoint.id] = checkpoint

        logger.info(
            "Checkpoint saved to S3",
            checkpoint_id=str(checkpoint.id),
            size=len(data),
            bucket=self.bucket,
            key=key,
        )

        return checkpoint.storage_path

    async def load(self, checkpoint_id: UUID) -> Optional[bytes]:
        return None

    async def delete(self, checkpoint_id: UUID) -> bool:
        if checkpoint_id in self._metadata:
            del self._metadata[checkpoint_id]

        logger.info(
            "Checkpoint deleted from S3",
            checkpoint_id=str(checkpoint_id),
        )

        return True

    async def exists(self, checkpoint_id: UUID) -> bool:
        return checkpoint_id in self._metadata

    async def get_size(self, checkpoint_id: UUID) -> Optional[int]:
        checkpoint = self._metadata.get(checkpoint_id)
        return checkpoint.size_bytes if checkpoint else None

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
