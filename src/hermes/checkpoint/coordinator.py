"""
Checkpoint coordinator for managing distributed checkpoints
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import structlog

from hermes.checkpoint.delta import DeltaEngine
from hermes.checkpoint.storage.base import StorageBackend
from hermes.core.models import Checkpoint, CheckpointState

logger = structlog.get_logger()


class CheckpointCoordinator:
    def __init__(
        self,
        storage: StorageBackend,
        delta_engine: DeltaEngine,
        max_concurrent: int = 10,
        retention_days: int = 7,
    ) -> None:
        self.storage = storage
        self.delta_engine = delta_engine
        self.max_concurrent = max_concurrent
        self.retention_days = retention_days
        self._active_checkpoints: dict[UUID, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def create_checkpoint(
        self,
        job_id: UUID,
        tenant_id: str,
        data: bytes,
        is_delta: bool = False,
        parent_id: Optional[UUID] = None,
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            job_id=job_id,
            tenant_id=tenant_id,
            state=CheckpointState.CREATING,
            is_delta=is_delta,
            parent_id=parent_id,
        )

        async with self._semaphore:
            try:
                if is_delta:
                    original_size = len(data)
                    data, is_delta = await self.delta_engine.compute_delta(
                        checkpoint.id, data
                    )
                    checkpoint.is_delta = is_delta
                    checkpoint.delta_bytes = len(data)
                    if is_delta:
                        checkpoint.compression_ratio = (
                            len(data) / original_size
                            if original_size > 0
                            else 1.0
                        )

                path = await self.storage.save(checkpoint, data)

                checkpoint.state = CheckpointState.COMPLETED
                checkpoint.size_bytes = len(data)
                checkpoint.storage_path = path

                logger.info(
                    "Checkpoint created",
                    checkpoint_id=str(checkpoint.id),
                    job_id=str(job_id),
                    is_delta=is_delta,
                    size=checkpoint.size_bytes,
                )

            except Exception as e:
                checkpoint.state = CheckpointState.FAILED
                logger.error(
                    "Checkpoint creation failed",
                    checkpoint_id=str(checkpoint.id),
                    error=str(e),
                )
                raise

        return checkpoint

    async def restore_checkpoint(
        self,
        checkpoint_id: UUID,
    ) -> bytes:
        checkpoint = await self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        if checkpoint.state != CheckpointState.COMPLETED:
            raise ValueError(f"Checkpoint not in completed state: {checkpoint.state}")

        data = await self.storage.load(checkpoint_id)
        if data is None:
            raise ValueError(f"Failed to load checkpoint data: {checkpoint_id}")

        if checkpoint.is_delta and checkpoint.parent_id:
            parent_data = await self.storage.load(checkpoint.parent_id)
            if parent_data:
                data = await self.delta_engine.apply_delta(
                    checkpoint_id, parent_data, data
                )

        logger.info(
            "Checkpoint restored",
            checkpoint_id=str(checkpoint_id),
            size=len(data),
        )

        return data

    async def get_checkpoint(self, checkpoint_id: UUID) -> Optional[Checkpoint]:
        checkpoints = await self.storage.list_checkpoints(limit=1)
        for cp in checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None

    async def delete_checkpoint(self, checkpoint_id: UUID) -> bool:
        result = await self.storage.delete(checkpoint_id)

        if result:
            self.delta_engine.clear_state(checkpoint_id)
            logger.info("Checkpoint deleted", checkpoint_id=str(checkpoint_id))

        return result

    async def list_checkpoints(
        self,
        job_id: Optional[UUID] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Checkpoint]:
        return await self.storage.list_checkpoints(job_id, tenant_id, limit)

    async def cleanup_expired(self) -> int:
        expired_threshold = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

        checkpoints = await self.storage.list_checkpoints(limit=1000)

        deleted_count = 0
        for checkpoint in checkpoints:
            if checkpoint.created_at < expired_threshold:
                if await self.delete_checkpoint(checkpoint.id):
                    deleted_count += 1

        logger.info("Cleaned up expired checkpoints", count=deleted_count)
        return deleted_count
