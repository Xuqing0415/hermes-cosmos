"""
Base storage backend interface
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from hermes.core.models import Checkpoint


class StorageBackend(ABC):
    @abstractmethod
    async def save(self, checkpoint: Checkpoint, data: bytes) -> str:
        pass

    @abstractmethod
    async def load(self, checkpoint_id: UUID) -> Optional[bytes]:
        pass

    @abstractmethod
    async def delete(self, checkpoint_id: UUID) -> bool:
        pass

    @abstractmethod
    async def exists(self, checkpoint_id: UUID) -> bool:
        pass

    @abstractmethod
    async def get_size(self, checkpoint_id: UUID) -> Optional[int]:
        pass

    @abstractmethod
    async def list_checkpoints(
        self,
        job_id: Optional[UUID] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Checkpoint]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass
