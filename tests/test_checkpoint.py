"""
Tests for Hermes checkpoint module
"""

import pytest
from datetime import datetime
from uuid import UUID

from hermes.core.models import Checkpoint, CheckpointState
from hermes.checkpoint.storage.memory import MemoryStorage
from hermes.checkpoint.delta import DeltaEngine
from hermes.checkpoint.coordinator import CheckpointCoordinator


class TestMemoryStorage:
    @pytest.fixture
    def storage(self) -> MemoryStorage:
        return MemoryStorage(max_size=1024 * 1024)

    @pytest.fixture
    def sample_checkpoint(self) -> Checkpoint:
        return Checkpoint(
            job_id=UUID("00000000-0000-0000-0000-000000000001"),
            tenant_id="tenant-1",
        )

    @pytest.mark.asyncio
    async def test_save_and_load(
        self,
        storage: MemoryStorage,
        sample_checkpoint: Checkpoint,
    ) -> None:
        data = b"test checkpoint data"
        path = await storage.save(sample_checkpoint, data)

        assert path is not None
        assert sample_checkpoint.size_bytes == len(data)

        loaded = await storage.load(sample_checkpoint.id)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_delete(
        self,
        storage: MemoryStorage,
        sample_checkpoint: Checkpoint,
    ) -> None:
        data = b"test checkpoint data"
        await storage.save(sample_checkpoint, data)

        result = await storage.delete(sample_checkpoint.id)
        assert result is True

        loaded = await storage.load(sample_checkpoint.id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_exists(
        self,
        storage: MemoryStorage,
        sample_checkpoint: Checkpoint,
    ) -> None:
        data = b"test checkpoint data"

        assert await storage.exists(sample_checkpoint.id) is False

        await storage.save(sample_checkpoint, data)
        assert await storage.exists(sample_checkpoint.id) is True

    @pytest.mark.asyncio
    async def test_list_checkpoints(
        self,
        storage: MemoryStorage,
    ) -> None:
        checkpoints = []
        for i in range(3):
            cp = Checkpoint(
                job_id=UUID("00000000-0000-0000-0000-000000000001"),
                tenant_id="tenant-1",
            )
            await storage.save(cp, f"data-{i}".encode())
            checkpoints.append(cp)

        listed = await storage.list_checkpoints()
        assert len(listed) == 3


class TestDeltaEngine:
    @pytest.fixture
    def engine(self) -> DeltaEngine:
        return DeltaEngine(enabled=True, compression=True)

    @pytest.mark.asyncio
    async def test_compute_delta(self, engine: DeltaEngine) -> None:
        checkpoint_id = UUID("00000000-0000-0000-0000-000000000001")

        data1 = b"a" * 1000
        delta1, is_delta1 = await engine.compute_delta(checkpoint_id, data1)
        assert is_delta1 is False

        data2 = b"b" * 1000
        delta2, is_delta2 = await engine.compute_delta(checkpoint_id, data2)
        assert is_delta2 is True

    def test_split_into_blocks(self, engine: DeltaEngine) -> None:
        data = b"a" * 10000
        blocks = engine._split_into_blocks(data)

        assert len(blocks) == 3
        assert all(len(block) <= engine.block_size for block in blocks)

    def test_compute_block_hashes(self, engine: DeltaEngine) -> None:
        blocks = [b"block1", b"block2", b"block3"]
        hashes = engine._compute_block_hashes(blocks)

        assert len(hashes) == 3
        assert all(len(h) == 64 for h in hashes)


class TestCheckpointCoordinator:
    @pytest.fixture
    def coordinator(self) -> CheckpointCoordinator:
        storage = MemoryStorage()
        delta_engine = DeltaEngine(enabled=True)
        return CheckpointCoordinator(storage, delta_engine)

    @pytest.mark.asyncio
    async def test_create_checkpoint(
        self,
        coordinator: CheckpointCoordinator,
    ) -> None:
        job_id = UUID("00000000-0000-0000-0000-000000000001")
        data = b"test checkpoint data"

        checkpoint = await coordinator.create_checkpoint(
            job_id=job_id,
            tenant_id="tenant-1",
            data=data,
        )

        assert checkpoint.state == CheckpointState.COMPLETED
        assert checkpoint.size_bytes == len(data)

    @pytest.mark.asyncio
    async def test_restore_checkpoint(
        self,
        coordinator: CheckpointCoordinator,
    ) -> None:
        job_id = UUID("00000000-0000-0000-0000-000000000001")
        data = b"test checkpoint data"

        checkpoint = await coordinator.create_checkpoint(
            job_id=job_id,
            tenant_id="tenant-1",
            data=data,
        )

        restored = await coordinator.restore_checkpoint(checkpoint.id)
        assert restored == data

    @pytest.mark.asyncio
    async def test_delete_checkpoint(
        self,
        coordinator: CheckpointCoordinator,
    ) -> None:
        job_id = UUID("00000000-0000-0000-0000-000000000001")
        data = b"test checkpoint data"

        checkpoint = await coordinator.create_checkpoint(
            job_id=job_id,
            tenant_id="tenant-1",
            data=data,
        )

        result = await coordinator.delete_checkpoint(checkpoint.id)
        assert result is True
