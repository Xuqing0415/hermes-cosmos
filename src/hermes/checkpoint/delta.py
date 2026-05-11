"""
Delta checkpoint engine for incremental state saving
"""

import hashlib
import zlib
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DeltaBlock:
    offset: int
    size: int
    checksum: str
    data: bytes


class DeltaEngine:
    def __init__(
        self,
        enabled: bool = True,
        compression: bool = True,
        block_size: int = 4096,
    ) -> None:
        self.enabled = enabled
        self.compression = compression
        self.block_size = block_size
        self._previous_state: dict[UUID, bytes] = {}
        self._block_hashes: dict[UUID, list[str]] = {}

    async def compute_delta(
        self,
        checkpoint_id: UUID,
        current_data: bytes,
    ) -> tuple[bytes, bool]:
        if not self.enabled:
            return current_data, False

        previous_data = self._previous_state.get(checkpoint_id)
        if previous_data is None:
            self._previous_state[checkpoint_id] = current_data
            return current_data, False

        delta = await self._compute_block_delta(checkpoint_id, previous_data, current_data)

        self._previous_state[checkpoint_id] = current_data

        logger.info(
            "Delta computed",
            checkpoint_id=str(checkpoint_id),
            original_size=len(current_data),
            delta_size=len(delta),
            compression_ratio=len(delta) / len(current_data) if current_data else 0,
        )

        return delta, True

    async def _compute_block_delta(
        self,
        checkpoint_id: UUID,
        previous: bytes,
        current: bytes,
    ) -> bytes:
        previous_blocks = self._split_into_blocks(previous)
        current_blocks = self._split_into_blocks(current)

        previous_hashes = self._compute_block_hashes(previous_blocks)
        current_hashes = self._compute_block_hashes(current_blocks)

        delta_blocks = []

        for i, (block, hash_val) in enumerate(zip(current_blocks, current_hashes)):
            if i >= len(previous_hashes) or previous_hashes[i] != hash_val:
                delta_blocks.append(block)

        delta_data = b"".join(delta_blocks)

        if self.compression:
            delta_data = zlib.compress(delta_data)

        self._block_hashes[checkpoint_id] = current_hashes

        return delta_data

    def _split_into_blocks(self, data: bytes) -> list[bytes]:
        blocks = []
        for i in range(0, len(data), self.block_size):
            blocks.append(data[i : i + self.block_size])
        return blocks

    def _compute_block_hashes(self, blocks: list[bytes]) -> list[str]:
        return [hashlib.sha256(block).hexdigest() for block in blocks]

    async def apply_delta(
        self,
        checkpoint_id: UUID,
        base_data: bytes,
        delta_data: bytes,
    ) -> bytes:
        if self.compression:
            delta_data = zlib.decompress(delta_data)

        logger.info(
            "Delta applied",
            checkpoint_id=str(checkpoint_id),
            base_size=len(base_data),
            delta_size=len(delta_data),
        )

        return delta_data

    def clear_state(self, checkpoint_id: UUID) -> None:
        self._previous_state.pop(checkpoint_id, None)
        self._block_hashes.pop(checkpoint_id, None)
