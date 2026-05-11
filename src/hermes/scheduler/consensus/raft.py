"""
Raft consensus protocol implementation
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class RaftState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    index: int
    term: int
    command: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RaftConfig:
    election_timeout_min: int = 3000
    election_timeout_max: int = 6000
    heartbeat_interval: int = 1000
    snapshot_threshold: int = 10000


class RaftNode:
    def __init__(
        self,
        node_id: Optional[str] = None,
        data_dir: str = "/var/lib/hermes/raft",
        election_timeout: int = 5000,
        heartbeat_interval: int = 1000,
        snapshot_threshold: int = 10000,
        peers: Optional[list[str]] = None,
    ) -> None:
        self.node_id = node_id or str(uuid4())
        self.data_dir = data_dir
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        self.snapshot_threshold = snapshot_threshold
        self.peers = peers or []

        self.state = RaftState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: list[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0

        self.next_index: dict[str, int] = {}
        self.match_index: dict[str, int] = {}

        self._leader_id: Optional[str] = None
        self._last_heartbeat: datetime = datetime.utcnow()
        self._election_timer: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        logger.info("Starting Raft node", node_id=self.node_id)
        self._running = True
        self._reset_election_timer()
        logger.info("Raft node started", state=self.state.value, term=self.current_term)

    async def stop(self) -> None:
        logger.info("Stopping Raft node", node_id=self.node_id)
        self._running = False

        if self._election_timer:
            self._election_timer.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        logger.info("Raft node stopped")

    def _reset_election_timer(self) -> None:
        if self._election_timer:
            self._election_timer.cancel()

        timeout = random.randint(
            self.election_timeout,
            self.election_timeout * 2,
        )

        async def election_timeout() -> None:
            await asyncio.sleep(timeout / 1000.0)
            if self._running and self.state != RaftState.LEADER:
                await self._start_election()

        self._election_timer = asyncio.create_task(election_timeout())

    async def _start_election(self) -> None:
        logger.info("Starting election", node_id=self.node_id, term=self.current_term + 1)

        self.state = RaftState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id

        votes = 1
        majority = (len(self.peers) + 1) // 2 + 1

        if votes >= majority:
            await self._become_leader()
        else:
            self._reset_election_timer()

    async def _become_leader(self) -> None:
        logger.info("Became leader", node_id=self.node_id, term=self.current_term)
        self.state = RaftState.LEADER
        self._leader_id = self.node_id

        for peer in self.peers:
            self.next_index[peer] = len(self.log) + 1
            self.match_index[peer] = 0

        self._start_heartbeat()

    def _start_heartbeat(self) -> None:
        async def send_heartbeats() -> None:
            while self._running and self.state == RaftState.LEADER:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval / 1000.0)

        self._heartbeat_task = asyncio.create_task(send_heartbeats())

    async def _send_heartbeat(self) -> None:
        self._last_heartbeat = datetime.utcnow()

    async def append_entry(self, command: dict[str, Any]) -> bool:
        if self.state != RaftState.LEADER:
            return False

        entry = LogEntry(
            index=len(self.log) + 1,
            term=self.current_term,
            command=command,
        )
        self.log.append(entry)

        logger.debug(
            "Appended entry",
            index=entry.index,
            term=entry.term,
        )

        return True

    async def request_vote(
        self,
        candidate_id: str,
        term: int,
        last_log_index: int,
        last_log_term: int,
    ) -> tuple[bool, int]:
        if term < self.current_term:
            return False, self.current_term

        if term > self.current_term:
            self.current_term = term
            self.state = RaftState.FOLLOWER
            self.voted_for = None

        if self.voted_for is None or self.voted_for == candidate_id:
            if last_log_index >= len(self.log):
                self.voted_for = candidate_id
                self._reset_election_timer()
                return True, self.current_term

        return False, self.current_term

    @property
    def is_leader(self) -> bool:
        return self.state == RaftState.LEADER

    @property
    def leader_id(self) -> Optional[str]:
        return self._leader_id
