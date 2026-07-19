"""
Cluster resource management
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

from hermes.core.config import Region
from hermes.core.models import Job, Resource, ResourceType

logger = structlog.get_logger()

try:
    import etcd3
    HAS_ETCD = True
except ImportError:
    etcd3 = None
    HAS_ETCD = False


@dataclass
class NodeState:
    id: str
    region: Region
    gpu_type: str
    total_gpus: int
    available_gpus: int
    status: str
    last_heartbeat: datetime


@dataclass
class Node:
    id: str
    region: str
    gpu_count: int
    gpu_type: str


@dataclass
class ClusterState:
    nodes: list[Node]


class ClusterManager:
    def __init__(self, etcd_endpoints: list[str] = None) -> None:
        self.etcd_endpoints = etcd_endpoints or ["localhost:2379"]
        self._etcd: Optional[Any] = None
        self._nodes: dict[str, NodeState] = {}
        self._resources: dict[str, Resource] = {}

    async def connect(self) -> None:
        if not HAS_ETCD:
            logger.info("etcd3 not available, running in standalone mode")
            return

        logger.info("Connecting to etcd", endpoints=self.etcd_endpoints)

        try:
            host, port = self.etcd_endpoints[0].split(":")
            self._etcd = etcd3.client(host=host, port=int(port))
            await self._load_cluster_state()
            logger.info("Connected to etcd, but note: etcd3 client is synchronous, running in degraded mode")
        except Exception as e:
            logger.warning(f"Failed to connect to etcd: {e}, running in standalone mode")
            self._etcd = None

    async def disconnect(self) -> None:
        if self._etcd and HAS_ETCD:
            self._etcd.close()
        logger.info("Disconnected from etcd")

    async def _load_cluster_state(self) -> None:
        if self._etcd is None or not HAS_ETCD:
            return

        try:
            # NOTE: etcd3 is synchronous; this call blocks the event loop.
            # A production fix would migrate to an async etcd client (e.g. aioetcd).
            nodes_data = await asyncio.to_thread(self._etcd.get_prefix, "/hermes/nodes/")
            for value, metadata in nodes_data:
                node_id = metadata.key.decode().split("/")[-1]
                node_state = NodeState(
                    id=node_id,
                    region=Region.US_EAST,
                    gpu_type="nvidia-h100",
                    total_gpus=8,
                    available_gpus=8,
                    status="ready",
                    last_heartbeat=datetime.utcnow(),
                )
                self._nodes[node_id] = node_state
        except Exception as e:
            logger.warning(f"Failed to load cluster state: {e}")

    async def get_cluster_state(self) -> dict[str, Any]:
        total_gpus = sum(n.total_gpus for n in self._nodes.values())
        available_gpus = sum(n.available_gpus for n in self._nodes.values())

        regions: dict[Region, dict[str, Any]] = {}
        for node in self._nodes.values():
            if node.region not in regions:
                regions[node.region] = {
                    "total": 0,
                    "available": 0,
                    "nodes": [],
                }
            regions[node.region]["total"] += node.total_gpus
            regions[node.region]["available"] += node.available_gpus
            regions[node.region]["nodes"].append(node.id)

        return {
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "regions": regions,
            "nodes": len(self._nodes),
            "timestamp": datetime.utcnow(),
        }

    async def get_available_resources(
        self,
        region: Optional[Region] = None,
        gpu_type: Optional[str] = None,
    ) -> list[Resource]:
        resources = list(self._resources.values())

        if region:
            resources = [r for r in resources if r.region == region]

        return resources

    async def allocate_resources(
        self,
        job: Job,
        placement: dict[str, Any],
    ) -> bool:
        logger.info(
            "Allocating resources",
            job_id=str(job.id),
            region=placement.get("region"),
            gpu_count=placement.get("gpu_count"),
        )

        return True

    async def release_resources(self, job_id: UUID) -> None:
        logger.info("Releasing resources", job_id=str(job_id))

    async def register_node(self, node: NodeState) -> None:
        self._nodes[node.id] = node
        logger.info("Node registered", node_id=node.id, region=node.region.value)

    async def unregister_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            del self._nodes[node_id]
            logger.info("Node unregistered", node_id=node_id)

    async def update_node_heartbeat(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].last_heartbeat = datetime.utcnow()

    async def get_nodes_by_region(self, region: Region) -> list[NodeState]:
        return [n for n in self._nodes.values() if n.region == region]
