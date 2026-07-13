"""
Placement engine for job scheduling
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

from hermes.core.config import Region
from hermes.core.models import Job, PlacementConstraints

logger = structlog.get_logger()


@dataclass
class PlacementResult:
    job_id: UUID
    region: Region
    gpu_count: int
    node_ids: list[str]
    cost_per_hour: float
    carbon_intensity: float
    data_locality: bool
    score: float
    scheduled_at: datetime


@dataclass
class NodeInfo:
    id: str
    region: Region
    gpu_type: str
    total_gpus: int
    available_gpus: int
    cost_per_hour: float
    carbon_intensity: float
    labels: dict[str, str]


class PlacementAlgorithm:
    def __init__(
        self,
        carbon_aware: bool = True,
        cost_weight: float = 0.4,
        carbon_weight: float = 0.3,
        locality_weight: float = 0.3,
    ) -> None:
        self.carbon_aware = carbon_aware
        self.cost_weight = cost_weight
        self.carbon_weight = carbon_weight
        self.locality_weight = locality_weight

    def find_best_placement(
        self,
        job,
        cluster_state,
    ):
        if not cluster_state.nodes:
            return None

        best_node = None
        best_score = -1.0

        for node in cluster_state.nodes:
            if node.gpu_count >= job.requirements.gpu_count:
                if job.constraints.regions and node.region not in job.constraints.regions:
                    continue

                score = self._calculate_score(node, job)
                if score > best_score:
                    best_score = score
                    best_node = node

        if best_node:
            return PlacementResult(
                job_id=job.id,
                region=best_node.region,
                gpu_count=job.requirements.gpu_count,
                node_ids=[best_node.id],
                cost_per_hour=0.0,
                carbon_intensity=0.0,
                data_locality=True,
                score=best_score,
                scheduled_at=datetime.utcnow(),
            )
        return None

    def _calculate_score(self, node, job) -> float:
        score = 0.0
        score += self.cost_weight * (1.0 / (node.gpu_count + 1))
        if job.constraints.regions and node.region in job.constraints.regions:
            score += self.locality_weight * 1.0
        else:
            score += self.locality_weight * 0.5
        return score


class PlacementEngine:
    def __init__(
        self,
        carbon_aware: bool = True,
        cost_weight: float = 0.4,
        carbon_weight: float = 0.3,
        locality_weight: float = 0.3,
    ) -> None:
        self.carbon_aware = carbon_aware
        self.cost_weight = cost_weight
        self.carbon_weight = carbon_weight
        self.locality_weight = locality_weight

    async def find_best_placement(
        self,
        job: Job,
        available_nodes: list[NodeInfo],
    ) -> Optional[PlacementResult]:
        candidates = self._filter_candidates(job, available_nodes)

        if not candidates:
            return None

        scored_candidates = [
            (node, self._score_placement(job, node))
            for node in candidates
        ]

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_node, best_score = scored_candidates[0]

        return self._create_placement(job, best_node, best_score)

    def _filter_candidates(
        self,
        job: Job,
        nodes: list[NodeInfo],
    ) -> list[NodeInfo]:
        candidates = []

        for node in nodes:
            if node.available_gpus < job.requirements.gpu_count:
                continue

            if job.constraints.regions and node.region not in job.constraints.regions:
                continue

            if job.constraints.max_carbon_intensity:
                if node.carbon_intensity > job.constraints.max_carbon_intensity:
                    continue

            candidates.append(node)

        return candidates

    def _score_placement(self, job: Job, node: NodeInfo) -> float:
        score = 0.0

        cost_score = self._normalize_cost(node.cost_per_hour)
        score += cost_score * self.cost_weight

        if self.carbon_aware and job.constraints.carbon_aware:
            carbon_score = self._normalize_carbon(node.carbon_intensity)
            score += carbon_score * self.carbon_weight

        if job.constraints.data_locality:
            locality_score = 1.0
            score += locality_score * self.locality_weight

        return score

    def _normalize_cost(self, cost: float) -> float:
        max_cost = 10.0
        min_cost = 1.0
        return 1.0 - (cost - min_cost) / (max_cost - min_cost)

    def _normalize_carbon(self, carbon: float) -> float:
        max_carbon = 500.0
        min_carbon = 0.0
        return 1.0 - (carbon - min_carbon) / (max_carbon - min_carbon)

    def _create_placement(
        self,
        job: Job,
        node: NodeInfo,
        score: float,
    ) -> PlacementResult:
        return PlacementResult(
            job_id=job.id,
            region=node.region,
            gpu_count=job.requirements.gpu_count,
            node_ids=[node.id],
            cost_per_hour=node.cost_per_hour,
            carbon_intensity=node.carbon_intensity,
            data_locality=job.constraints.data_locality,
            score=score,
            scheduled_at=datetime.utcnow(),
        )

    async def find_migration_target(
        self,
        job: Job,
        current_region: Region,
        available_nodes: list[NodeInfo],
    ) -> Optional[PlacementResult]:
        filtered_nodes = [
            n for n in available_nodes
            if n.region != current_region
        ]

        return await self.find_best_placement(job, filtered_nodes)
