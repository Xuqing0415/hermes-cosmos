"""
Tests for Hermes scheduler module
"""

import pytest
from datetime import datetime
from uuid import UUID

from hermes.core.config import Region, GPUType
from hermes.core.models import Job, JobRequirements, PlacementConstraints
from hermes.scheduler.algorithms.mcts import (
    MCTSScheduler,
    MCTSNode,
    ClusterState,
    PlacementDecision,
)
from hermes.scheduler.algorithms.placement import (
    PlacementEngine,
    NodeInfo,
    PlacementResult,
)


class TestMCTSScheduler:
    @pytest.fixture
    def scheduler(self) -> MCTSScheduler:
        return MCTSScheduler(
            iterations=100,
            exploration_constant=1.414,
        )

    @pytest.fixture
    def sample_job(self) -> Job:
        return Job(
            name="test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            image="pytorch:latest",
            requirements=JobRequirements(gpu_count=8),
            constraints=PlacementConstraints(
                regions=[Region.US_EAST, Region.US_WEST],
            ),
        )

    @pytest.fixture
    def cluster_state(self) -> ClusterState:
        return ClusterState(
            regions={
                Region.US_EAST: {
                    "available": 100,
                    "total": 200,
                    "cost_per_hour": 3.50,
                    "carbon_intensity": 300.0,
                },
                Region.US_WEST: {
                    "available": 50,
                    "total": 100,
                    "cost_per_hour": 3.20,
                    "carbon_intensity": 250.0,
                },
            },
            total_gpus=300,
            available_gpus=150,
            running_jobs=10,
        )

    @pytest.mark.asyncio
    async def test_schedule_job(
        self,
        scheduler: MCTSScheduler,
        sample_job: Job,
        cluster_state: ClusterState,
    ) -> None:
        placement = await scheduler.schedule(sample_job, cluster_state)

        assert placement is not None
        assert str(placement.job_id) == str(sample_job.id)
        assert placement.region in ['us-east', 'us-west']
        assert placement.gpu_count == 8

    def test_mcts_node_ucb1(self) -> None:
        parent = MCTSNode(state={})
        child = MCTSNode(state={}, parent=parent)
        child.visits = 10
        child.value = 5.0
        parent.visits = 100

        ucb1 = child.ucb1(1.414)
        assert ucb1 > 0

    def test_mcts_node_expand(self) -> None:
        node = MCTSNode(state={}, untried_actions=[{"region": "us-east"}])
        child = node.expand({"region": "us-east"})

        assert child in node.children.values()
        assert child.state == {"region": "us-east"}


class TestPlacementEngine:
    @pytest.fixture
    def engine(self) -> PlacementEngine:
        return PlacementEngine(
            carbon_aware=True,
            cost_weight=0.4,
            carbon_weight=0.3,
        )

    @pytest.fixture
    def sample_nodes(self) -> list[NodeInfo]:
        return [
            NodeInfo(
                id="node-1",
                region=Region.US_EAST,
                gpu_type="nvidia-h100",
                total_gpus=8,
                available_gpus=8,
                cost_per_hour=3.50,
                carbon_intensity=300.0,
                labels={},
            ),
            NodeInfo(
                id="node-2",
                region=Region.US_WEST,
                gpu_type="nvidia-h100",
                total_gpus=8,
                available_gpus=4,
                cost_per_hour=3.20,
                carbon_intensity=250.0,
                labels={},
            ),
        ]

    @pytest.fixture
    def sample_job(self) -> Job:
        return Job(
            name="test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            image="pytorch:latest",
            requirements=JobRequirements(gpu_count=4),
        )

    @pytest.mark.asyncio
    async def test_find_best_placement(
        self,
        engine: PlacementEngine,
        sample_job: Job,
        sample_nodes: list[NodeInfo],
    ) -> None:
        placement = await engine.find_best_placement(sample_job, sample_nodes)

        assert placement is not None
        assert placement.job_id == sample_job.id
        assert placement.gpu_count == 4

    def test_filter_candidates(
        self,
        engine: PlacementEngine,
        sample_nodes: list[NodeInfo],
    ) -> None:
        job = Job(
            name="test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            image="pytorch:latest",
            requirements=JobRequirements(gpu_count=10),
        )

        candidates = engine._filter_candidates(job, sample_nodes)
        assert len(candidates) == 0

    def test_score_placement(
        self,
        engine: PlacementEngine,
        sample_nodes: list[NodeInfo],
    ) -> None:
        job = Job(
            name="test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            image="pytorch:latest",
            requirements=JobRequirements(gpu_count=4),
            constraints=PlacementConstraints(carbon_aware=True),
        )

        score = engine._score_placement(job, sample_nodes[0])
        assert 0 <= score <= 1
