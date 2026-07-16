"""
Tests for scheduler decision reproducibility
"""

import pytest
from uuid import UUID
from datetime import datetime

from hermes.core.models import Job, ResourceRequest, JobPriority, JobRequirements, PlacementConstraints
from hermes.scheduler.algorithms.placement import PlacementAlgorithm
from hermes.scheduler.resources.cluster import ClusterState, Node


class TestSchedulerReproducibility:
    """"""

    @pytest.fixture
    def cluster_state(self):
        """"""
        nodes = [
            Node(id="node-1", region="us-west", gpu_count=4, gpu_type="A100"),
            Node(id="node-2", region="us-west", gpu_count=4, gpu_type="A100"),
            Node(id="node-3", region="eu-west", gpu_count=8, gpu_type="A100"),
        ]
        return ClusterState(nodes=nodes)

    @pytest.fixture
    def job(self):
        """"""
        return Job(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            name="test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            image="pytorch:latest",
            requirements=JobRequirements(gpu_count=4),
            priority=JobPriority.HIGH,
            constraints=PlacementConstraints(regions=["us-west"]),
        )

    def test_placement_reproducibility(self, cluster_state, job):
        """"""
        algorithm = PlacementAlgorithm()
        
        result1 = algorithm.find_best_placement(job, cluster_state)
        
        algorithm2 = PlacementAlgorithm()
        result2 = algorithm2.find_best_placement(job, cluster_state)
        
        assert result1 is not None and result2 is not None
        assert result1.job_id == result2.job_id
        assert result1.region == result2.region
        assert result1.gpu_count == result2.gpu_count
        assert result1.node_ids == result2.node_ids
        assert result1.score == result2.score

    def test_multiple_jobs_reproducibility(self, cluster_state):
        """"""
        jobs = [
            Job(
                id=UUID(f"00000000-0000-0000-0000-{i:012d}"),
                name=f"test-job-{i}",
                tenant_id="tenant-1",
                user_id="user-1",
                image="pytorch:latest",
                requirements=JobRequirements(gpu_count=2),
                priority=JobPriority.NORMAL,
                constraints=PlacementConstraints(regions=["us-west"]),
            )
            for i in range(5)
        ]
        
        algorithm1 = PlacementAlgorithm()
        placements1 = [algorithm1.find_best_placement(job, cluster_state) for job in jobs]
        
        algorithm2 = PlacementAlgorithm()
        placements2 = [algorithm2.find_best_placement(job, cluster_state) for job in jobs]
        
        for p1, p2 in zip(placements1, placements2):
            assert p1 is not None and p2 is not None
            assert p1.job_id == p2.job_id
            assert p1.region == p2.region
            assert p1.gpu_count == p2.gpu_count
            assert p1.node_ids == p2.node_ids
            assert p1.score == p2.score
