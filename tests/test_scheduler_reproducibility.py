"""
Tests for scheduler decision reproducibility
"""

import pytest
from uuid import UUID
from datetime import datetime

from hermes.core.models import Job, ResourceRequest, JobPriority
from hermes.scheduler.algorithms.placement import PlacementAlgorithm
from hermes.scheduler.resources.cluster import ClusterState, Node


class TestSchedulerReproducibility:
    """测试调度决策的可复现性"""

    @pytest.fixture
    def cluster_state(self):
        """创建固定的集群状态"""
        nodes = [
            Node(id="node-1", region="us-west", gpu_count=4, gpu_type="A100"),
            Node(id="node-2", region="us-west", gpu_count=4, gpu_type="A100"),
            Node(id="node-3", region="eu-west", gpu_count=8, gpu_type="A100"),
        ]
        return ClusterState(nodes=nodes)

    @pytest.fixture
    def job(self):
        """创建固定的作业请求"""
        return Job(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            tenant_id="tenant-1",
            resource_request=ResourceRequest(gpu_count=4, gpu_type="A100"),
            priority=JobPriority.HIGH,
            data_residency="us-west",
            submitted_at=datetime(2024, 1, 1, 12, 0, 0),
        )

    def test_placement_reproducibility(self, cluster_state, job):
        """测试放置算法的决策可复现性"""
        algorithm = PlacementAlgorithm()
        
        # 第一次运行
        result1 = algorithm.find_best_placement(job, cluster_state)
        
        # 重置状态后第二次运行
        algorithm2 = PlacementAlgorithm()
        result2 = algorithm2.find_best_placement(job, cluster_state)
        
        # 验证结果一致
        assert result1 == result2, "Placement decisions should be reproducible"
        
        if result1 is not None:
            assert result1.node_id == result2.node_id
            assert result1.score == result2.score

    def test_multiple_jobs_reproducibility(self, cluster_state):
        """测试多个作业调度的可复现性"""
        jobs = [
            Job(
                id=UUID(f"00000000-0000-0000-0000-{i:012d}"),
                tenant_id="tenant-1",
                resource_request=ResourceRequest(gpu_count=2, gpu_type="A100"),
                priority=JobPriority.NORMAL,
                data_residency="us-west",
                submitted_at=datetime(2024, 1, 1, 12, 0, i),
            )
            for i in range(5)
        ]
        
        algorithm1 = PlacementAlgorithm()
        placements1 = []
        
        for job in jobs:
            placement = algorithm1.find_best_placement(job, cluster_state)
            placements1.append(placement)
        
        algorithm2 = PlacementAlgorithm()
        placements2 = []
        
        for job in jobs:
            placement = algorithm2.find_best_placement(job, cluster_state)
            placements2.append(placement)
        
        for p1, p2 in zip(placements1, placements2):
            assert p1 == p2, "Multiple job placements should be reproducible"
