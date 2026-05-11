"""
Tests for Hermes core module
"""

import pytest
from datetime import datetime
from uuid import UUID

from hermes.core.config import (
    GatewayConfig,
    SchedulerConfig,
    Region,
    GPUType,
)
from hermes.core.models import (
    Job,
    JobStatus,
    JobPriority,
    JobRequirements,
    PlacementConstraints,
    Checkpoint,
    CheckpointState,
    Resource,
    ResourceType,
)
from hermes.core.exceptions import (
    HermesError,
    SchedulerError,
    CheckpointError,
    ResourceNotFoundError,
)


class TestConfig:
    def test_gateway_config_defaults(self) -> None:
        config = GatewayConfig()
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8080
        assert config.auth.enabled is True
        assert config.rate_limit.enabled is True

    def test_scheduler_config_defaults(self) -> None:
        config = SchedulerConfig()
        assert config.server.port == 8080
        assert config.algorithm.algorithm == "mcts"
        assert config.algorithm.carbon_aware is True

    def test_region_enum(self) -> None:
        assert Region.US_EAST.value == "us-east"
        assert Region.EU_WEST.value == "eu-west"
        assert Region.ASIA_EAST.value == "asia-east"

    def test_gpu_type_enum(self) -> None:
        assert GPUType.NVIDIA_H100.value == "nvidia-h100"
        assert GPUType.NVIDIA_A100.value == "nvidia-a100"


class TestModels:
    def test_job_creation(self) -> None:
        job = Job(
            name="test-job",
            tenant_id="tenant-1",
            user_id="user-1",
            image="pytorch/pytorch:latest",
            requirements=JobRequirements(gpu_count=8),
        )

        assert job.name == "test-job"
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert isinstance(job.id, UUID)

    def test_job_requirements(self) -> None:
        requirements = JobRequirements(
            gpu_count=16,
            gpu_type=GPUType.NVIDIA_H100,
            cpu_cores=64,
            memory_gb=512,
        )

        assert requirements.gpu_count == 16
        assert requirements.gpu_type == GPUType.NVIDIA_H100
        assert requirements.max_duration_hours == 72

    def test_placement_constraints(self) -> None:
        constraints = PlacementConstraints(
            regions=[Region.US_EAST, Region.US_WEST],
            data_locality=True,
            carbon_aware=True,
        )

        assert len(constraints.regions) == 2
        assert constraints.carbon_aware is True

    def test_checkpoint_creation(self) -> None:
        checkpoint = Checkpoint(
            job_id=UUID("00000000-0000-0000-0000-000000000001"),
            tenant_id="tenant-1",
            size_bytes=1024 * 1024 * 1024,
        )

        assert checkpoint.state == CheckpointState.CREATING
        assert checkpoint.is_delta is False

    def test_resource_utilization(self) -> None:
        resource = Resource(
            name="gpu-cluster-1",
            type=ResourceType.GPU,
            region=Region.US_EAST,
            total_capacity=100,
            allocated_capacity=50,
            cost_per_hour=3.50,
        )

        assert resource.utilization == 0.5


class TestExceptions:
    def test_hermes_error(self) -> None:
        error = HermesError("Test error", code="TEST_ERROR")
        assert error.message == "Test error"
        assert error.code == "TEST_ERROR"

    def test_hermes_error_to_dict(self) -> None:
        error = HermesError("Test error", code="TEST_ERROR", details={"key": "value"})
        error_dict = error.to_dict()

        assert error_dict["error"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"]["key"] == "value"

    def test_scheduler_error(self) -> None:
        error = SchedulerError("Scheduling failed", job_id="job-123")
        assert error.code == "SCHEDULER_ERROR"
        assert error.details["job_id"] == "job-123"

    def test_checkpoint_error(self) -> None:
        error = CheckpointError("Checkpoint failed", checkpoint_id="cp-123")
        assert error.code == "CHECKPOINT_ERROR"
        assert error.details["checkpoint_id"] == "cp-123"

    def test_resource_not_found_error(self) -> None:
        error = ResourceNotFoundError("Job", "job-123")
        assert error.code == "RESOURCE_NOT_FOUND"
        assert "Job not found" in error.message
