"""
Core data models for Hermes
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from hermes.core.config import GPUType, Region


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SCHEDULING = "scheduling"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    MIGRATING = "migrating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class Resource(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    type: ResourceType
    region: Region
    gpu_type: Optional[GPUType] = None
    total_capacity: int
    available_capacity: int
    allocated_capacity: int = 0
    cost_per_hour: float
    carbon_intensity: float = Field(default=0.0, description="gCO2/kWh")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def utilization(self) -> float:
        if self.total_capacity == 0:
            return 0.0
        return self.allocated_capacity / self.total_capacity


class GPUResource(Resource):
    type: ResourceType = ResourceType.GPU
    gpu_count: int = Field(default=1, ge=1)
    gpu_memory_gb: int = Field(default=80, ge=1)
    interconnect_type: str = Field(default="NVLink")
    rdma_enabled: bool = Field(default=True)


class JobRequirements(BaseModel):
    gpu_count: int = Field(default=1, ge=1, description="Number of GPUs required")
    gpu_type: GPUType = Field(default=GPUType.NVIDIA_H100)
    gpu_memory_gb: int = Field(default=80, ge=1)
    cpu_cores: int = Field(default=8, ge=1)
    memory_gb: int = Field(default=64, ge=1)
    storage_gb: int = Field(default=500, ge=1)
    network_bandwidth_gbps: int = Field(default=100, ge=1)
    max_duration_hours: int = Field(default=72, ge=1)
    checkpoint_interval_seconds: int = Field(default=300, ge=60)


class PlacementConstraints(BaseModel):
    regions: Optional[list[Region]] = None
    data_locality: bool = Field(default=True, description="Require data locality")
    anti_affinity: list[str] = Field(default_factory=list)
    affinity: list[str] = Field(default_factory=list)
    carbon_aware: bool = Field(default=True)
    max_carbon_intensity: Optional[float] = None


class Job(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    tenant_id: str
    user_id: str
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    requirements: JobRequirements
    constraints: PlacementConstraints = Field(default_factory=PlacementConstraints)
    framework: str = Field(default="pytorch")
    command: Optional[str] = None
    image: str
    environment: dict[str, str] = Field(default_factory=dict)
    volumes: list[str] = Field(default_factory=list)
    checkpoint_enabled: bool = Field(default=True)
    checkpoint_id: Optional[UUID] = None
    allocated_resources: list[UUID] = Field(default_factory=list)
    region: Optional[Region] = None
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class CheckpointState(str, Enum):
    CREATING = "creating"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class Checkpoint(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    tenant_id: str
    state: CheckpointState = CheckpointState.CREATING
    size_bytes: int = 0
    delta_bytes: int = 0
    is_delta: bool = Field(default=False, description="Is this a delta checkpoint")
    parent_id: Optional[UUID] = None
    storage_path: Optional[str] = None
    checksum: Optional[str] = None
    compression_ratio: float = Field(default=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FaultPrediction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    resource_id: UUID
    prediction_type: str
    probability: float = Field(ge=0.0, le=1.0)
    predicted_failure_time: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Metric(BaseModel):
    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: str
    source: str
    subject: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Tenant(BaseModel):
    id: str
    name: str
    quota: dict[ResourceType, int] = Field(default_factory=dict)
    roles: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class User(BaseModel):
    id: str
    tenant_id: str
    email: str
    roles: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
