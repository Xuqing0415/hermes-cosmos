"""
Core data models for Hermes
"""

from datetime import datetime, timezone
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

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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


class ResourceRequest(BaseModel):
    gpu_count: int = Field(default=1, ge=1)
    gpu_type: str = Field(default="NVIDIA_H100")


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Metric(BaseModel):
    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: str
    source: str
    subject: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Tenant(BaseModel):
    id: str
    name: str
    quota: dict[ResourceType, int] = Field(default_factory=dict)
    roles: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class User(BaseModel):
    id: str
    tenant_id: str
    email: str
    roles: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobType(str, Enum):
    TRAINING = "training"
    INFERENCE_SERVICE = "inference_service"
    BATCH_INFERENCE = "batch_inference"


class InferenceEngine(str, Enum):
    VLLM = "vllm"
    TGI = "tgi"
    TRITON = "triton"
    CUSTOM = "custom"


class InferenceConfig(BaseModel):
    model: str = Field(..., description="Model name or path")
    model_path: Optional[str] = Field(None, description="Path to model weights")
    max_batch_size: int = Field(default=32, ge=1, description="Maximum batch size")
    max_sequence_length: int = Field(default=4096, ge=1, description="Maximum sequence length")
    tensor_parallel: int = Field(default=1, ge=1, description="Tensor parallelism size")
    pipeline_parallel: int = Field(default=1, ge=1, description="Pipeline parallelism size")
    quantization: Optional[str] = Field(None, description="Quantization type: fp16, int8, int4")
    engine: InferenceEngine = Field(default=InferenceEngine.VLLM, description="Inference engine")
    kv_cache_dtype: Optional[str] = Field(None, description="KV cache data type")


class HealthCheckConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable health check")
    endpoint: str = Field(default="/health", description="Health check endpoint")
    interval_seconds: int = Field(default=30, ge=5, description="Health check interval")
    timeout_seconds: int = Field(default=5, ge=1, description="Health check timeout")
    unhealthy_threshold: int = Field(default=3, ge=1, description="Unhealthy threshold")


class AutoscalingConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable autoscaling")
    min_replicas: int = Field(default=1, ge=1, description="Minimum replicas")
    max_replicas: int = Field(default=10, ge=1, description="Maximum replicas")
    target_gpu_utilization: float = Field(default=70.0, ge=10.0, le=100.0, description="Target GPU utilization")
    target_request_rate: Optional[float] = Field(None, description="Target requests per second")
    scale_up_cooldown_seconds: int = Field(default=60, ge=10, description="Scale up cooldown")
    scale_down_cooldown_seconds: int = Field(default=300, ge=60, description="Scale down cooldown")


class InferenceService(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    tenant_id: str
    user_id: str
    job_id: Optional[UUID] = None
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    requirements: JobRequirements
    constraints: PlacementConstraints = Field(default_factory=PlacementConstraints)
    inference_config: InferenceConfig
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    image: str
    command: Optional[str] = None
    environment: dict[str, str] = Field(default_factory=dict)
    volumes: list[str] = Field(default_factory=list)
    endpoint: Optional[str] = None
    replicas: int = Field(default=1, ge=1)
    available_replicas: int = Field(default=0, ge=0)
    region: Optional[Region] = None
    sla_availability: float = Field(default=99.9, description="SLA availability target")
    sla_latency_p99_ms: float = Field(default=100.0, description="SLA P99 latency target")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(default=100, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, ge=1, description="Top-k sampling")
    stop_sequences: Optional[list[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(default=False, description="Stream response")


class InferenceResponse(BaseModel):
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class InferenceMetrics(BaseModel):
    requests_total: int = 0
    requests_per_second: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    gpu_utilization: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
