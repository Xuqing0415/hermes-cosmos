"""
Configuration models for Hermes components
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class Region(str, Enum):
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_EAST = "asia-east"
    ASIA_SOUTHEAST = "asia-southeast"


class GPUType(str, Enum):
    NVIDIA_A100 = "nvidia-a100"
    NVIDIA_H100 = "nvidia-h100"
    NVIDIA_V100 = "nvidia-v100"
    AMD_MI300 = "amd-mi300"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")
    workers: int = Field(default=4, ge=1, description="Number of workers")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")


class AuthConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable authentication")
    spiffe_host: str = Field(default="unix:///tmp/spire-agent/public/api.sock")
    jwks_url: Optional[str] = Field(default=None, description="JWKS URL for JWT validation")
    token_ttl: int = Field(default=3600, ge=60, description="Token TTL in seconds")
    admin_roles: list[str] = Field(default=["admin"], description="Admin roles")


class RateLimitConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable rate limiting")
    rps: int = Field(default=100, ge=1, description="Requests per second")
    burst: int = Field(default=200, ge=1, description="Burst capacity")
    per_tenant: bool = Field(default=True, description="Per-tenant rate limiting")


class TLSConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable TLS")
    cert_file: Optional[str] = Field(default=None, description="Certificate file path")
    key_file: Optional[str] = Field(default=None, description="Private key file path")
    ca_file: Optional[str] = Field(default=None, description="CA certificate file path")


class UpstreamConfig(BaseModel):
    scheduler_addr: str = Field(default="localhost:50051", description="Scheduler service address")
    checkpoint_addr: str = Field(default="localhost:50052", description="Checkpoint service address")
    agent_addr: str = Field(default="localhost:50053", description="Agent service address")


class DatabaseConfig(BaseModel):
    url: str = Field(default="postgresql+asyncpg://hermes:hermes@localhost/hermes")
    pool_size: int = Field(default=20, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    echo: bool = Field(default=False)


class RedisConfig(BaseModel):
    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=50, ge=1)


class EtcdConfig(BaseModel):
    endpoints: list[str] = Field(default=["localhost:2379"])
    prefix: str = Field(default="/hermes/")


class PrometheusConfig(BaseModel):
    enabled: bool = Field(default=True)
    port: int = Field(default=9090, ge=1, le=65535)
    path: str = Field(default="/metrics")


class TracingConfig(BaseModel):
    enabled: bool = Field(default=True)
    service_name: str = Field(default="hermes")
    exporter: str = Field(default="otlp")
    endpoint: str = Field(default="http://localhost:4317")
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class SchedulerAlgorithmConfig(BaseModel):
    algorithm: str = Field(default="mcts", description="Scheduling algorithm")
    mcts_iterations: int = Field(default=1000, ge=100, description="MCTS iterations")
    mcts_exploration: float = Field(default=1.414, ge=0.0, description="MCTS exploration constant")
    placement_strategy: str = Field(default="cost_aware", description="Placement strategy")
    carbon_aware: bool = Field(default=True, description="Enable carbon-aware scheduling")


class RaftConfig(BaseModel):
    election_timeout: int = Field(default=5000, ge=1000)
    heartbeat_interval: int = Field(default=1000, ge=100)
    snapshot_threshold: int = Field(default=10000, ge=1000)
    data_dir: str = Field(default="/var/lib/hermes/raft")


class CheckpointStorageConfig(BaseModel):
    backend: str = Field(default="memory", description="Storage backend")
    pmem_path: Optional[str] = Field(default="/mnt/pmem", description="PMem mount path")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: Optional[str] = Field(default=None, description="S3 region")
    compression: bool = Field(default=True, description="Enable compression")
    delta_enabled: bool = Field(default=True, description="Enable delta checkpointing")


class FaultPredictionConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable fault prediction")
    model_path: str = Field(default="/var/lib/hermes/models/fault_predictor.onnx")
    prediction_window: int = Field(default=30, description="Prediction window in minutes")
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class NetworkQoSConfig(BaseModel):
    enabled: bool = Field(default=True)
    rdma_enabled: bool = Field(default=True)
    sriov_enabled: bool = Field(default=True)
    ecn_enabled: bool = Field(default=True)
    default_bandwidth: int = Field(default=100, description="Default bandwidth in Gbps")


class GatewayConfig(BaseSettings):
    model_config = {"env_prefix": "HERMES_GATEWAY_"}

    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    tls: TLSConfig = Field(default_factory=TLSConfig)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    log_level: LogLevel = Field(default=LogLevel.INFO)


class SchedulerConfig(BaseSettings):
    model_config = {"env_prefix": "HERMES_SCHEDULER_"}

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    etcd: EtcdConfig = Field(default_factory=EtcdConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    algorithm: SchedulerAlgorithmConfig = Field(default_factory=SchedulerAlgorithmConfig)
    raft: RaftConfig = Field(default_factory=RaftConfig)
    log_level: LogLevel = Field(default=LogLevel.INFO)


class CheckpointConfig(BaseSettings):
    model_config = {"env_prefix": "HERMES_CHECKPOINT_"}

    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: CheckpointStorageConfig = Field(default_factory=CheckpointStorageConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    log_level: LogLevel = Field(default=LogLevel.INFO)


class AgentConfig(BaseSettings):
    model_config = {"env_prefix": "HERMES_AGENT_"}

    server: ServerConfig = Field(default_factory=ServerConfig)
    region: Region = Field(default=Region.US_EAST)
    fault_prediction: FaultPredictionConfig = Field(default_factory=FaultPredictionConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    log_level: LogLevel = Field(default=LogLevel.INFO)


class ObservabilityConfig(BaseSettings):
    model_config = {"env_prefix": "HERMES_OBSERVABILITY_"}

    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    ebpf_enabled: bool = Field(default=True, description="Enable eBPF metrics collection")
    gpu_metrics_interval: int = Field(default=10, ge=1, description="GPU metrics collection interval in seconds")


class SecurityConfig(BaseSettings):
    model_config = {"env_prefix": "HERMES_SECURITY_"}

    auth: AuthConfig = Field(default_factory=AuthConfig)
    tls: TLSConfig = Field(default_factory=TLSConfig)
    network: NetworkQoSConfig = Field(default_factory=NetworkQoSConfig)
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_log_path: str = Field(default="/var/log/hermes/audit.log")
