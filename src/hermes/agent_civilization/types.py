from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time


class AgentRole(Enum):
    TEST_OFFICER = "test_officer"
    FIXER = "fixer"
    PROVER = "prover"
    LANGUAGE_PRIEST = "language_priest"
    MIGRATION_APOSTLE = "migration_apostle"
    CAUSAL_ORACLE = "causal_oracle"
    ARCHITECTURE_JUDGE = "architecture_judge"
    RESOURCE_OVERSEER = "resource_overseer"
    COMPILER_VERIFIER = "compiler_verifier"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    TASK_PUBLISH = "task_publish"
    TASK_BID = "task_bid"
    TASK_ASSIGN = "task_assign"
    TASK_UPDATE = "task_update"
    TASK_COMPLETE = "task_complete"
    KNOWLEDGE_SHARE = "knowledge_share"
    KNOWLEDGE_REQUEST = "knowledge_request"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    HEARTBEAT = "heartbeat"
    REGISTER = "register"
    DEREGISTER = "deregister"


class CivilizationPhase(Enum):
    STONE_AGE = "stone_age"
    BRONZE_AGE = "bronze_age"
    IRON_AGE = "iron_age"
    INDUSTRIAL_AGE = "industrial_age"
    INFORMATION_AGE = "information_age"
    SINGULARITY = "singularity"


@dataclass
class AgentInfo:
    agent_id: str
    role: AgentRole
    name: str
    capabilities: List[str]
    status: str = "active"
    last_heartbeat: float = 0.0
    performance_score: float = 0.0
    task_count: int = 0
    success_rate: float = 0.0
    resources: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.agent_id == "":
            self.agent_id = str(uuid.uuid4())[:8]


@dataclass
class Task:
    task_id: str
    title: str
    description: str
    type: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    assignee: Optional[str] = None
    bids: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    
    def __post_init__(self):
        if self.task_id == "":
            self.task_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()
            self.updated_at = self.created_at


@dataclass
class Message:
    message_id: str
    type: MessageType
    sender_id: str
    receiver_id: Optional[str] = None
    task_id: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.message_id == "":
            self.message_id = str(uuid.uuid4())[:8]
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class KnowledgeItem:
    knowledge_id: str
    type: str
    content: Dict[str, Any]
    source_agent_id: str
    source_role: AgentRole
    created_at: float = 0.0
    usage_count: int = 0
    rating: float = 0.0
    
    def __post_init__(self):
        if self.knowledge_id == "":
            self.knowledge_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class Vote:
    vote_id: str
    task_id: str
    question: str
    options: List[str]
    votes: Dict[str, str] = field(default_factory=dict)
    deadline: float = 0.0
    result: Optional[str] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.vote_id == "":
            self.vote_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class CivilizationLog:
    log_id: str
    timestamp: float
    agent_id: str
    agent_role: AgentRole
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.log_id == "":
            self.log_id = str(uuid.uuid4())[:8]
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class CivilizationState:
    phase: CivilizationPhase = CivilizationPhase.STONE_AGE
    total_agents: int = 0
    total_tasks: int = 0
    total_knowledge: int = 0
    total_logs: int = 0
    creation_time: float = 0.0
    last_update: float = 0.0
    
    def __post_init__(self):
        if self.creation_time == 0.0:
            self.creation_time = time.time()
            self.last_update = self.creation_time


@dataclass
class CollaborationResult:
    success: bool
    task_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_output: Dict[str, Any] = field(default_factory=dict)
    logs: List[CivilizationLog] = field(default_factory=list)
    execution_time: float = 0.0