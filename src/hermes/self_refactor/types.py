from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SmellType(Enum):
    CYCLE_DEPENDENCY = "cycle_dependency"
    DUPLICATE_CODE = "duplicate_code"
    GOD_MODULE = "god_module"
    LONG_FUNCTION = "long_function"
    DATA_CLUMP = "data_clump"
    SHOTGUN_SURGERY = "shotgun_surgery"
    FEATURE_ENVY = "feature_envy"


class SmellSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RefactorOperation(Enum):
    EXTRACT_FUNCTION = "extract_function"
    MOVE_METHOD = "move_method"
    EXTRACT_CLASS = "extract_class"
    MOVE_FIELD = "move_field"
    SPLIT_MODULE = "split_module"
    INTRODUCE_INTERFACE = "introduce_interface"
    REMOVE_DUPLICATION = "remove_duplication"
    BREAK_CYCLE = "break_cycle"


@dataclass
class CodeLocation:
    file_path: str
    line_start: int
    line_end: int


@dataclass
class SmellInstance:
    smell_type: SmellType
    severity: SmellSeverity
    description: str
    locations: List[CodeLocation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0


@dataclass
class ArchMetrics:
    file_path: str
    loc: int
    cyclomatic_complexity: int
    halstead_volume: float
    maintainability_index: float
    function_count: int
    class_count: int


@dataclass
class ModuleMetrics:
    module_name: str
    file_metrics: List[ArchMetrics]
    total_loc: int
    total_functions: int
    total_classes: int
    avg_complexity: float
    coupling_score: float
    cohesion_score: float


@dataclass
class RefactorAction:
    operation: RefactorOperation
    description: str
    source_location: Optional[CodeLocation] = None
    target_location: Optional[CodeLocation] = None
    new_name: Optional[str] = None
    affected_files: List[str] = field(default_factory=list)


@dataclass
class RefactorPlan:
    plan_id: str
    smell: SmellInstance
    actions: List[RefactorAction]
    estimated_effort: float
    risk_level: str
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass
class RefactorResult:
    plan_id: str
    success: bool
    applied_actions: List[str]
    failed_actions: List[str] = field(default_factory=list)
    verification_passed: bool = False
    rollback_required: bool = False
    execution_time: float = 0.0


@dataclass
class HealthReport:
    timestamp: str
    total_smells: int
    smells_by_type: Dict[SmellType, int]
    smells_by_severity: Dict[SmellSeverity, int]
    module_metrics: List[ModuleMetrics]
    overall_health_score: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RefactorHistoryEntry:
    timestamp: str
    plan_id: str
    smell_type: SmellType
    actions_taken: List[str]
    success: bool
    verification_result: str
    execution_time: float
    rollback_used: bool = False
    notes: str = ""