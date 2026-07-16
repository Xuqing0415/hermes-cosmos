"""
Type definitions for neural-symbolic proof generation
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from uuid import UUID


class ProofStatus(Enum):
    PROVEN = "proven"
    DISPROVEN = "disproven"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


class PathCondition:
    """Represents a path condition in symbolic execution"""
    
    def __init__(self, condition: str, variables: Dict[str, Any] = None):
        self.condition = condition
        self.variables = variables or {}
    
    def __repr__(self):
        return f"PathCondition({self.condition})"


@dataclass
class ProofResult:
    """Result of a theorem proving attempt"""
    
    status: ProofStatus
    smt_expression: str
    model: Optional[Dict[str, Any]] = None
    certificate: Optional[str] = None
    duration: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ProofTarget:
    """Target for proof generation"""
    
    id: str
    function_name: str
    path_condition: PathCondition
    precondition: Optional[str] = None
    postcondition: Optional[str] = None
    risk_score: float = 0.0
    priority: int = 0


@dataclass
class VerificationCondition:
    """Verification condition (VC) for theorem proving"""
    
    id: str
    smt_formula: str
    description: str
    source: str


@dataclass
class GeneratedTest:
    """Test case generated from proof results"""
    
    id: str
    function_name: str
    inputs: Dict[str, Any]
    expected_output: Optional[Any] = None
    assertion: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ProofReport:
    """Comprehensive report of proof results"""
    
    proven_count: int
    disproven_count: int
    unknown_count: int
    timeout_count: int
    proven_targets: List[ProofTarget]
    disproven_targets: List[ProofTarget]
    generated_tests: List[GeneratedTest]
    total_duration: float
