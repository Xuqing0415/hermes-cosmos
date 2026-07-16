from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


class MLIRDialect(str, Enum):
    FUNC = "func"
    ARITH = "arith"
    MEMREF = "memref"
    SCF = "scf"
    TENSOR = "tensor"


class IRNodeType(str, Enum):
    OPERATION = "operation"
    BLOCK = "block"
    REGION = "region"
    VALUE = "value"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ErrorType(str, Enum):
    INDEX_OUT_OF_BOUNDS = "index_out_of_bounds"
    MEMORY_ALIASING = "memory_aliasing"
    TYPE_MISMATCH = "type_mismatch"
    UNDEFINED_BEHAVIOR = "undefined_behavior"
    EQUIVALENCE_FAILURE = "equivalence_failure"


@dataclass
class MemRefType:
    shape: List[int]
    element_type: str
    layout: Optional[List[int]] = None
    memory_space: int = 0
    
    def get_num_elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    def is_valid_index(self, indices: List[int]) -> bool:
        if len(indices) != len(self.shape):
            return False
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= self.shape[i]:
                return False
        return True


@dataclass
class IRNode:
    id: str
    type: IRNodeType
    dialect: Optional[MLIRDialect] = None
    operation_name: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    operands: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    children: List["IRNode"] = field(default_factory=list)
    parent: Optional["IRNode"] = None


@dataclass
class SymbolicValue:
    name: str
    dtype: str
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None
    constraints: List[str] = field(default_factory=list)
    
    def to_smt_expression(self) -> str:
        expr = self.name
        if self.lower_bound is not None:
            expr = f"(>= {expr} {self.lower_bound})"
        if self.upper_bound is not None:
            expr = f"(and {expr} (< {expr} {self.upper_bound}))"
        return expr


@dataclass
class VerificationError:
    type: ErrorType
    message: str
    location: Optional[str] = None
    node_id: Optional[str] = None
    smt_query: Optional[str] = None
    confidence: float = 1.0


@dataclass
class VerificationResult:
    status: VerificationStatus
    errors: List[VerificationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    smt_queries: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    def add_error(self, error_type: ErrorType, message: str, 
                  location: Optional[str] = None, node_id: Optional[str] = None,
                  smt_query: Optional[str] = None):
        self.errors.append(VerificationError(
            type=error_type,
            message=message,
            location=location,
            node_id=node_id,
            smt_query=smt_query
        ))
        if self.status == VerificationStatus.PASSED:
            self.status = VerificationStatus.FAILED


@dataclass
class TensorInput:
    name: str
    shape: List[int]
    dtype: str = "f32"
    data: Optional[List[Any]] = None
    
    def generate_random_data(self) -> List[Any]:
        import random
        num_elements = 1
        for dim in self.shape:
            num_elements *= dim
        
        if self.dtype == "f32":
            self.data = [random.uniform(-1.0, 1.0) for _ in range(num_elements)]
        elif self.dtype == "i32":
            self.data = [random.randint(-100, 100) for _ in range(num_elements)]
        elif self.dtype == "i64":
            self.data = [random.randint(-1000000, 1000000) for _ in range(num_elements)]
        else:
            self.data = [random.uniform(-1.0, 1.0) for _ in range(num_elements)]
        
        return self.data


@dataclass
class KernelTest:
    input_data: List[TensorInput]
    expected_output: Optional[List[TensorInput]] = None
    actual_output: Optional[List[TensorInput]] = None
    passed: bool = False
    duration: float = 0.0


@dataclass
class PassEquivalenceResult:
    pass_name: str
    original_output: Any
    optimized_output: Any
    is_equivalent: bool
    diff: Optional[str] = None
    num_tests: int = 0
    passed_tests: int = 0