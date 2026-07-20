from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid
import random


class TestCaseDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TestCase:
    id: str
    domain: str
    pattern_type: str
    description: str
    synthetic_input: Dict[str, Any]
    expected_abstract_type: str
    expected_fix_description: str
    difficulty: TestCaseDifficulty = TestCaseDifficulty.MEDIUM
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "synthetic_input": self.synthetic_input,
            "expected_abstract_type": self.expected_abstract_type,
            "expected_fix_description": self.expected_fix_description,
            "difficulty": self.difficulty.value,
            "tags": self.tags,
        }


KNOWN_PATTERN_TYPES = [
    "boundary_check_missing",
    "resource_limit_missing",
    "input_validation_missing",
    "api_version_deprecated",
    "performance_bottleneck",
    "division_by_zero",
    "uninitialized_variable",
]

UNKNOWN_PATTERN_TYPES = [
    "concurrency_race_condition",
    "type_mismatch",
    "infinite_recursion",
    "memory_leak",
]

SYNTHETIC_CASES: List[Dict[str, Any]] = [
    # === Python (default) domain cases ===
    {
        "domain": "default",
        "pattern_type": "null_pointer",
        "description": "Python函数缺少空值检查",
        "synthetic_input": {"function": "get_user", "line": 23, "code": "return user.profile"},
        "expected_abstract_type": "boundary_check_missing",
        "expected_fix_description": "Add bounds checking before array access",
        "difficulty": "easy",
        "tags": ["null_safety", "python_common"],
    },
    {
        "domain": "default",
        "pattern_type": "missing_validation",
        "description": "Python API缺少输入验证",
        "synthetic_input": {"function": "create_item", "line": 50, "code": "db.insert(request.body)"},
        "expected_abstract_type": "input_validation_missing",
        "expected_fix_description": "Add input validation and sanitization",
        "difficulty": "easy",
        "tags": ["validation", "api_security"],
    },
    {
        "domain": "default",
        "pattern_type": "performance",
        "description": "Python数据处理性能瓶颈",
        "synthetic_input": {"function": "process_batch", "line": 120, "code": "for item in items: process(item)"},
        "expected_abstract_type": "performance_bottleneck",
        "expected_fix_description": "Optimize loop or use batch processing",
        "difficulty": "medium",
        "tags": ["performance", "optimization"],
    },
    {
        "domain": "default",
        "pattern_type": "type_mismatch",
        "description": "Python类型不匹配导致运行时错误",
        "synthetic_input": {"function": "calculate", "line": 15, "code": "return a + b", "types": {"a": "str", "b": "int"}},
        "expected_abstract_type": "type_mismatch",
        "expected_fix_description": "Add type checking and conversion",
        "difficulty": "medium",
        "tags": ["type_safety", "unknown_pattern"],
    },
    {
        "domain": "default",
        "pattern_type": "infinite_recursion",
        "description": "Python递归缺少终止条件",
        "synthetic_input": {"function": "traverse", "line": 30, "code": "return traverse(node.next)"},
        "expected_abstract_type": "infinite_recursion",
        "expected_fix_description": "Add base case and depth limit",
        "difficulty": "hard",
        "tags": ["recursion", "unknown_pattern"],
    },

    # === MLIR domain cases ===
    {
        "domain": "mlir",
        "pattern_type": "index_out_of_bounds",
        "description": "MLIR memref访问越界",
        "synthetic_input": {"operation": "memref.load", "index": "%idx", "bound": "%size", "file": "test.mlir"},
        "expected_abstract_type": "boundary_check_missing",
        "expected_fix_description": "Add cf.assert for index range",
        "difficulty": "easy",
        "tags": ["memory_safety", "mlir_common"],
    },
    {
        "domain": "mlir",
        "pattern_type": "missing_bound_check",
        "description": "MLIR缓冲区访问缺少边界检查",
        "synthetic_input": {"operation": "memref.store", "buffer": "%A", "dim": 0, "file": "matmul.mlir"},
        "expected_abstract_type": "boundary_check_missing",
        "expected_fix_description": "Add cf.assert for index range",
        "difficulty": "easy",
        "tags": ["bounds_check", "mlir_common"],
    },
    {
        "domain": "mlir",
        "pattern_type": "potential_divide_by_zero",
        "description": "MLIR算术运算可能除零",
        "synthetic_input": {"operation": "arith.divsi", "operand": "%divisor", "file": "arith.mlir"},
        "expected_abstract_type": "division_by_zero",
        "expected_fix_description": "Add divisor zero check before arithmetic",
        "difficulty": "medium",
        "tags": ["arithmetic_safety", "mlir"],
    },
    {
        "domain": "mlir",
        "pattern_type": "uninitialized_variable",
        "description": "MLIR未初始化变量使用",
        "synthetic_input": {"variable": "%result", "block": "^bb1", "file": "loop.mlir"},
        "expected_abstract_type": "uninitialized_variable",
        "expected_fix_description": "Add variable initialization",
        "difficulty": "medium",
        "tags": ["init", "mlir"],
    },
    {
        "domain": "mlir",
        "pattern_type": "concurrency_race",
        "description": "MLIR并行操作存在竞态条件",
        "synthetic_input": {"operation": "gpu.launch", "shared_memory": "%buf", "file": "gpu.mlir"},
        "expected_abstract_type": "concurrency_race_condition",
        "expected_fix_description": "Add synchronization barrier",
        "difficulty": "hard",
        "tags": ["concurrency", "unknown_pattern"],
    },

    # === K8s domain cases ===
    {
        "domain": "k8s",
        "pattern_type": "missing_resources",
        "description": "K8s Deployment缺少资源限制",
        "synthetic_input": {"deployment": "api-service", "missing_fields": ["limits.cpu", "limits.memory"], "file": "deploy.yaml"},
        "expected_abstract_type": "resource_limit_missing",
        "expected_fix_description": "Add CPU/memory limits to deployment",
        "difficulty": "easy",
        "tags": ["resource_management", "k8s_common"],
    },
    {
        "domain": "k8s",
        "pattern_type": "hpa_missing",
        "description": "K8s Deployment没有HPA自动伸缩",
        "synthetic_input": {"deployment": "worker", "replicas": 3, "file": "deploy.yaml"},
        "expected_abstract_type": "resource_limit_missing",
        "expected_fix_description": "Add HPA configuration for autoscaling",
        "difficulty": "medium",
        "tags": ["autoscaling", "k8s"],
    },
    {
        "domain": "k8s",
        "pattern_type": "deprecated_api",
        "description": "K8s使用已弃用API版本",
        "synthetic_input": {"api_version": "networking.k8s.io/v1beta1", "target_version": "networking.k8s.io/v1", "file": "ingress.yaml"},
        "expected_abstract_type": "api_version_deprecated",
        "expected_fix_description": "Update to stable API version",
        "difficulty": "easy",
        "tags": ["api_upgrade", "k8s_common"],
    },
    {
        "domain": "k8s",
        "pattern_type": "old_image",
        "description": "K8s使用过旧的容器镜像版本",
        "synthetic_input": {"deployment": "gateway", "current_version": "v1.2.3", "latest_version": "v1.3.0", "file": "deploy.yaml"},
        "expected_abstract_type": "api_version_deprecated",
        "expected_fix_description": "Upgrade container image version",
        "difficulty": "medium",
        "tags": ["image_management", "k8s"],
    },
    {
        "domain": "k8s",
        "pattern_type": "memory_leak",
        "description": "K8s容器内存泄漏风险",
        "synthetic_input": {"deployment": "data-processor", "memory_usage_gb": 8, "memory_limit_gb": 4, "file": "deploy.yaml"},
        "expected_abstract_type": "memory_leak",
        "expected_fix_description": "Add memory budget and OOM kill policy",
        "difficulty": "hard",
        "tags": ["memory", "unknown_pattern"],
    },
]


class ExperimentDesigner:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._cases = SYNTHETIC_CASES

    def generate_all_cases(self) -> List[TestCase]:
        test_cases = []
        for i, case in enumerate(self._cases):
            test_cases.append(TestCase(
                id=f"exp-{i + 1:04d}",
                domain=case["domain"],
                pattern_type=case["pattern_type"],
                description=case["description"],
                synthetic_input=case["synthetic_input"],
                expected_abstract_type=case["expected_abstract_type"],
                expected_fix_description=case["expected_fix_description"],
                difficulty=TestCaseDifficulty(case["difficulty"]),
                tags=case["tags"],
            ))
        return test_cases

    def generate_by_domain(self, domain: str) -> List[TestCase]:
        return [tc for tc in self.generate_all_cases() if tc.domain == domain]

    def generate_by_pattern(self, pattern_type: str) -> List[TestCase]:
        return [tc for tc in self.generate_all_cases() if tc.expected_abstract_type == pattern_type]

    def generate_known_pattern_cases(self) -> List[TestCase]:
        return [tc for tc in self.generate_all_cases() if tc.expected_abstract_type in KNOWN_PATTERN_TYPES]

    def generate_unknown_pattern_cases(self) -> List[TestCase]:
        return [tc for tc in self.generate_all_cases() if tc.expected_abstract_type not in KNOWN_PATTERN_TYPES]

    def summary(self) -> Dict[str, Any]:
        all_cases = self.generate_all_cases()
        domain_counts = {}
        pattern_counts = {}
        difficulty_counts = {}

        for tc in all_cases:
            domain_counts[tc.domain] = domain_counts.get(tc.domain, 0) + 1
            pattern_counts[tc.expected_abstract_type] = pattern_counts.get(tc.expected_abstract_type, 0) + 1
            difficulty_counts[tc.difficulty.value] = difficulty_counts.get(tc.difficulty.value, 0) + 1

        return {
            "total": len(all_cases),
            "by_domain": domain_counts,
            "by_pattern": pattern_counts,
            "by_difficulty": difficulty_counts,
            "known_patterns": len(self.generate_known_pattern_cases()),
            "unknown_patterns": len(self.generate_unknown_pattern_cases()),
        }

    def shuffle(self, cases: Optional[List[TestCase]] = None) -> List[TestCase]:
        if cases is None:
            cases = self.generate_all_cases()
        shuffled = list(cases)
        self._rng.shuffle(shuffled)
        return shuffled