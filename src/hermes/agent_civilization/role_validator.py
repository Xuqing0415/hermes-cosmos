import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .agents import BaseAgent
from .role_inventor import RoleDefinition


class ValidationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class ValidationResult:
    validation_id: str
    role_definition: RoleDefinition
    status: ValidationStatus = ValidationStatus.PENDING
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.validation_id == "":
            self.validation_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    @property
    def success_rate(self) -> float:
        if self.tests_run == 0:
            return 0.0
        return self.tests_passed / self.tests_run


class RoleValidator:
    def __init__(self):
        self.validation_results: Dict[str, ValidationResult] = {}
    
    def validate_role(self, agent: BaseAgent, role_def: RoleDefinition) -> ValidationResult:
        validation = ValidationResult(
            validation_id="",
            role_definition=role_def,
            status=ValidationStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            tests = self._generate_test_cases(role_def)
            
            for test_case in tests:
                validation.tests_run += 1
                try:
                    result = self._execute_test(agent, test_case)
                    if result.get("success", False) or result.get("status") == "processed":
                        validation.tests_passed += 1
                    else:
                        validation.tests_failed += 1
                        validation.errors.append(result.get("error", f"Test {test_case['name']} failed"))
                except Exception as e:
                    validation.tests_failed += 1
                    validation.errors.append(f"Test {test_case['name']} exception: {str(e)}")
            
            if validation.tests_failed == 0:
                validation.status = ValidationStatus.PASSED
            else:
                validation.status = ValidationStatus.FAILED
            
        except Exception as e:
            validation.status = ValidationStatus.ERROR
            validation.errors.append(f"Validation error: {str(e)}")
        
        validation.execution_time = time.time() - start_time
        self.validation_results[validation.validation_id] = validation
        
        return validation
    
    def _generate_test_cases(self, role_def: RoleDefinition) -> List[Dict[str, Any]]:
        test_cases = []
        
        for capability in role_def.capabilities:
            test_cases.append({
                "name": f"test_capability_{capability}",
                "task_type": capability,
                "input_data": self._generate_test_input(role_def, capability),
                "expected_output_keys": list(role_def.output_spec.keys())
            })
        
        for responsibility in role_def.responsibilities:
            method_name = f"_{responsibility.lower().replace(' ', '_').replace('.', '')}"
            test_cases.append({
                "name": f"test_responsibility_{method_name}",
                "task_type": responsibility,
                "input_data": {"responsibility": responsibility},
                "expected_output_keys": ["responsibility", "status"]
            })
        
        test_cases.append({
            "name": "test_init",
            "task_type": "init",
            "input_data": {},
            "expected_output_keys": []
        })
        
        return test_cases
    
    def _generate_test_input(self, role_def: RoleDefinition, capability: str) -> Dict[str, Any]:
        input_spec = role_def.input_spec
        
        test_input = {}
        for key, description in input_spec.items():
            if key == "source_code":
                test_input[key] = """def test_function(a, b):
    return a + b
"""
            elif key == "execution_context":
                test_input[key] = {"environment": "test", "test_data": [1, 2, 3]}
            elif key == "performance_baseline":
                test_input[key] = {"expected_time_ms": 100}
            elif key == "complexity_thresholds":
                test_input[key] = {"max_cyclomatic": 10, "max_lines": 50}
            elif key == "component_specs":
                test_input[key] = ["component_a", "component_b"]
            elif key == "security_policies":
                test_input[key] = ["no_sql_injection", "input_validation"]
            else:
                test_input[key] = f"test_{key}"
        
        return test_input
    
    def _execute_test(self, agent: BaseAgent, test_case: Dict[str, Any]) -> Dict[str, Any]:
        task_type = test_case["task_type"]
        input_data = test_case["input_data"]
        
        if task_type == "init":
            return {"success": True, "agent_id": agent.agent_id, "name": agent.name}
        
        try:
            result = agent.execute_task(task_type, input_data)
            
            if result is None:
                return {"success": False, "error": "Agent returned None"}
            
            if "error" in result:
                return {"success": False, "error": result["error"]}
            
            expected_keys = test_case.get("expected_output_keys", [])
            for key in expected_keys:
                if key not in result:
                    return {"success": False, "error": f"Missing expected key: {key}", "got_keys": list(result.keys())}
            
            return {"success": True, "result": result}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_validation_result(self, validation_id: str) -> Optional[ValidationResult]:
        return self.validation_results.get(validation_id)
    
    def get_validation_results(self) -> List[ValidationResult]:
        return list(self.validation_results.values())
    
    def get_validation_results_by_status(self, status: ValidationStatus) -> List[ValidationResult]:
        return [r for r in self.validation_results.values() if r.status == status]
    
    def validate_performance_watcher(self, agent: BaseAgent) -> ValidationResult:
        test_cases = [
            {
                "name": "test_benchmark_testing",
                "task_type": "benchmark_testing",
                "input_data": {
                    "source_code": """def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total
""",
                    "execution_context": {"environment": "test"},
                    "performance_baseline": {"expected_time_ms": 100}
                },
                "expected_output_keys": ["execution_time", "bottlenecks", "optimization_suggestions"]
            },
            {
                "name": "test_performance_analysis",
                "task_type": "performance_analysis",
                "input_data": {
                    "source_code": """def fast_function(a, b):
    return a * b
""",
                    "execution_context": {"environment": "test"}
                },
                "expected_output_keys": ["execution_time", "bottlenecks", "optimization_suggestions"]
            },
            {
                "name": "test_bottleneck_detection",
                "task_type": "bottleneck_detection",
                "input_data": {
                    "source_code": """def nested_loop():
    result = []
    for i in range(100):
        for j in range(100):
            for k in range(100):
                result.append(i + j + k)
    return result
"""
                },
                "expected_output_keys": ["bottlenecks"]
            }
        ]
        
        role_def = RoleDefinition(
            role_id="",
            name="PerformanceWatcher",
            category=None,
            description="Performance Watcher",
            responsibilities=["run_benchmark", "analyze_performance"],
            capabilities=["benchmark_testing", "performance_analysis", "bottleneck_detection"],
            input_spec={"source_code": "", "execution_context": {}, "performance_baseline": {}},
            output_spec={"execution_time": 0.0, "bottlenecks": [], "optimization_suggestions": []},
            required_skills=["profiling"]
        )
        
        validation = ValidationResult(
            validation_id="",
            role_definition=role_def,
            status=ValidationStatus.RUNNING
        )
        
        start_time = time.time()
        
        for test_case in test_cases:
            validation.tests_run += 1
            try:
                result = self._execute_test(agent, test_case)
                if result.get("success", False):
                    validation.tests_passed += 1
                else:
                    validation.tests_failed += 1
                    validation.errors.append(result.get("error", "Test failed"))
            except Exception as e:
                validation.tests_failed += 1
                validation.errors.append(f"Exception: {str(e)}")
        
        validation.execution_time = time.time() - start_time
        
        if validation.tests_failed == 0:
            validation.status = ValidationStatus.PASSED
        else:
            validation.status = ValidationStatus.FAILED
        
        self.validation_results[validation.validation_id] = validation
        
        return validation
    
    def get_stats(self) -> Dict[str, Any]:
        total = len(self.validation_results)
        passed = len(self.get_validation_results_by_status(ValidationStatus.PASSED))
        failed = len(self.get_validation_results_by_status(ValidationStatus.FAILED))
        error = len(self.get_validation_results_by_status(ValidationStatus.ERROR))
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "error": error,
            "success_rate": passed / total if total > 0 else 0.0,
            "avg_execution_time": sum(r.execution_time for r in self.validation_results.values()) / total if total > 0 else 0.0
        }


from enum import Enum
ValidationStatus.__module__ = __name__