from typing import Dict, List, Optional, Any
import time

from ..agent_civilization import BaseAgent, AgentRole
from .types import VerificationResult, PassEquivalenceResult, VerificationStatus
from .symbolic_executor import MLIRSymbolicExecutor
from .test_generator import MLIRTestGenerator
from .pass_verifier import MLIRPassVerifier


class CompilerVerifier(BaseAgent):
    """编译器验证官：面向MLIR任务，验证优化Pass的正确性"""
    
    def __init__(self, knowledge_base: Optional[Any] = None):
        super().__init__(
            role=AgentRole.COMPILER_VERIFIER,
            name="CompilerVerifier",
            capabilities=["mlir_verification", "pass_validation", "bug_detection", "equivalence_proof"],
            knowledge_base=knowledge_base
        )
        self.symbolic_executor = MLIRSymbolicExecutor()
        self.test_generator = MLIRTestGenerator()
        self.pass_verifier = MLIRPassVerifier()
    
    def execute_task(self, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if task_type == "mlir_verification":
            return self.verify_mlir(input_data)
        elif task_type == "pass_validation":
            return self.validate_pass(input_data)
        elif task_type == "bug_detection":
            return self.detect_bugs(input_data)
        elif task_type == "equivalence_proof":
            return self.prove_equivalence(input_data)
        elif task_type == "function_verification":
            return self.verify_function(input_data)
        return {"error": f"Unknown task type: {task_type}"}
    
    def verify_mlir(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        mlir_code = input_data.get("mlir_code", "")
        
        if not mlir_code:
            return {"error": "mlir_code is required"}
        
        result = self.symbolic_executor.execute(mlir_code)
        
        test_results = []
        if result.status == VerificationStatus.PASSED:
            tests = self.test_generator.generate_test_inputs(mlir_code, num_tests=5)
            for test in tests:
                test_result = self.test_generator.run_test(mlir_code, test)
                test_results.append({
                    "passed": test_result.passed,
                    "shapes": [inpt.shape for inpt in test_result.input_data]
                })
        
        self.learn_from_experience({
            "action": "mlir_verification",
            "status": result.status.value,
            "errors_count": len(result.errors),
            "warnings_count": len(result.warnings),
            "execution_time": result.execution_time,
            "test_results": test_results
        })
        
        return {
            "status": result.status.value,
            "errors": [{"type": e.type.value, "message": e.message} for e in result.errors],
            "warnings": result.warnings,
            "smt_queries": len(result.smt_queries),
            "execution_time": result.execution_time,
            "test_results": test_results
        }
    
    def verify_function(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        mlir_code = input_data.get("mlir_code", "")
        test_cases = input_data.get("test_cases", [])
        
        if not mlir_code:
            return {"error": "mlir_code is required"}
        
        symbolic_result = self.symbolic_executor.execute(mlir_code)
        
        test_results = []
        for test_case in test_cases:
            result = self.test_generator.run_scalar_test(mlir_code, test_case)
            test_results.append({
                "input": test_case,
                **result
            })
        
        matched_count = sum(1 for r in test_results if r.get("matched", False))
        
        return {
            "symbolic_status": symbolic_result.status.value,
            "symbolic_errors": [{"type": e.type.value, "message": e.message} for e in symbolic_result.errors],
            "test_results": test_results,
            "matched_count": matched_count,
            "total_tests": len(test_results),
            "confidence": matched_count / len(test_results) if test_results else 0.0
        }
    
    def validate_pass(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        mlir_code = input_data.get("mlir_code", "")
        pass_name = input_data.get("pass_name", "canonicalize")
        
        if not mlir_code:
            return {"error": "mlir_code is required"}
        
        result = self.pass_verifier.apply_pass_and_verify(mlir_code, pass_name)
        
        self.learn_from_experience({
            "action": "pass_validation",
            "pass_name": pass_name,
            "is_equivalent": result.is_equivalent,
            "num_tests": result.num_tests,
            "passed_tests": result.passed_tests
        })
        
        return {
            "pass_name": pass_name,
            "is_equivalent": result.is_equivalent,
            "num_tests": result.num_tests,
            "passed_tests": result.passed_tests,
            "diff": result.diff
        }
    
    def detect_bugs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        mlir_code = input_data.get("mlir_code", "")
        
        if not mlir_code:
            return {"error": "mlir_code is required"}
        
        verification_result = self.symbolic_executor.execute(mlir_code)
        
        bugs = []
        for error in verification_result.errors:
            bugs.append({
                "type": error.type.value,
                "message": error.message,
                "severity": "high" if error.type.value == "index_out_of_bounds" else "medium",
                "suggested_fix": self._generate_fix_suggestion(error)
            })
        
        test_suite = self.test_generator.generate_comprehensive_test_suite(mlir_code)
        
        return {
            "bugs_found": bugs,
            "total_bugs": len(bugs),
            "verification_status": verification_result.status.value,
            "test_coverage": test_suite
        }
    
    def _generate_fix_suggestion(self, error: Any) -> str:
        if error.type.value == "index_out_of_bounds":
            return "Check the index calculation and ensure it's within memref bounds. Consider adding bounds checks or adjusting loop limits."
        elif error.type.value == "memory_aliasing":
            return "Ensure that store operations don't overwrite memory that's still being read. Consider using temporary buffers."
        elif error.type.value == "type_mismatch":
            return "Check the types of operands and results. Ensure all operations use consistent types."
        return "Review the operation and fix the identified issue."
    
    def prove_equivalence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        original_code = input_data.get("original_code", "")
        optimized_code = input_data.get("optimized_code", "")
        
        if not original_code or not optimized_code:
            return {"error": "original_code and optimized_code are required"}
        
        result = self.pass_verifier.verify_equivalence(original_code, optimized_code)
        
        return {
            "is_equivalent": result.is_equivalent,
            "num_tests": result.num_tests,
            "passed_tests": result.passed_tests,
            "confidence": result.passed_tests / result.num_tests if result.num_tests > 0 else 0.0,
            "diff": result.diff
        }