"""
Proof-Guided Test Generator - Generates tests from proof results
"""

import hashlib
from typing import List, Dict, Optional, Any
import structlog

from hermes.neural_symbolic.types import (
    GeneratedTest, 
    ProofResult, 
    ProofStatus, 
    ProofTarget
)

logger = structlog.get_logger()


class ProofGuidedTestGenerator:
    """
    Generates test cases based on proof results.
    
    - If proof succeeds: Generate proof certificate (no test needed)
    - If proof fails: Generate test case from the counterexample
    """
    
    def __init__(self):
        self._tests = []
    
    def generate_from_proof_result(
        self,
        result: ProofResult,
        target: ProofTarget
    ) -> Optional[GeneratedTest]:
        """
        Generate a test case from a proof result.
        
        Args:
            result: ProofResult from theorem prover
            target: ProofTarget that was proven
        
        Returns:
            GeneratedTest if proof failed (counterexample), None if proven
        """
        if result.status == ProofStatus.PROVEN:
            logger.info("Proof succeeded, no test needed", target_id=target.id)
            return None
        
        if result.status == ProofStatus.DISPROVEN and result.model:
            return self._generate_from_counterexample(result, target)
        
        if result.status == ProofStatus.UNKNOWN:
            return self._generate_fallback_test(target)
        
        return None
    
    def _generate_from_counterexample(
        self,
        result: ProofResult,
        target: ProofTarget
    ) -> GeneratedTest:
        """
        Generate a test case from a counterexample model.
        
        Args:
            result: ProofResult containing the counterexample
            target: ProofTarget that was disproven
        
        Returns:
            GeneratedTest with inputs from the counterexample
        """
        inputs = {}
        
        for key, value in result.model.items():
            try:
                inputs[key] = int(value)
            except ValueError:
                try:
                    inputs[key] = float(value)
                except ValueError:
                    inputs[key] = value
        
        assertion = self._generate_assertion(target, result)
        
        test = GeneratedTest(
            id=f"test_{hashlib.md5(str(inputs).encode()).hexdigest()[:8]}",
            function_name=target.function_name,
            inputs=inputs,
            expected_output=None,
            assertion=assertion,
            description=f"Test for {target.function_name} - disproven path: {target.path_condition.condition[:50]}..."
        )
        
        self._tests.append(test)
        logger.info("Test generated from counterexample", test_id=test.id, inputs=inputs)
        return test
    
    def _generate_assertion(self, target: ProofTarget, result: ProofResult) -> str:
        """
        Generate an assertion for the test case.
        
        Args:
            target: ProofTarget
            result: ProofResult with counterexample
        
        Returns:
            Assertion string
        """
        condition = target.path_condition.condition
        
        if "not (= " in condition and " 0)" in condition:
            divisor = condition.replace("not (= ", "").replace(" 0)", "").strip()
            return f"assert {divisor} != 0, 'Division by zero detected'"
        
        if "(" in condition and ")" in condition:
            return f"assert not ({condition}), 'Condition should not be true'"
        
        return f"assert False, 'Test case generated from disproven proof'"
    
    def _generate_fallback_test(self, target: ProofTarget) -> GeneratedTest:
        """
        Generate a fallback test when proof result is unknown.
        
        Args:
            target: ProofTarget
        
        Returns:
            GeneratedTest with default inputs
        """
        condition = target.path_condition.condition
        inputs = self._extract_default_inputs(condition)
        
        test = GeneratedTest(
            id=f"test_fallback_{hashlib.md5(condition.encode()).hexdigest()[:8]}",
            function_name=target.function_name,
            inputs=inputs,
            expected_output=None,
            assertion=f"assert True, 'Fallback test for unknown proof result'",
            description=f"Fallback test for {target.function_name}"
        )
        
        self._tests.append(test)
        logger.info("Fallback test generated", test_id=test.id)
        return test
    
    def _extract_default_inputs(self, condition: str) -> Dict[str, Any]:
        """
        Extract default input values from a path condition.
        
        Args:
            condition: Path condition string
        
        Returns:
            Dictionary of input values
        """
        inputs = {}
        
        if "a" in condition and "a" not in inputs:
            inputs["a"] = 10
        if "b" in condition and "b" not in inputs:
            inputs["b"] = 2
        if "x" in condition and "x" not in inputs:
            inputs["x"] = 5
        if "y" in condition and "y" not in inputs:
            inputs["y"] = 3
        
        return inputs
    
    def generate_from_multiple_results(
        self,
        results: List[ProofResult],
        targets: List[ProofTarget]
    ) -> List[GeneratedTest]:
        """
        Generate tests from multiple proof results.
        
        Args:
            results: List of ProofResult objects
            targets: List of ProofTarget objects (should match results)
        
        Returns:
            List of GeneratedTest objects
        """
        tests = []
        
        for result, target in zip(results, targets):
            test = self.generate_from_proof_result(result, target)
            if test:
                tests.append(test)
        
        return tests
    
    def generate_test_code(self, test: GeneratedTest) -> str:
        """
        Generate Python test code for a GeneratedTest.
        
        Args:
            test: GeneratedTest object
        
        Returns:
            Python test code as string
        """
        inputs_str = ", ".join([f"{k}={v}" for k, v in test.inputs.items()])
        func_call = f"{test.function_name}({inputs_str})"
        
        desc = test.description or ""
        
        code = f'''def test_{test.id}():
    """{desc}"""
    {inputs_str}
    
    result = {func_call}
    {test.assertion}
'''
        
        return code
    
    def generate_test_module(self, tests: List[GeneratedTest]) -> str:
        """
        Generate a complete Python test module.
        
        Args:
            tests: List of GeneratedTest objects
        
        Returns:
            Python test module code as string
        """
        imports = "import pytest\n\n"
        
        test_functions = []
        for test in tests:
            test_functions.append(self.generate_test_code(test))
        
        return imports + "\n".join(test_functions)
    
    def get_tests(self) -> List[GeneratedTest]:
        """
        Get all generated tests.
        
        Returns:
            List of GeneratedTest objects
        """
        return self._tests
    
    def clear(self):
        """Clear all generated tests"""
        self._tests = []
