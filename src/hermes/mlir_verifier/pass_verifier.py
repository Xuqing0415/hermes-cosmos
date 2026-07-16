from typing import Dict, List, Optional, Any
import re
import random
import time

from .types import PassEquivalenceResult, VerificationResult, ErrorType
from .test_generator import MLIRTestGenerator


class MLIRPassVerifier:
    def __init__(self):
        self.test_generator = MLIRTestGenerator()
        self.pass_patterns = {
            "canonicalize": self._apply_canonicalize,
            "constant_fold": self._apply_constant_fold,
            "memref_copy": self._apply_memref_copy,
            "inline": self._apply_inline
        }
    
    def verify_equivalence(self, original_mlir: str, optimized_mlir: str, 
                           pass_name: str = "unknown") -> PassEquivalenceResult:
        test_suite = self.test_generator.generate_test_inputs(original_mlir, num_tests=5)
        
        original_results = []
        optimized_results = []
        
        for test in test_suite:
            original_result = self.test_generator.run_test(original_mlir, test)
            optimized_result = self.test_generator.run_test(optimized_mlir, test)
            
            original_results.append(original_result)
            optimized_results.append(optimized_result)
        
        is_equivalent = all(o.passed == opt.passed for o, opt in zip(original_results, optimized_results))
        
        diff = None
        if not is_equivalent:
            diff = self._generate_diff(original_mlir, optimized_mlir)
        
        return PassEquivalenceResult(
            pass_name=pass_name,
            original_output=original_results,
            optimized_output=optimized_results,
            is_equivalent=is_equivalent,
            diff=diff,
            num_tests=len(test_suite),
            passed_tests=sum(1 for o, opt in zip(original_results, optimized_results) if o.passed == opt.passed)
        )
    
    def apply_pass_and_verify(self, mlir_code: str, pass_name: str) -> PassEquivalenceResult:
        if pass_name not in self.pass_patterns:
            return PassEquivalenceResult(
                pass_name=pass_name,
                original_output=None,
                optimized_output=None,
                is_equivalent=False,
                diff=f"Unknown pass: {pass_name}"
            )
        
        optimized_mlir = self.pass_patterns[pass_name](mlir_code)
        
        return self.verify_equivalence(mlir_code, optimized_mlir, pass_name)
    
    def _apply_canonicalize(self, mlir_code: str) -> str:
        mlir_code = re.sub(r'arith\.constant 0 : (\w+)', r'arith.constant 0 : \1', mlir_code)
        mlir_code = re.sub(r'arith\.constant 1 : (\w+)', r'arith.constant 1 : \1', mlir_code)
        
        return mlir_code
    
    def _apply_constant_fold(self, mlir_code: str) -> str:
        lines = mlir_code.split('\n')
        constants = {}
        new_lines = []
        
        for line in lines:
            const_match = re.match(r'%(\w+)\s*=\s*arith\.constant\s*(\d+)\s*:\s*(\w+)', line)
            if const_match:
                name = const_match.group(1)
                value = int(const_match.group(2))
                dtype = const_match.group(3)
                constants[name] = (value, dtype)
                new_lines.append(line)
                continue
            
            add_match = re.match(r'%(\w+)\s*=\s*arith\.addi\s*%(\w+),\s*%(\w+)\s*:\s*(\w+)', line)
            if add_match and add_match.group(2) in constants and add_match.group(3) in constants:
                val1, _ = constants[add_match.group(2)]
                val2, _ = constants[add_match.group(3)]
                result = val1 + val2
                new_lines.append(f'%{add_match.group(1)} = arith.constant {result} : {add_match.group(4)}')
                continue
            
            mul_match = re.match(r'%(\w+)\s*=\s*arith\.muli\s*%(\w+),\s*%(\w+)\s*:\s*(\w+)', line)
            if mul_match and mul_match.group(2) in constants and mul_match.group(3) in constants:
                val1, _ = constants[mul_match.group(2)]
                val2, _ = constants[mul_match.group(3)]
                result = val1 * val2
                new_lines.append(f'%{mul_match.group(1)} = arith.constant {result} : {mul_match.group(4)}')
                continue
            
            new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def _apply_memref_copy(self, mlir_code: str) -> str:
        return mlir_code
    
    def _apply_inline(self, mlir_code: str) -> str:
        return mlir_code
    
    def _generate_diff(self, original: str, optimized: str) -> str:
        original_lines = original.split('\n')
        optimized_lines = optimized.split('\n')
        
        max_len = max(len(original_lines), len(optimized_lines))
        diff_lines = []
        
        for i in range(max_len):
            orig_line = original_lines[i].strip() if i < len(original_lines) else ""
            opt_line = optimized_lines[i].strip() if i < len(optimized_lines) else ""
            
            if orig_line != opt_line:
                diff_lines.append(f"- {orig_line}")
                diff_lines.append(f"+ {opt_line}")
        
        return '\n'.join(diff_lines)
    
    def batch_verify_passes(self, mlir_code: str, passes: List[str]) -> Dict[str, PassEquivalenceResult]:
        results = {}
        for pass_name in passes:
            result = self.apply_pass_and_verify(mlir_code, pass_name)
            results[pass_name] = result
        
        return results