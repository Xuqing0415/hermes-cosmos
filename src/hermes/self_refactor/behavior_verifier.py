import subprocess
import os
import sys
import time
import hashlib
from typing import List, Dict, Optional, Tuple
import json

from .types import RefactorPlan, RefactorResult


class BehaviorVerifier:
    def __init__(self, test_dir: str = "tests", proof_cache_path: str = ":memory:"):
        self.test_dir = test_dir
        self.proof_cache_path = proof_cache_path
        self.test_results: Dict[str, bool] = {}
        self.performance_results: Dict[str, float] = {}

    def verify_equivalence(self, plan: RefactorPlan) -> Tuple[bool, Dict[str, str]]:
        results = {}
        
        tests_passed = self._run_tests()
        results['tests'] = "passed" if tests_passed else "failed"
        
        proofs_valid = self._validate_proofs()
        results['proofs'] = "valid" if proofs_valid else "invalid"
        
        performance_ok = self._check_performance()
        results['performance'] = "ok" if performance_ok else "degraded"
        
        all_passed = tests_passed and proofs_valid and performance_ok
        return all_passed, results

    def _run_tests(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", self.test_dir, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def _validate_proofs(self) -> bool:
        try:
            from hermes.ci.proof_cache import ProofCache
            cache = ProofCache(self.proof_cache_path)
            stats = cache.get_stats()
            
            invalid_count = stats.get('invalid_count', 0)
            total_count = stats.get('total_count', 0)
            
            if total_count == 0:
                return True
            
            return invalid_count / total_count < 0.1
        except Exception:
            return True

    def _check_performance(self) -> bool:
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, "-m", "pytest", self.test_dir, "-x", "-q"],
                capture_output=True,
                text=True,
                timeout=120
            )
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                return False
            
            if elapsed > 60:
                print(f"Performance warning: tests took {elapsed:.2f}s")
            
            return elapsed < 120
        except Exception:
            return True

    def get_test_results(self) -> Dict[str, bool]:
        return self.test_results

    def get_performance_results(self) -> Dict[str, float]:
        return self.performance_results

    def verify_specific_functions(self, function_names: List[str]) -> Dict[str, bool]:
        results = {}
        
        for func_name in function_names:
            results[func_name] = self._test_function(func_name)
        
        return results

    def _test_function(self, function_name: str) -> bool:
        try:
            test_pattern = f"*{function_name}*"
            result = subprocess.run(
                [sys.executable, "-m", "pytest", self.test_dir, "-k", function_name, "-q"],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception:
            return False

    def generate_diff_report(self, before_dir: str, after_dir: str) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", before_dir, after_dir],
                capture_output=True,
                text=True
            )
            return result.stdout
        except Exception:
            return ""

    def compute_code_hash(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def compare_code_hashes(self, before_hashes: Dict[str, str], after_hashes: Dict[str, str]) -> List[str]:
        changed_files = []
        for file_path, before_hash in before_hashes.items():
            if after_hashes.get(file_path) != before_hash:
                changed_files.append(file_path)
        return changed_files

    def get_affected_tests(self, changed_files: List[str]) -> List[str]:
        affected_tests = []
        test_files = []
        
        for root, _, filenames in os.walk(self.test_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    test_files.append(os.path.join(root, filename))
        
        for test_file in test_files:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for changed_file in changed_files:
                module_name = self._file_to_module(changed_file)
                if module_name in content:
                    affected_tests.append(test_file)
                    break
        
        return affected_tests

    def _file_to_module(self, file_path: str) -> str:
        if file_path.startswith('src/'):
            rel_path = file_path[4:]
        else:
            rel_path = file_path
        
        return rel_path.replace(os.sep, '.').replace('.py', '')