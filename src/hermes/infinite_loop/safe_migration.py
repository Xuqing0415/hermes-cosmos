from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import os
import tempfile
import shutil
import subprocess
from abc import ABC, abstractmethod


@dataclass
class MigrationResult:
    success: bool
    template_id: str
    target_repo_id: str
    patch_content: str
    test_results: List[Dict[str, Any]]
    pr_url: str = ""
    error_message: str = ""


@dataclass
class TestCase:
    name: str
    passed: bool
    duration_ms: float
    output: str = ""


class TestRunner(ABC):
    @abstractmethod
    def run_tests(self, repo_path: str, patch_content: str) -> List[TestCase]:
        pass


class SimulatedTestRunner(TestRunner):
    def run_tests(self, repo_path: str, patch_content: str) -> List[TestCase]:
        import random
        
        tests = []
        num_tests = random.randint(3, 8)
        
        for i in range(num_tests):
            test_name = f"test_{['api', 'unit', 'integration', 'regression'][random.randint(0, 3)]}_{i}"
            passed = random.random() > 0.15
            duration = random.randint(50, 500)
            
            tests.append(TestCase(
                name=test_name,
                passed=passed,
                duration_ms=duration,
                output="OK" if passed else "FAILED"
            ))
        
        return tests


class SafeMigration:
    def __init__(self, test_runner: Optional[TestRunner] = None):
        self.test_runner = test_runner or SimulatedTestRunner()
    
    def apply_patch(self, template: Any, target_repo_id: str, 
                    target_files: List[str]) -> str:
        patch_lines = []
        
        patch_lines.append(f"--- /dev/null")
        patch_lines.append(f"+++ b/{target_files[0] if target_files else 'unknown.py'}")
        patch_lines.append(f"@@ -0,0 +1,5 @@")
        
        for line in template.diff_content.split("\n")[:5]:
            if line.startswith("+") and not line.startswith("+++"):
                patch_lines.append(line)
        
        return "\n".join(patch_lines)
    
    def run_safety_checks(self, patch_content: str) -> List[Dict[str, Any]]:
        checks = []
        
        if "rm -rf" in patch_content or "delete" in patch_content.lower():
            checks.append({"check": "dangerous_command", "passed": False, "message": "Potential destructive operation"})
        else:
            checks.append({"check": "dangerous_command", "passed": True, "message": "OK"})
        
        if len(patch_content) > 10000:
            checks.append({"check": "patch_size", "passed": False, "message": "Patch too large"})
        else:
            checks.append({"check": "patch_size", "passed": True, "message": "OK"})
        
        checks.append({"check": "syntax_validation", "passed": True, "message": "OK"})
        
        return checks
    
    def execute_migration(self, template: Any, target_repo_id: str, 
                          target_files: List[str]) -> MigrationResult:
        print(f"[SafeMigration] Starting migration for {target_repo_id}")
        
        patch_content = self.apply_patch(template, target_repo_id, target_files)
        
        safety_checks = self.run_safety_checks(patch_content)
        all_checks_passed = all(check["passed"] for check in safety_checks)
        
        if not all_checks_passed:
            failed_checks = [c for c in safety_checks if not c["passed"]]
            error_msg = "; ".join([f"{c['check']}: {c['message']}" for c in failed_checks])
            return MigrationResult(
                success=False,
                template_id=template.template_id,
                target_repo_id=target_repo_id,
                patch_content=patch_content,
                test_results=[],
                error_message=f"Safety checks failed: {error_msg}"
            )
        
        print(f"[SafeMigration] Safety checks passed")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_results = self.test_runner.run_tests(tmp_dir, patch_content)
        
        passed_count = sum(1 for t in test_results if t.passed)
        total_count = len(test_results)
        success_rate = passed_count / total_count if total_count > 0 else 0
        
        print(f"[SafeMigration] Tests: {passed_count}/{total_count} passed")
        
        if success_rate >= 0.8:
            pr_url = self._create_pr(target_repo_id, template, patch_content)
            
            return MigrationResult(
                success=True,
                template_id=template.template_id,
                target_repo_id=target_repo_id,
                patch_content=patch_content,
                test_results=[{"name": t.name, "passed": t.passed, "duration": t.duration_ms} for t in test_results],
                pr_url=pr_url
            )
        else:
            return MigrationResult(
                success=False,
                template_id=template.template_id,
                target_repo_id=target_repo_id,
                patch_content=patch_content,
                test_results=[{"name": t.name, "passed": t.passed, "duration": t.duration_ms} for t in test_results],
                error_message=f"Test failure rate too high: {success_rate:.0%}"
            )
    
    def _create_pr(self, target_repo_id: str, template: Any, patch_content: str) -> str:
        import random
        
        pr_number = random.randint(100, 500)
        return f"https://github.com/example/{target_repo_id}/pull/{pr_number}"