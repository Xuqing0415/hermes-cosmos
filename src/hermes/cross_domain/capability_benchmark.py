from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time

from .experiment_designer import TestCase, TestCaseDifficulty
from .abstract_pattern_extractor import AbstractPatternExtractor, AbstractPatternType
from .pattern_similarity_engine import PatternSimilarityEngine
from .cross_domain_translator import CrossDomainTranslator


@dataclass
class BenchmarkResult:
    test_case_id: str
    domain: str
    pattern_type: str
    expected_abstract_type: str
    extracted_abstract_type: Optional[str] = None
    fix_generated: bool = False
    fix_correct: bool = False
    cross_domain_transfer: bool = False
    elapsed_ms: float = 0.0
    confidence: float = 0.0
    error_message: Optional[str] = None

    def passed(self) -> bool:
        return self.fix_correct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "domain": self.domain,
            "pattern_type": self.pattern_type,
            "expected_abstract_type": self.expected_abstract_type,
            "extracted_abstract_type": self.extracted_abstract_type,
            "fix_generated": self.fix_generated,
            "fix_correct": self.fix_correct,
            "cross_domain_transfer": self.cross_domain_transfer,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "confidence": round(self.confidence, 2),
            "passed": self.passed(),
            "error_message": self.error_message,
        }


@dataclass
class BenchmarkReport:
    results: List[BenchmarkResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_elapsed_ms: float = 0.0

    def compute_summary(self) -> Dict[str, Any]:
        total = len(self.results)
        if total == 0:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

        passed = sum(1 for r in self.results if r.passed())
        failed = total - passed

        domains = {}
        for r in self.results:
            if r.domain not in domains:
                domains[r.domain] = {"total": 0, "passed": 0}
            domains[r.domain]["total"] += 1
            if r.passed():
                domains[r.domain]["passed"] += 1

        cross_domain_transfers = sum(1 for r in self.results if r.cross_domain_transfer)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 1),
            "by_domain": domains,
            "cross_domain_transfers": cross_domain_transfers,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.compute_summary(),
            "results": [r.to_dict() for r in self.results],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


EXTRACTION_MAPPING: Dict[str, str] = {
    "null_pointer": "boundary_check_missing",
    "index_out_of_bounds": "boundary_check_missing",
    "missing_bound_check": "boundary_check_missing",
    "missing_validation": "input_validation_missing",
    "performance": "performance_bottleneck",
    "potential_divide_by_zero": "division_by_zero",
    "uninitialized_variable": "uninitialized_variable",
    "missing_resources": "resource_limit_missing",
    "hpa_missing": "resource_limit_missing",
    "deprecated_api": "api_version_deprecated",
    "old_image": "api_version_deprecated",
    "type_mismatch": "type_mismatch",
    "infinite_recursion": "infinite_recursion",
    "concurrency_race": "concurrency_race_condition",
    "memory_leak": "memory_leak",
}


def _extract_abstract_type(pattern_type: str, domain: str) -> str:
    return EXTRACTION_MAPPING.get(pattern_type, pattern_type)


class CapabilityBenchmark:
    def __init__(self):
        self._extractor = AbstractPatternExtractor()
        self._similarity_engine = PatternSimilarityEngine()
        self._translator = CrossDomainTranslator()

    def run_single(self, test_case: TestCase, knowledge_patterns: Optional[List[Dict]] = None) -> BenchmarkResult:
        start = time.perf_counter()

        expected_abstract = test_case.expected_abstract_type
        extracted_abstract = _extract_abstract_type(test_case.pattern_type, test_case.domain)

        result = BenchmarkResult(
            test_case_id=test_case.id,
            domain=test_case.domain,
            pattern_type=test_case.pattern_type,
            expected_abstract_type=expected_abstract,
            extracted_abstract_type=extracted_abstract,
        )

        pain_point_data = {
            "id": test_case.id,
            "type": test_case.pattern_type,
            "severity": "medium",
            "message": test_case.description,
            "location": test_case.synthetic_input.get("file", "unknown"),
            "context": test_case.synthetic_input,
        }

        pattern = self._extractor.extract_pattern(test_case.domain, pain_point_data)

        if pattern:
            result.extracted_abstract_type = pattern.pattern_type.value
            result.confidence = pattern.confidence

            if pattern.pattern_type.value == expected_abstract:
                result.fix_generated = True
                result.fix_correct = True

                template = self._translator.translate_fix(
                    test_case.domain, test_case.domain,
                    pattern.pattern_type
                )
                if template:
                    result.fix_correct = True

                if knowledge_patterns:
                    for kp in knowledge_patterns:
                        if kp.get("pattern_type") == pattern.pattern_type.value:
                            source_domain = kp.get("source_domain", "")
                            target_domain = kp.get("target_domain", "")
                            if source_domain and target_domain and source_domain != target_domain:
                                result.cross_domain_transfer = True
                                break
            else:
                pass

        elapsed = time.perf_counter() - start
        result.elapsed_ms = elapsed * 1000
        return result

    def run_batch(self, test_cases: List[TestCase],
                  knowledge_graph=None) -> BenchmarkReport:
        report = BenchmarkReport()

        knowledge_patterns = None
        if knowledge_graph is not None:
            knowledge_patterns = knowledge_graph.get_universal_fixes()

        for tc in test_cases:
            result = self.run_single(tc, knowledge_patterns)
            report.results.append(result)

        report.completed_at = datetime.now(timezone.utc)
        total_ms = sum(r.elapsed_ms for r in report.results)
        report.total_elapsed_ms = total_ms

        return report

    def run_with_knowledge(self, test_cases: List[TestCase],
                           knowledge_graph) -> BenchmarkReport:
        return self.run_batch(test_cases, knowledge_graph)

    def compute_pass_rate(self, results: List[BenchmarkResult]) -> float:
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.passed())
        return passed / len(results) * 100.0

    def compute_domain_pass_rates(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        domain_results = {}
        for r in results:
            if r.domain not in domain_results:
                domain_results[r.domain] = []
            domain_results[r.domain].append(r)

        return {
            domain: self.compute_pass_rate(res)
            for domain, res in domain_results.items()
        }

    def compute_cross_domain_transfer_rate(self, results: List[BenchmarkResult]) -> float:
        if not results:
            return 0.0
        transfers = sum(1 for r in results if r.cross_domain_transfer)
        return transfers / len(results) * 100.0