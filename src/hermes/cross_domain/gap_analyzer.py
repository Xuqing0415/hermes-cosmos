from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .capability_benchmark import BenchmarkResult, BenchmarkReport
from .knowledge_amalgamator import KnowledgeAmalgamator


@dataclass
class GapAnalysis:
    pattern_type: str
    success_rate: float
    total_cases: int
    passed_cases: int
    avg_confidence: float
    avg_elapsed_ms: float
    domains: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "success_rate": round(self.success_rate, 1),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "avg_confidence": round(self.avg_confidence, 2),
            "avg_elapsed_ms": round(self.avg_elapsed_ms, 1),
            "domains": self.domains,
            "recommendation": self.recommendation,
        }


@dataclass
class GapReport:
    analyses: List[GapAnalysis] = field(default_factory=list)
    weakest_pattern: Optional[str] = None
    weakest_rate: float = 100.0
    strongest_pattern: Optional[str] = None
    strongest_rate: float = 0.0
    overall_success_rate: float = 0.0
    knowledge_coverage: float = 0.0
    suggested_priority: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_success_rate": round(self.overall_success_rate, 1),
            "knowledge_coverage": round(self.knowledge_coverage, 1),
            "weakest_pattern": self.weakest_pattern,
            "weakest_rate": round(self.weakest_rate, 1),
            "strongest_pattern": self.strongest_pattern,
            "strongest_rate": round(self.strongest_rate, 1),
            "suggested_priority": self.suggested_priority,
            "analyses": [a.to_dict() for a in self.analyses],
        }


class GapAnalyzer:
    def __init__(self, knowledge_amalgamator: Optional[KnowledgeAmalgamator] = None):
        self._knowledge = knowledge_amalgamator

    def analyze(self, report: BenchmarkReport) -> GapReport:
        gap_report = GapReport()

        pattern_results: Dict[str, List[BenchmarkResult]] = {}
        for result in report.results:
            ptype = result.expected_abstract_type
            if ptype not in pattern_results:
                pattern_results[ptype] = []
            pattern_results[ptype].append(result)

        all_passed = sum(1 for r in report.results if r.passed())
        all_total = len(report.results)
        gap_report.overall_success_rate = (all_passed / all_total * 100) if all_total > 0 else 0.0

        for ptype, results in sorted(pattern_results.items()):
            total = len(results)
            passed = sum(1 for r in results if r.passed())
            success_rate = (passed / total * 100) if total > 0 else 0.0
            avg_conf = sum(r.confidence for r in results) / total if total > 0 else 0.0
            avg_time = sum(r.elapsed_ms for r in results) / total if total > 0 else 0.0
            domains = list(set(r.domain for r in results))

            if success_rate < gap_report.weakest_rate:
                gap_report.weakest_rate = success_rate
                gap_report.weakest_pattern = ptype
            if success_rate > gap_report.strongest_rate and total >= 1:
                gap_report.strongest_rate = success_rate
                gap_report.strongest_pattern = ptype

            analysis = GapAnalysis(
                pattern_type=ptype,
                success_rate=success_rate,
                total_cases=total,
                passed_cases=passed,
                avg_confidence=avg_conf,
                avg_elapsed_ms=avg_time,
                domains=domains,
                recommendation=self._generate_recommendation(ptype, success_rate),
            )
            gap_report.analyses.append(analysis)

        gap_report.suggested_priority = self._compute_priorities(gap_report.analyses)

        if self._knowledge is not None:
            known_types = set()
            for node in self._knowledge.get_graph().nodes:
                known_types.add(node.pattern_type.value)

            analyzed_types = set(a.pattern_type for a in gap_report.analyses)
            covered = analyzed_types & known_types
            gap_report.knowledge_coverage = (
                len(covered) / len(analyzed_types) * 100
            ) if analyzed_types else 0.0

        return gap_report

    def _generate_recommendation(self, pattern_type: str, success_rate: float) -> str:
        if success_rate >= 80.0:
            return f"{pattern_type}表现良好，维持当前策略"
        elif success_rate >= 50.0:
            return f"Increase sampling weight for {pattern_type} from 0.1 to {0.1 + (80 - success_rate) / 100:.1f}"
        else:
            return f"URGENT: {pattern_type} success rate critically low ({success_rate:.0f}%). Consider adding new training data."

    def _compute_priorities(self, analyses: List[GapAnalysis]) -> List[Dict[str, Any]]:
        priorities = []
        for analysis in analyses:
            if analysis.success_rate < 80:
                urgency = "high" if analysis.success_rate < 50 else "medium"
                weight_delta = (80 - analysis.success_rate) / 100
                priorities.append({
                    "pattern_type": analysis.pattern_type,
                    "urgency": urgency,
                    "current_success_rate": round(analysis.success_rate, 1),
                    "recommended_weight_increase": round(weight_delta, 2),
                    "current_weight": round(0.1, 2),
                    "new_weight": round(0.1 + weight_delta, 2),
                })

        priorities.sort(key=lambda x: x["current_success_rate"])
        return priorities