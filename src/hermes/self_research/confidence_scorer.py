"""论文置信度评分。

分数由两部分构成，权重固定、公式公开：

    置信度 = 0.7 x 证据分 + 0.3 x 数据覆盖率 - 0.05 x 关键降级结论数

* 证据分：各发现证据等级的加权均值（A=1.0，B=0.6，C=0.3）
* 数据覆盖率：可用的主数据源数量 / 期望的主数据源数量

分数越低，说明这篇论文越依赖降级或缺失的数据；输出里同时给出“补齐什么能把分数提到多少”。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from hermes.self_research.data_provenance import SOURCE_LABELS, ProvenanceReport
from hermes.self_research.evidence_grader import GRADE_A, GRADE_B, GRADE_C, EvidenceGrade

GRADE_WEIGHTS = {GRADE_A: 1.0, GRADE_B: 0.6, GRADE_C: 0.3}

EVIDENCE_WEIGHT = 0.7
COVERAGE_WEIGHT = 0.3
CRITICAL_PENALTY = 0.05

EXPECTED_PRIMARY_SOURCES = ("snapshot_db", "notification_log", "policy", "knowledge_graph", "git_history")


@dataclass
class ConfidenceBreakdown:
    score: float
    evidence_score: float
    coverage: float
    grade_mix: Dict[str, int] = field(default_factory=dict)
    available_sources: List[str] = field(default_factory=list)
    missing_sources: List[str] = field(default_factory=list)
    penalty: float = 0.0
    potential_score: float = 0.0
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "coverage": round(self.coverage, 4),
            "grade_mix": self.grade_mix,
            "available_sources": self.available_sources,
            "missing_sources": self.missing_sources,
            "penalty": round(self.penalty, 4),
            "potential_score": round(self.potential_score, 4),
            "suggestions": self.suggestions,
        }


class ConfidenceScorer:
    def __init__(
        self,
        evidence_weight: float = EVIDENCE_WEIGHT,
        coverage_weight: float = COVERAGE_WEIGHT,
        critical_penalty: float = CRITICAL_PENALTY,
        expected_sources: Sequence[str] = EXPECTED_PRIMARY_SOURCES,
    ):
        self._evidence_weight = evidence_weight
        self._coverage_weight = coverage_weight
        self._critical_penalty = critical_penalty
        self._expected_sources = tuple(expected_sources)

    def score(
        self,
        grades: Sequence[EvidenceGrade],
        provenance: ProvenanceReport,
        critical_count: int = 0,
    ) -> ConfidenceBreakdown:
        evidence = self._evidence_score(grades)
        coverage, available, missing = self._coverage(provenance)

        penalty = self._critical_penalty * max(0, critical_count)
        score = self._clamp(self._evidence_weight * evidence + self._coverage_weight * coverage - penalty)

        breakdown = ConfidenceBreakdown(
            score=score,
            evidence_score=evidence,
            coverage=coverage,
            grade_mix=self._grade_mix(grades),
            available_sources=available,
            missing_sources=missing,
            penalty=penalty,
            potential_score=self._clamp(self._evidence_weight + self._coverage_weight),
        )
        breakdown.suggestions = self._suggestions(grades, provenance, breakdown)
        return breakdown

    # ------------------------------------------------------------------ 分项

    @staticmethod
    def _evidence_score(grades: Sequence[EvidenceGrade]) -> float:
        if not grades:
            return 0.0
        total = sum(GRADE_WEIGHTS.get(item.grade, 0.0) for item in grades)
        return total / len(grades)

    def _coverage(self, provenance: ProvenanceReport) -> tuple:
        if not self._expected_sources:
            return 1.0, [], []
        missing = [name for name in self._expected_sources if name in provenance.missing_sources]
        available = [name for name in self._expected_sources if name not in missing]
        return len(available) / len(self._expected_sources), available, missing

    @staticmethod
    def _grade_mix(grades: Sequence[EvidenceGrade]) -> Dict[str, int]:
        mix = {GRADE_A: 0, GRADE_B: 0, GRADE_C: 0}
        for item in grades:
            mix[item.grade] = mix.get(item.grade, 0) + 1
        return mix

    def _suggestions(
        self,
        grades: Sequence[EvidenceGrade],
        provenance: ProvenanceReport,
        breakdown: ConfidenceBreakdown,
    ) -> List[str]:
        suggestions: List[str] = []

        weak = [item for item in grades if item.grade != GRADE_A]
        if weak:
            ids = "、".join(item.finding_id for item in weak[:5])
            upgraded = self._clamp(
                self._evidence_weight * 1.0 + self._coverage_weight * breakdown.coverage - breakdown.penalty
            )
            suggestions.append(f"补齐 {ids} 的原始数据后置信度可提升至 {upgraded:.2f}")

        if breakdown.missing_sources:
            labels = "、".join(SOURCE_LABELS.get(name, name) for name in breakdown.missing_sources)
            upgraded = self._clamp(
                self._evidence_weight * breakdown.evidence_score + self._coverage_weight * 1.0 - breakdown.penalty
            )
            suggestions.append(f"补齐数据源（{labels}）后置信度可提升至 {upgraded:.2f}")

        if provenance.git_unavailable:
            suggestions.append(
                f"补充真实 git 分析以恢复演化阶段结论，覆盖率可从 " f"{breakdown.coverage:.0%} 提升到 100%"
            )

        return suggestions

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
