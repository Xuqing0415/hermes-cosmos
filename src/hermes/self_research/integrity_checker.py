"""研究完整性检查：论文生成前的最后一道闸门。

扫描所有发现与数据来源标注，凡是“关键结论依赖非真实数据”的，都必须在论文里
显式声明，而不是悄悄当成结论发表。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from hermes.self_research.data_provenance import (
    PROVENANCE_LABELS,
    SOURCE_LABELS,
    ProvenanceReport,
)
from hermes.self_research.evidence_grader import GRADE_A, EvidenceGrade

# 论文的“核心结论”类别：这些类别一旦降级，必须给出显著声明
CRITICAL_CATEGORIES = ("evolution", "transfer")

CATEGORY_SECTIONS = {
    "evolution": "5 结果与分析（演化阶段）",
    "transfer": "5 结果与分析（跨域迁移）",
    "strategy": "5 结果与分析（策略收益）",
    "capability": "5 结果与分析（基础能力）",
    "metacognition": "5 结果与分析（自我认知）",
    "self_repair": "5 结果与分析（自修复）",
    "policy": "6 讨论（策略权重）",
    "other": "6 讨论",
}


@dataclass
class Disclaimer:
    finding_ids: List[str]
    severity: str
    text: str
    target: str = "6 讨论"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_ids": self.finding_ids,
            "severity": self.severity,
            "text": self.text,
            "target": self.target,
        }


@dataclass
class IntegrityReport:
    grades: List[EvidenceGrade] = field(default_factory=list)
    disclaimers: List[Disclaimer] = field(default_factory=list)
    data_warnings: List[str] = field(default_factory=list)
    critical_findings: List[str] = field(default_factory=list)
    missing_sources: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return not self.disclaimers and not self.data_warnings

    @property
    def critical_count(self) -> int:
        return sum(1 for item in self.disclaimers if item.severity == "critical")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "critical_count": self.critical_count,
            "grades": [item.to_dict() for item in self.grades],
            "disclaimers": [item.to_dict() for item in self.disclaimers],
            "data_warnings": self.data_warnings,
            "critical_findings": self.critical_findings,
            "missing_sources": self.missing_sources,
            "provenance": self.provenance,
        }


class IntegrityChecker:
    """依据证据等级与数据源可用性，生成强制免责声明。"""

    def __init__(self, critical_categories: Sequence[str] = CRITICAL_CATEGORIES):
        self._critical_categories = tuple(critical_categories)

    def check(
        self,
        grades: List[EvidenceGrade],
        provenance: ProvenanceReport,
    ) -> IntegrityReport:
        report = IntegrityReport(grades=list(grades), provenance=provenance.to_dict())
        report.missing_sources = sorted(provenance.missing_sources)

        for grade in grades:
            if grade.grade == GRADE_A:
                continue

            critical = grade.category in self._critical_categories and grade.grade != GRADE_A
            severity = "critical" if critical else "caution"
            if critical:
                report.critical_findings.append(grade.finding_id)

            label = "结论降级" if critical else "证据降级"
            report.disclaimers.append(
                Disclaimer(
                    finding_ids=[grade.finding_id],
                    severity=severity,
                    text=(
                        f"{label}：发现 {grade.finding_id}「{grade.statement}」的证据等级为 "
                        f"{grade.grade}（{PROVENANCE_LABELS.get(grade.provenance, grade.provenance)}）；"
                        f"原因：{'；'.join(grade.reasons) or '未知'}。该结论待真实数据验证。"
                    ),
                    target=CATEGORY_SECTIONS.get(grade.category, "6 讨论"),
                )
            )

        for name in report.missing_sources:
            label = SOURCE_LABELS.get(name, name)
            report.data_warnings.append(f"数据源不可用：{label}，相关结论未被计算（未做任何填充或推断）。")

        if provenance.git_unavailable:
            report.data_warnings.append("git 历史不可用：本次未划分演化阶段，论文中不包含任何阶段对比结论。")

        for note in provenance.notes:
            if note not in report.data_warnings:
                report.data_warnings.append(note)

        return report

    @staticmethod
    def summary_line(report: IntegrityReport) -> str:
        if report.is_clean:
            return "完整性检查通过：所有结论均基于真实数据。"
        return (
            f"完整性检查发现 {len(report.disclaimers)} 条需声明的降级结论"
            f"（其中关键结论 {report.critical_count} 条）"
            f"、{len(report.data_warnings)} 条数据源警告。"
        )
