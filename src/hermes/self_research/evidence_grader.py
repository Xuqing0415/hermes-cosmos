"""证据等级评定：对每条发现给出 A / B / C。

* A —— 全部输入数据为 REAL，且样本量达到该结论所需的最低要求
* B —— 输入含 FALLBACK / UNVERIFIED 数据，或存在非样本类的保留意见
* C —— 输入含 SYNTHETIC 数据，或样本量不足以支撑该结论

评级只看“数据可不可信 + 样本够不够”，不看结论好不好看。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes.self_research.data_provenance import (
    FALLBACK,
    PROVENANCE_LABELS,
    SYNTHETIC,
    UNVERIFIED,
    ProvenanceReport,
)

GRADE_A = "A"
GRADE_B = "B"
GRADE_C = "C"

GRADE_LABELS = {
    GRADE_A: "A（全部真实数据）",
    GRADE_B: "B（含降级或未标注数据）",
    GRADE_C: "C（含合成数据或样本不足）",
}


@dataclass
class EvidenceGrade:
    finding_id: str
    statement: str
    grade: str
    provenance: str
    reasons: List[str] = field(default_factory=list)
    category: str = "other"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "statement": self.statement,
            "grade": self.grade,
            "provenance": self.provenance,
            "provenance_label": PROVENANCE_LABELS.get(self.provenance, self.provenance),
            "reasons": self.reasons,
            "category": self.category,
        }


class EvidenceGrader:
    def grade(
        self,
        finding: Dict[str, Any],
        provenance: ProvenanceReport,
    ) -> EvidenceGrade:
        fields = [str(name) for name in (finding.get("data_fields") or [])]
        stamped = finding.get("provenance")
        resolved = str(stamped) if stamped else provenance.worst_for(fields) if fields else UNVERIFIED

        reasons: List[str] = []
        grade = GRADE_A

        if resolved == SYNTHETIC:
            grade = GRADE_C
            reasons.append(f"依赖{PROVENANCE_LABELS[SYNTHETIC]}")
        elif resolved in (FALLBACK, UNVERIFIED):
            grade = GRADE_B
            reasons.append(f"依赖{PROVENANCE_LABELS.get(resolved, resolved)}")

        if fields:
            reasons.append("数据字段：" + "、".join(fields))

        sample_size = finding.get("sample_size")
        required = finding.get("required_sample")
        if isinstance(sample_size, int) and isinstance(required, int) and sample_size < required:
            grade = GRADE_C
            reasons.append(f"样本量不足：n={sample_size} < {required}")

        caveats = list(finding.get("caveats") or [])
        if caveats and grade == GRADE_A:
            grade = GRADE_B
        reasons.extend(str(caveat) for caveat in caveats)

        return EvidenceGrade(
            finding_id=str(finding.get("finding_id", "")),
            statement=str(finding.get("statement", "")),
            grade=grade,
            provenance=resolved,
            reasons=reasons,
            category=str(finding.get("category", "other")),
        )

    def grade_all(
        self,
        findings: List[Dict[str, Any]],
        provenance: ProvenanceReport,
    ) -> List[EvidenceGrade]:
        return [self.grade(finding, provenance) for finding in findings]

    @staticmethod
    def summarise(grades: List[EvidenceGrade]) -> Dict[str, int]:
        summary = {GRADE_A: 0, GRADE_B: 0, GRADE_C: 0}
        for item in grades:
            summary[item.grade] = summary.get(item.grade, 0) + 1
        return summary


def grade_of(grades: List[EvidenceGrade], finding_id: str) -> Optional[EvidenceGrade]:
    for item in grades:
        if item.finding_id == finding_id:
            return item
    return None
