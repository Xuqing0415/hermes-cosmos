"""把审稿意见转成论文修订，并用同一套公式重算“诚实置信度”。

修订只会让结论变弱，不会变强：

* 撤回：该结论不再作为结论出现，同时计为一次关键降级（撤回一个弱结论
  不能变成“论文变干净了”的加分项）；
* 降级：证据等级下调到 C；
* 改写/限定：证据等级至多降到 B；
* 不指向具体结论的意见（例如缺少对照组、幸存者偏差）按类别作用到该类别
  的全部结论上——否则批评了策略收益却一条都不改，等于没改。

新置信度不是另写的公式，而是把修订后的证据等级交给 `ConfidenceScorer`
重算一次，因此和原论文里的数字可以直接比较。
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from hermes.self_research.confidence_scorer import ConfidenceScorer
from hermes.self_research.data_provenance import ProvenanceReport
from hermes.self_research.evidence_grader import GRADE_A, GRADE_B, GRADE_C, EvidenceGrade
from hermes.self_research.integrity_checker import IntegrityChecker
from hermes.self_research.rebuttal_generator import (
    REVISION_DECLARE_LIMIT,
    REVISION_DOWNGRADE,
    REVISION_MAINTAIN,
    REVISION_REWORD,
    REVISION_SCOPE,
    REVISION_WITHDRAW,
)
from hermes.self_research.weakness_ranker import (
    ACTION_DOWNGRADE,
    ACTION_LABELS,
    ACTION_MAINTAIN,
    ACTION_REWORD,
    ACTION_SCOPE,
    ACTION_WITHDRAW,
    WeaknessRanking,
)

GRADE_ORDER = (GRADE_A, GRADE_B, GRADE_C)

#: 给单条结论定性时，修订动作的轻重顺序（越靠后越重）
ACTION_SEVERITY = (ACTION_MAINTAIN, ACTION_REWORD, ACTION_SCOPE, ACTION_DOWNGRADE, ACTION_WITHDRAW)

ACTION_ANNOTATIONS = {
    ACTION_MAINTAIN: "维持（反驳成立）",
    ACTION_WITHDRAW: "已撤回",
    ACTION_DOWNGRADE: "降级",
    ACTION_REWORD: "限定",
    ACTION_SCOPE: "限定",
}

#: 作者回应 -> 论文必须做的动作（有回应时以回应为准，没有回应才用排序动作）
REVISION_ACTIONS = {
    REVISION_MAINTAIN: ACTION_MAINTAIN,
    REVISION_WITHDRAW: ACTION_WITHDRAW,
    REVISION_DOWNGRADE: ACTION_DOWNGRADE,
    REVISION_REWORD: ACTION_REWORD,
    REVISION_SCOPE: ACTION_SCOPE,
    REVISION_DECLARE_LIMIT: ACTION_REWORD,
}

#: 动作 -> 证据等级上限（A 表示不降级）
ACTION_CEILINGS = {
    ACTION_MAINTAIN: GRADE_A,
    ACTION_WITHDRAW: GRADE_C,
    ACTION_DOWNGRADE: GRADE_C,
    ACTION_REWORD: GRADE_B,
    ACTION_SCOPE: GRADE_B,
}

FINDING_LINE_RE = re.compile(r"^\s*-\s*(F-\d+)\s")
ANNOTATION_RE = re.compile(r"\s*〔审稿后[^〕]*〕")
AUDIT_HEADING_RE = re.compile(r"^#{1,3}\s*审稿意见与作者回应.*$")
HEADING_RE = re.compile(r"^#{1,3}\s+\S")
REFERENCES_HEADING_RE = re.compile(r"^#{1,3}\s*(参考文献|References)\s*$", re.IGNORECASE)


@dataclass
class Revision:
    finding_id: str
    action: str
    before: str
    after: str
    reason: str
    attack_ids: List[str] = field(default_factory=list)
    category: str = "other"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "action": self.action,
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "attack_ids": self.attack_ids,
            "category": self.category,
        }


@dataclass
class RevisionResult:
    revisions: List[Revision] = field(default_factory=list)
    withdrawn_findings: List[str] = field(default_factory=list)
    revised_grades: List[Dict[str, Any]] = field(default_factory=list)
    baseline_confidence: Optional[float] = None
    revised_confidence: float = 0.0
    critical_count: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def confidence_delta(self) -> float:
        if self.baseline_confidence is None:
            return 0.0
        return self.revised_confidence - self.baseline_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_count": len(self.revisions),
            "revisions": [item.to_dict() for item in self.revisions],
            "withdrawn_findings": self.withdrawn_findings,
            "revised_grades": self.revised_grades,
            "baseline_confidence": self.baseline_confidence,
            "revised_confidence": round(self.revised_confidence, 4),
            "confidence_delta": round(self.confidence_delta, 4),
            "critical_count": self.critical_count,
            "notes": self.notes,
        }


def _weaken(grade: str, ceiling: str) -> str:
    """把等级降到不高于 ceiling（A 最强，C 最弱）。"""

    if grade not in GRADE_ORDER:
        return ceiling
    return grade if GRADE_ORDER.index(grade) >= GRADE_ORDER.index(ceiling) else ceiling


class PaperRevisionEngine:
    def __init__(
        self,
        scorer: Optional[ConfidenceScorer] = None,
        checker: Optional[IntegrityChecker] = None,
    ):
        self._scorer = scorer or ConfidenceScorer()
        self._checker = checker or IntegrityChecker()

    def revise(self, context: Any, ranking: WeaknessRanking, rebuttals: Sequence[Any] = ()) -> RevisionResult:
        grades = {str(key): dict(value) for key, value in (context.grades or {}).items()}
        revisions: List[Revision] = []
        withdrawn: List[str] = []
        notes: List[str] = []
        by_attack = {str(item.attack_id): item for item in rebuttals}

        for weakness in ranking.weaknesses:
            rebuttal = by_attack.get(weakness.attack_id)
            action = self._action_for(weakness, rebuttal)
            targets = self._targets(weakness, grades)
            if not targets:
                revisions.append(
                    Revision(
                        finding_id=weakness.finding_id,
                        action=action,
                        before="",
                        after=weakness.suggested_fix,
                        reason=weakness.statement,
                        attack_ids=[weakness.attack_id],
                        category=weakness.category,
                    )
                )
                continue

            for finding_id in targets:
                grade = grades[finding_id]
                before = str(grade.get("grade") or GRADE_A)
                revised = self._revised_grade(action, before)
                grade["grade"] = revised
                reason = self._reason(weakness, action, rebuttal)
                grade["reasons"] = list(grade.get("reasons") or []) + [f"审稿后修订：{reason}"]
                grade["review_action"] = action
                if action == ACTION_WITHDRAW and finding_id not in withdrawn:
                    withdrawn.append(finding_id)

                after = self._after_text(action, before, revised)
                self._append_revision(
                    revisions,
                    finding_id=finding_id,
                    action=action,
                    before=before,
                    after=after,
                    reason=reason,
                    attack_id=weakness.attack_id,
                    category=str(grade.get("category") or "other"),
                )

        revised_grades = [grades[key] for key in sorted(grades)]
        provenance = ProvenanceReport.from_dict((context.integrity or {}).get("provenance") or {})
        evidence_grades = self._to_evidence_grades(revised_grades)

        integrity = self._checker.check(evidence_grades, provenance)
        # 撤回也是一次关键降级：撤回弱结论不能让论文分数变高
        critical_count = integrity.critical_count + len(withdrawn)
        breakdown = self._scorer.score(evidence_grades, provenance, critical_count=critical_count)

        baseline = context.confidence
        if baseline is None:
            notes.append("落盘产物里没有原置信度，本次只报告修订后的值")
        if not withdrawn and not revisions:
            notes.append("没有任何审稿意见需要修订")

        return RevisionResult(
            revisions=revisions,
            withdrawn_findings=sorted(withdrawn),
            revised_grades=revised_grades,
            baseline_confidence=baseline,
            revised_confidence=breakdown.score,
            critical_count=critical_count,
            notes=notes,
        )

    # ------------------------------------------------------------------ 内部

    @staticmethod
    def _action_for(weakness: Any, rebuttal: Any) -> str:
        """有回应时以回应为准，没有回应才用排序给出的动作。

        反驳成立（`maintain`）维持原等级，反驳无法核对（`downgrade`）压到 C——
        是否降级取决于反驳理由的质量，而不是“是否反驳”。
        """

        if rebuttal is not None:
            mapped = REVISION_ACTIONS.get(str(getattr(rebuttal, "revision", "")))
            if mapped:
                return mapped
        return weakness.required_action

    @staticmethod
    def _reason(weakness: Any, action: str, rebuttal: Any) -> str:
        """维持原等级时，论文里必须带上作者的反驳理由（否则等于白反驳）。"""

        if action == ACTION_MAINTAIN and rebuttal is not None:
            return f"{weakness.statement}；作者反驳：{rebuttal.response}"
        return weakness.statement

    @staticmethod
    def _targets(weakness: Any, grades: Dict[str, Dict[str, Any]]) -> List[str]:
        if weakness.finding_id:
            return [weakness.finding_id] if weakness.finding_id in grades else []
        # 全局意见：作用到同类别的全部结论（"other" 太宽泛，不做扩张）
        category = str(weakness.category or "")
        if category in ("", "other"):
            return []
        return sorted(key for key, grade in grades.items() if str(grade.get("category") or "") == category)

    @staticmethod
    def _revised_grade(action: str, before: str) -> str:
        return _weaken(before, ACTION_CEILINGS.get(action, GRADE_B))

    @staticmethod
    def _after_text(action: str, before: str, after: str) -> str:
        if action == ACTION_MAINTAIN:
            return f"维持 {before}（反驳理由成立，不降级）"
        if action == ACTION_WITHDRAW:
            return "已撤回（不再作为结论）"
        if before == after:
            return f"维持 {after}（已在该动作的下限）"
        if action == ACTION_DOWNGRADE:
            return f"证据等级 {before} → {after}"
        return f"证据等级 {before} → {after}（措辞/范围已限定）"

    @staticmethod
    def _append_revision(revisions, finding_id, action, before, after, reason, attack_id, category):
        for item in revisions:
            if item.finding_id == finding_id and item.action == action:
                if attack_id not in item.attack_ids:
                    item.attack_ids.append(attack_id)
                return
        revisions.append(
            Revision(
                finding_id=finding_id,
                action=action,
                before=before,
                after=after,
                reason=reason,
                attack_ids=[attack_id],
                category=category,
            )
        )

    @staticmethod
    def _to_evidence_grades(revised_grades: Sequence[Dict[str, Any]]) -> List[EvidenceGrade]:
        return [
            EvidenceGrade(
                finding_id=str(item.get("finding_id") or ""),
                statement=str(item.get("statement") or ""),
                grade=str(item.get("grade") or GRADE_A),
                provenance=str(item.get("provenance") or "REAL"),
                reasons=list(item.get("reasons") or []),
                category=str(item.get("category") or "other"),
            )
            for item in revised_grades
        ]

    # ------------------------------------------------------------------ 正文改写

    def revise_markdown(self, text: str, result: RevisionResult, audit_section: str = "") -> str:
        if audit_section:
            # 重复审计时先摘掉上一次的审计章节，否则会在论文里插第二次
            text = self._strip_audit_section(text)
        by_finding: Dict[str, List[Revision]] = {}
        for item in result.revisions:
            if item.finding_id:
                by_finding.setdefault(item.finding_id, []).append(item)
        revised_grades = {str(item.get("finding_id")): str(item.get("grade")) for item in result.revised_grades}
        lines: List[str] = []

        for line in text.splitlines():
            match = FINDING_LINE_RE.match(line)
            items = by_finding.get(match.group(1)) if match else None
            if not items:
                lines.append(line)
                continue
            # 一条结论可能被多条意见命中：取最重的动作给这句话定性，原因合并列出
            strongest = max(items, key=lambda item: ACTION_SEVERITY.index(item.action))
            reasons = "；".join(dict.fromkeys(item.reason for item in items if item.reason))
            originals = [item.before for item in items if item.before in GRADE_ORDER]
            before = min(originals, key=GRADE_ORDER.index) if originals else ""
            after = revised_grades.get(strongest.finding_id, before)
            # 对同一篇论文重复审计时，先去掉上一次的标注再写，避免叠加成一串
            text_line = ANNOTATION_RE.sub("", line.rstrip())
            # 行内的“（证据等级 X）”是审稿前的标注，修订后必须同步，否则正文与
            # 审计表自相矛盾：读者会以为论文仍在主张 A 级证据。
            if before and after and before != after:
                text_line = text_line.replace(f"（证据等级 {before}）", f"（证据等级 {before} → {after}）")
            lines.append(f"{text_line} 〔审稿后{ACTION_ANNOTATIONS[strongest.action]}：{reasons}〕")

        revised = "\n".join(lines)
        if not audit_section:
            return revised

        insert_at = self._references_index(revised)
        if insert_at is None:
            return revised.rstrip() + "\n\n" + audit_section.strip() + "\n"
        return revised[:insert_at] + audit_section.strip() + "\n\n" + revised[insert_at:]

    @staticmethod
    def _references_index(text: str) -> Optional[int]:
        offset = 0
        for line in text.splitlines(keepends=True):
            if REFERENCES_HEADING_RE.match(line.strip()):
                return offset
            offset += len(line)
        return None

    @staticmethod
    def _strip_audit_section(text: str) -> str:
        """删掉已有的“审稿意见与作者回应”章节，让重复审计保持幂等。"""

        lines = text.splitlines()
        start = next((index for index, line in enumerate(lines) if AUDIT_HEADING_RE.match(line.strip())), None)
        if start is None:
            return text

        end = len(lines)
        for index in range(start + 1, len(lines)):
            if HEADING_RE.match(lines[index].strip()):
                end = index
                break
        return "\n".join(lines[:start] + lines[end:])

    def audit_section(
        self,
        context: Any,
        audit: Any,
        ranking: WeaknessRanking,
        rebuttals: Sequence[Any],
        result: RevisionResult,
        heading: str = "## 审稿意见与作者回应（对抗性审计）",
    ) -> str:
        lines = [heading, ""]
        lines.append(
            "本节由对抗性审计模块自动生成：系统以审稿人的身份攻击自己的结论，"
            "再以作者的身份逐条回应。审计只依据已落盘的数据产物，不解析正文来猜数字。"
        )
        lines.append("")
        lines.append(
            f"- 审稿意见：{len(audit.attacks)} 条"
            f"（致命 {ranking.severity_counts.get('FATAL', 0)}、"
            f"重大 {ranking.severity_counts.get('MAJOR', 0)}、"
            f"次要 {ranking.severity_counts.get('MINOR', 0)}）"
        )
        lines.append(f"- 实际执行的检查：{('、'.join(audit.checks_run) or '无')}")
        if audit.skipped:
            lines.append(f"- 因缺少产物而未执行的检查：{('；'.join(audit.skipped))}")
        if getattr(audit, "self_limited", None):
            lines.append(f"- 已在正文中自我限定、不再重复攻击的结论：{'、'.join(audit.self_limited)}")
        if result.withdrawn_findings:
            lines.append(f"- 撤回结论：{'、'.join(result.withdrawn_findings)}")
        if result.baseline_confidence is not None:
            lines.append(
                f"- 置信度：{result.baseline_confidence:.2f} → {result.revised_confidence:.2f}"
                f"（修订后重算，扣分 {abs(result.confidence_delta):.2f}）"
            )
        lines.append("")
        lines.append("| 编号 | 类型 | 严重度 | 对象 | 审稿意见 | 作者回应 | 论文修订 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        by_attack = {item.attack_id: item for item in rebuttals}
        # “论文修订”一列要写实际落地的动作，而不是排序建议的动作：
        # 反驳成立时实际动作是“维持原等级”，写“限定适用范围”会误导读者
        applied: Dict[str, str] = {}
        for item in result.revisions:
            for attack_id in item.attack_ids:
                applied.setdefault(attack_id, item.action)
        for weakness in ranking.weaknesses:
            rebuttal = by_attack.get(weakness.attack_id)
            stance = rebuttal.stance_with_strength if rebuttal else "未回应"
            response = self._escape_cell(rebuttal.response if rebuttal else "")
            target = self._escape_cell(weakness.finding_id or weakness.target or "—")
            action_label = ACTION_LABELS.get(applied.get(weakness.attack_id, ""), weakness.action_label)
            lines.append(
                f"| {weakness.attack_id} | {weakness.attack_label} | {weakness.severity_label} | "
                f"{target} | {self._escape_cell(weakness.statement)} | "
                f"{stance}：{response} | {action_label} |"
            )
        lines.append("")
        if result.revisions:
            lines.append("具体修订：")
            for item in result.revisions:
                target = item.finding_id or "（全局）"
                lines.append(f"- {target}：{item.after}；原因：{item.reason}")
        else:
            lines.append("本次审计没有产生需要落地的修订。")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _escape_cell(text: str) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    @staticmethod
    def summarise(result: RevisionResult, ranking: WeaknessRanking) -> str:
        parts = [
            f"审稿意见 {len(ranking.weaknesses)} 条",
            f"修订 {len(result.revisions)} 处",
        ]
        if result.withdrawn_findings:
            parts.append(f"撤回 {len(result.withdrawn_findings)} 条结论")
        if result.baseline_confidence is not None:
            parts.append(f"置信度 {result.baseline_confidence:.2f} → {result.revised_confidence:.2f}")
        else:
            parts.append(f"修订后置信度 {result.revised_confidence:.2f}")
        return "，".join(parts)
