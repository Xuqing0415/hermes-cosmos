"""把审稿意见按严重程度排序，并给出每条意见必须触发什么动作。

排序规则是固定的、公开的：致命 > 重大 > 次要；同级按“是否指向关键结论”
（演化阶段、跨域迁移）优先。本模块只做排序与定级，不计算置信度——
置信度的新值由 `PaperRevisionEngine` 用同一套 `ConfidenceScorer`
在修订后的证据等级上重算，避免出现第二个“自创公式”。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes.self_research.adversarial_reviewer import (
    ATTACK_LABELS,
    FATAL,
    MULTIPLE_COMPARISONS,
    NO_CONTROL_GROUP,
    OVERGENERALIZATION,
    SAMPLE_SIZE,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
    WEAK_EVIDENCE,
    Attack,
    AuditReport,
)

# 审稿意见 -> 论文必须做的动作
ACTION_WITHDRAW = "withdraw"  # 撤回结论
ACTION_DOWNGRADE = "downgrade"  # 降级为初步观察
ACTION_REWORD = "reword"  # 改写措辞（保留结论）
ACTION_SCOPE = "scope"  # 限定适用范围

ACTION_LABELS = {
    ACTION_WITHDRAW: "撤回结论",
    ACTION_DOWNGRADE: "降级为初步观察",
    ACTION_REWORD: "改写措辞",
    ACTION_SCOPE: "限定适用范围",
}

ACTION_BY_TYPE = {
    SAMPLE_SIZE: ACTION_DOWNGRADE,
    WEAK_EVIDENCE: ACTION_DOWNGRADE,
    MULTIPLE_COMPARISONS: ACTION_DOWNGRADE,
    NO_CONTROL_GROUP: ACTION_REWORD,
    OVERGENERALIZATION: ACTION_SCOPE,
}

#: 这些类别的结论是论文的核心主张，同级优先处理
CRITICAL_CATEGORIES = ("evolution", "transfer")


@dataclass
class RankedWeakness:
    rank: int
    attack_id: str
    attack_type: str
    severity: str
    required_action: str
    finding_id: str
    category: str
    target: str
    statement: str
    evidence: str
    suggested_fix: str

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABELS.get(self.severity, self.severity)

    @property
    def attack_label(self) -> str:
        return ATTACK_LABELS.get(self.attack_type, self.attack_type)

    @property
    def action_label(self) -> str:
        return ACTION_LABELS.get(self.required_action, self.required_action)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "attack_id": self.attack_id,
            "attack_type": self.attack_type,
            "attack_label": self.attack_label,
            "severity": self.severity,
            "severity_label": self.severity_label,
            "required_action": self.required_action,
            "required_action_label": self.action_label,
            "finding_id": self.finding_id,
            "category": self.category,
            "target": self.target,
            "statement": self.statement,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class WeaknessRanking:
    weaknesses: List[RankedWeakness] = field(default_factory=list)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    affected_findings: List[str] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return self.severity_counts.get(FATAL, 0)

    def actions_for(self, finding_id: str) -> List[str]:
        return [item.required_action for item in self.weaknesses if item.finding_id and item.finding_id == finding_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weakness_count": len(self.weaknesses),
            "severity_counts": self.severity_counts,
            "affected_findings": self.affected_findings,
            "weaknesses": [item.to_dict() for item in self.weaknesses],
        }


class WeaknessRanker:
    def __init__(self, critical_categories=CRITICAL_CATEGORIES):
        self._critical_categories = tuple(critical_categories)

    def rank(self, report: AuditReport) -> WeaknessRanking:
        ordered = sorted(report.attacks, key=self._sort_key)
        weaknesses = [
            RankedWeakness(
                rank=index,
                attack_id=attack.attack_id,
                attack_type=attack.attack_type,
                severity=attack.severity,
                required_action=self._action_for(attack),
                finding_id=attack.finding_id,
                category=attack.category,
                target=attack.target,
                statement=attack.statement,
                evidence=attack.evidence,
                suggested_fix=attack.suggested_fix,
            )
            for index, attack in enumerate(ordered, 1)
        ]

        counts = dict.fromkeys(SEVERITY_ORDER, 0)
        for item in weaknesses:
            counts[item.severity] = counts.get(item.severity, 0) + 1

        affected = sorted({item.finding_id for item in weaknesses if item.finding_id})
        return WeaknessRanking(weaknesses=weaknesses, severity_counts=counts, affected_findings=affected)

    def _sort_key(self, attack: Attack):
        severity_rank = (
            SEVERITY_ORDER.index(attack.severity) if attack.severity in SEVERITY_ORDER else len(SEVERITY_ORDER)
        )
        critical = 0 if attack.category in self._critical_categories else 1
        has_finding = 0 if attack.finding_id else 1
        return (severity_rank, critical, has_finding, attack.attack_type, attack.attack_id)

    @staticmethod
    def _action_for(attack: Attack) -> str:
        if attack.severity == FATAL:
            return ACTION_WITHDRAW
        return ACTION_BY_TYPE.get(attack.attack_type, ACTION_REWORD)

    @staticmethod
    def severity_of(ranking: WeaknessRanking, attack_id: str) -> Optional[str]:
        for item in ranking.weaknesses:
            if item.attack_id == attack_id:
                return item.severity
        return None
