from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .event_pattern_learner import ExternalPattern, EventPatternLearner
from .repository_miner import RepoMiningReport
from .self_improvement_policy import EvolutionPolicy


@dataclass
class ExternalInsight:
    insight_id: str
    source_pattern: str
    mapped_type: str
    suggestion: str
    estimated_impact: float
    confidence: float
    affected_domain: Optional[str] = None
    suggested_priority_change: Optional[Tuple[str, int, int]] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "insight_id": self.insight_id,
            "source_pattern": self.source_pattern,
            "mapped_type": self.mapped_type,
            "suggestion": self.suggestion,
            "estimated_impact": round(self.estimated_impact, 3),
            "confidence": round(self.confidence, 2),
            "affected_domain": self.affected_domain,
            "rationale": self.rationale,
        }
        if self.suggested_priority_change:
            result["suggested_priority_change"] = {
                "pattern": self.suggested_priority_change[0],
                "from": self.suggested_priority_change[1],
                "to": self.suggested_priority_change[2],
            }
        return result


INSIGHT_TEMPLATES = [
    {
        "mapped_type": "type_mismatch",
        "suggestion": "优先探索 '类型检查' 模式在 MLIR 领域的应用",
        "rationale": "外部项目经验表明类型检查改进贯穿整个生命周期，"
                     "在 MLIR 这一强类型领域有巨大迁移潜力",
        "estimated_impact": 0.07,
    },
    {
        "mapped_type": "boundary_check_missing",
        "suggestion": "调整边界检查优化策略，加速成熟阶段的迁移速度",
        "rationale": "外部数据显示边界检查修复集中在项目早期。"
                     "当系统进入成熟期后，应加速将边界检查经验迁移到新领域",
        "estimated_impact": 0.05,
    },
    {
        "mapped_type": "performance_bottleneck",
        "suggestion": "将性能优化与 API 调整计划同步",
        "rationale": "外部模式表明性能优化往往伴随 API 调整。"
                     "建议在规划 API 变更时同步安排性能回归测试",
        "estimated_impact": 0.04,
    },
    {
        "mapped_type": "memory_leak",
        "suggestion": "在功能完善阶段提前引入内存监测机制",
        "rationale": "外部数据显示后期内存问题关注度显著上升。"
                     "建议在系统成熟前就建立内存基线监控",
        "estimated_impact": 0.06,
    },
    {
        "mapped_type": "input_validation_missing",
        "suggestion": "在 MLIR 和 Kubernetes 领域部署统一输入验证管道",
        "rationale": "外部项目显示输入验证改进可显著减少后续的安全事件。"
                     "多领域统一验证可降低维护成本",
        "estimated_impact": 0.05,
    },
]


class ExternalInsightMapper:
    def __init__(self):
        self._insights: List[ExternalInsight] = []

    def map_patterns(self, patterns: List[ExternalPattern],
                     current_policy: Optional[EvolutionPolicy] = None) -> List[ExternalInsight]:
        self._insights.clear()
        counter = 0
        mapped_types = set(p.mapped_type for p in patterns)

        for template in INSIGHT_TEMPLATES:
            if template["mapped_type"] not in mapped_types:
                continue

            counter += 1
            matched_patterns = [p for p in patterns if p.mapped_type == template["mapped_type"]]
            avg_confidence = sum(p.confidence for p in matched_patterns) / len(matched_patterns) if matched_patterns else 0.5

            priority_change = None
            if current_policy and template["mapped_type"] in current_policy.pattern_weights:
                current_weight = current_policy.pattern_weights[template["mapped_type"]]
                if current_weight < 0.2:
                    priority_change = (
                        template["mapped_type"],
                        int(current_weight * 10),
                        min(10, int(current_weight * 10) + 3),
                    )

            self._insights.append(ExternalInsight(
                insight_id=f"insight-{counter:03d}",
                source_pattern=template["mapped_type"],
                mapped_type=template["mapped_type"],
                suggestion=template["suggestion"],
                estimated_impact=template["estimated_impact"],
                confidence=min(0.85, avg_confidence * 0.8 + 0.2),
                affected_domain=matched_patterns[0].domain if matched_patterns else None,
                suggested_priority_change=priority_change,
                rationale=template["rationale"],
            ))

        self._insights.sort(key=lambda i: i.estimated_impact, reverse=True)
        return self._insights

    def get_insights(self) -> List[ExternalInsight]:
        return self._insights