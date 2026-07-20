from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import statistics

from .value_tracker import ValueTracker, TrackedChange
from .evolution_tracker import EvolutionTracker


@dataclass
class StrategyEvaluation:
    strategy_type: str
    target: str
    instances: int
    avg_benefit: float
    median_benefit: float
    success_rate: float
    total_positive: int
    total_negative: int
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "target": self.target,
            "instances": self.instances,
            "avg_benefit": round(self.avg_benefit, 4),
            "median_benefit": round(self.median_benefit, 4),
            "success_rate": round(self.success_rate, 2),
            "total_positive": self.total_positive,
            "total_negative": self.total_negative,
            "recommendation": self.recommendation,
        }


@dataclass
class StrategyReport:
    evaluations: List[StrategyEvaluation] = field(default_factory=list)
    best_strategy: Optional[str] = None
    worst_strategy: Optional[str] = None
    overall_effectiveness: float = 0.0
    total_strategies_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_strategies_analyzed": self.total_strategies_analyzed,
            "overall_effectiveness": round(self.overall_effectiveness, 2),
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }


class StrategyEvaluator:
    def __init__(self, tracker: Optional[EvolutionTracker] = None):
        self._tracker = tracker

    def evaluate(self, changes: List[TrackedChange]) -> StrategyReport:
        report = StrategyReport()

        # Group by strategy_type
        type_groups: Dict[str, List[TrackedChange]] = {}
        for c in changes:
            if c.change_type not in type_groups:
                type_groups[c.change_type] = []
            type_groups[c.change_type].append(c)

        for stype, st_changes in type_groups.items():
            if not st_changes:
                continue

            benefits = [c.actual_benefit for c in st_changes]
            positive = sum(1 for b in benefits if b > 0)
            negative = sum(1 for b in benefits if b <= 0)

            avg_b = statistics.mean(benefits) if benefits else 0.0
            med_b = statistics.median(benefits) if len(benefits) >= 2 else (benefits[0] if benefits else 0.0)
            success = positive / len(benefits) if benefits else 0.0

            if avg_b > 0.05:
                rec = f"有效策略: {stype} 平均收益 {avg_b:+.0%}"
            elif avg_b > 0.01:
                rec = f"轻微有效: {stype} 平均收益 {avg_b:+.0%}"
            elif avg_b < -0.01:
                rec = f"无效策略: {stype} 平均收益 {avg_b:+.0%}，建议停止"
            else:
                rec = f"中性策略: {stype} 影响不显著"

            report.evaluations.append(StrategyEvaluation(
                strategy_type=stype,
                target="global",
                instances=len(st_changes),
                avg_benefit=avg_b,
                median_benefit=med_b,
                success_rate=success,
                total_positive=positive,
                total_negative=negative,
                recommendation=rec,
            ))

        # Group by target (pattern type)
        target_groups: Dict[str, List[TrackedChange]] = {}
        for c in changes:
            if c.target not in target_groups:
                target_groups[c.target] = []
            target_groups[c.target].append(c)

        for target, t_changes in target_groups.items():
            if len(t_changes) < 2:
                continue

            benefits = [c.actual_benefit for c in t_changes]
            avg_b = statistics.mean(benefits) if benefits else 0.0

            report.evaluations.append(StrategyEvaluation(
                strategy_type="target_analysis",
                target=target,
                instances=len(t_changes),
                avg_benefit=avg_b,
                median_benefit=statistics.median(benefits) if len(benefits) >= 2 else avg_b,
                success_rate=sum(1 for b in benefits if b > 0) / len(benefits) if benefits else 0.0,
                total_positive=sum(1 for b in benefits if b > 0),
                total_negative=sum(1 for b in benefits if b <= 0),
                recommendation=f"{target} 累计 {len(t_changes)} 次策略调整，平均收益 {avg_b:+.0%}",
            ))

        report.evaluations.sort(key=lambda e: e.avg_benefit, reverse=True)

        if report.evaluations:
            best = report.evaluations[0]
            worst = report.evaluations[-1]
            report.best_strategy = f"{best.strategy_type} ({best.avg_benefit:+.0%})" if best.avg_benefit > 0 else None
            report.worst_strategy = f"{worst.strategy_type} ({worst.avg_benefit:+.0%})" if worst.avg_benefit < 0 else None

            all_benefits = [e.avg_benefit for e in report.evaluations]
            report.overall_effectiveness = statistics.mean(all_benefits) if all_benefits else 0.0

        report.total_strategies_analyzed = len(changes)
        return report