from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .report_interpreter import Signal, SignalReport
from .evolution_tracker import EvolutionTracker


@dataclass
class Insight:
    insight_type: str
    title: str
    description: str
    confidence: float
    affected_pattern: Optional[str] = None
    suggested_action: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "title": self.title,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "affected_pattern": self.affected_pattern,
            "suggested_action": self.suggested_action,
            "data": self.data,
            "priority": self.priority,
        }


class InsightExtractor:
    def __init__(self, tracker: Optional[EvolutionTracker] = None):
        self._tracker = tracker

    def extract(self, signal_report: SignalReport) -> List[Insight]:
        insights: List[Insight] = []

        down_signals = [s for s in signal_report.signals if s.direction == "down" and s.magnitude > 0.05]
        for sig in down_signals:
            insight = self._analyze_down_signal(sig, signal_report)
            if insight:
                insights.append(insight)

        weak_signals = [s for s in signal_report.signals if s.signal_type == "weakness"]
        for sig in weak_signals:
            insight = self._analyze_weakness(sig)
            if insight:
                insights.append(insight)

        if signal_report.anomalies > 0:
            insights.append(Insight(
                insight_type="anomaly_warning",
                title=f"检测到 {signal_report.anomalies} 个异常信号",
                description=f"在最近分析中发现 {signal_report.anomalies} 个指标下降幅度超过10%，"
                            f"需要重点关注",
                confidence=0.8,
                suggested_action="Run root cause analysis and consider auto-correction",
                data={"anomaly_count": signal_report.anomalies},
                priority=3,
            ))

        if self._tracker:
            snapshots = self._tracker.get_recent_snapshots(5)
            if len(snapshots) >= 2:
                insight = self._analyze_overall_trend(snapshots)
                if insight:
                    insights.append(insight)

        if not insights and signal_report.signals:
            insights.append(Insight(
                insight_type="status_ok",
                title="系统运行正常",
                description="未检测到显著的下降趋势或异常信号",
                confidence=0.9,
                suggested_action="Continue current evolution strategy",
                priority=1,
            ))

        insights.sort(key=lambda x: x.priority, reverse=True)
        return insights

    def _analyze_down_signal(self, signal: Signal, report: SignalReport) -> Optional[Insight]:
        ptype = signal.pattern_type or "unknown"
        if signal.magnitude > 0.15:
            return Insight(
                insight_type="critical_decline",
                title=f"{ptype} 模式严重下降",
                description=f"{ptype} 成功率下降 {signal.magnitude:.0%}，"
                            f"相似度矩阵可能已过时",
                confidence=0.85,
                affected_pattern=ptype,
                suggested_action=f"Reset similarity matrix for {ptype} to baseline",
                data={"pattern": ptype, "decline": signal.magnitude, "metric": signal.metric},
                priority=5,
            )
        elif signal.magnitude > 0.08:
            return Insight(
                insight_type="moderate_decline",
                title=f"{ptype} 模式出现持续下降",
                description=f"{ptype} 成功率下降 {signal.magnitude:.0%}，建议重新评估",
                confidence=0.7,
                affected_pattern=ptype,
                suggested_action=f"Increase sampling weight for {ptype}",
                data={"pattern": ptype, "decline": signal.magnitude},
                priority=3,
            )
        return None

    def _analyze_weakness(self, signal: Signal) -> Optional[Insight]:
        ptype = signal.pattern_type or "unknown"
        return Insight(
            insight_type="weakness_identified",
            title=f"{ptype} 模式置信度偏低",
            description=f"检测到 {ptype} 模式为薄弱环节，置信度有待提升，"
                        f"且在其他领域可能未得到充分利用",
            confidence=0.65,
            affected_pattern=ptype,
            suggested_action=f"Inject cross-domain examples for {ptype}",
            data={"pattern": ptype, "confidence": 1.0 - signal.magnitude},
            priority=4,
        )

    def _analyze_overall_trend(self, snapshots: List) -> Optional[Insight]:
        sr_values = [s.success_rate for s in snapshots]
        if len(sr_values) >= 2:
            delta = sr_values[-1] - sr_values[0]
            if delta > 0.05:
                return Insight(
                    insight_type="positive_trend",
                    title="系统整体呈上升趋势",
                    description=f"最近 {len(snapshots)} 次快照总体成功率提升 {delta:.0%}",
                    confidence=0.8,
                    suggested_action="Maintain current strategy, monitor for plateau",
                    data={"delta": delta, "snapshots": len(snapshots)},
                    priority=2,
                )
            elif delta < -0.05:
                return Insight(
                    insight_type="negative_trend",
                    title="系统整体呈下降趋势",
                    description=f"最近 {len(snapshots)} 次快照总体成功率下降 {abs(delta):.0%}，"
                                f"建议全面审查知识图谱",
                    confidence=0.85,
                    suggested_action="Run full auto-correction cycle",
                    data={"delta": delta, "snapshots": len(snapshots)},
                    priority=5,
                )
        return None