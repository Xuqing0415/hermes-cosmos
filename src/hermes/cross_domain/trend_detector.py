from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

from .evolution_tracker import EvolutionSnapshot


@dataclass
class TrendAlert:
    metric: str
    message: str
    severity: str
    current_value: float
    previous_value: float
    consecutive_declines: int
    threshold: int
    snapshot_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "message": self.message,
            "severity": self.severity,
            "current_value": round(self.current_value, 2),
            "previous_value": round(self.previous_value, 2),
            "consecutive_declines": self.consecutive_declines,
            "threshold": self.threshold,
            "snapshot_indices": self.snapshot_indices,
        }


@dataclass
class TrendReport:
    alerts: List[TrendAlert] = field(default_factory=list)
    moving_averages: Dict[str, float] = field(default_factory=dict)
    variances: Dict[str, float] = field(default_factory=dict)
    degradation_detected: bool = False
    summary: str = "正常"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "degradation_detected": self.degradation_detected,
            "summary": self.summary,
            "alert_count": len(self.alerts),
            "alerts": [a.to_dict() for a in self.alerts],
            "moving_averages": {k: round(v, 3) for k, v in self.moving_averages.items()},
            "variances": {k: round(v, 3) for k, v in self.variances.items()},
        }


class TrendDetector:
    CONSECUTIVE_DECLINE_THRESHOLD = 3
    MOVING_AVERAGE_WINDOW = 5

    def __init__(self, consecutive_threshold: int = 3, window: int = 5):
        self._consecutive_threshold = consecutive_threshold
        self._window = window

    def analyze(self, snapshots: List[EvolutionSnapshot]) -> TrendReport:
        report = TrendReport()

        if len(snapshots) < 2:
            report.summary = "数据不足，至少需要2个快照"
            return report

        metrics = {
            "success_rate": [s.success_rate for s in snapshots],
            "cross_domain_success": [s.cross_domain_success for s in snapshots],
            "pattern_coverage": [s.pattern_coverage for s in snapshots],
        }

        for metric_name, values in metrics.items():
            report.moving_averages[metric_name] = self._compute_moving_average(values)
            report.variances[metric_name] = self._compute_variance(values)

            decline_count = 0
            decline_start = 0
            best_consecutive = 0
            best_start = 0

            for i in range(1, len(values)):
                if values[i] < values[i - 1]:
                    if decline_count == 0:
                        decline_start = i - 1
                    decline_count += 1
                    if decline_count > best_consecutive:
                        best_consecutive = decline_count
                        best_start = decline_start
                else:
                    decline_count = 0

            if best_consecutive >= self._consecutive_threshold:
                severity = "critical" if best_consecutive >= 5 else "warning"
                involved_indices = [snapshots[best_start + j].snapshot_index
                                    for j in range(best_consecutive + 1)
                                    if best_start + j < len(snapshots)]

                alert = TrendAlert(
                    metric=metric_name,
                    message=f"{metric_name} has decreased for {best_consecutive} consecutive snapshots",
                    severity=severity,
                    current_value=values[-1],
                    previous_value=values[-best_consecutive - 1] if best_consecutive < len(values) else values[0],
                    consecutive_declines=best_consecutive,
                    threshold=self._consecutive_threshold,
                    snapshot_indices=involved_indices,
                )
                report.alerts.append(alert)

        if report.alerts:
            report.degradation_detected = True
            critical = [a for a in report.alerts if a.severity == "critical"]
            if critical:
                report.summary = f"严重退化: {len(critical)}个关键指标持续下降"
            else:
                report.summary = f"轻度退化: {len(report.alerts)}个指标出现下降趋势"
        else:
            report.summary = "所有指标正常"

        return report

    def _compute_moving_average(self, values: List[float]) -> float:
        if not values:
            return 0.0
        window_values = values[-self._window:]
        return sum(window_values) / len(window_values)

    def _compute_variance(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)