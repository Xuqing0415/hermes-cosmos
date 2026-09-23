"""把心智模型的预测与真实历史快照做对比。

历史实现里有两条伪造路径，现已被移除：

1. 没有任何快照时，用固定演示数值（实际 0.45/0.65/0.72、预测 0.50/0.72/0.68）
   拼出一份“预测 vs 实际”对比报告，并让 `overall_accuracy > 0`；
2. 某个阶段没有对应快照时，把该阶段的实际成功率默认成 0.5。

前者是无中生有，后者是拿凭空的中位数冒充观测值。现在这两种情况都不产出对比，
而是把原因写进 `ComparisonReport.notes`，让调用方自己决定怎么处理。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evolution_tracker import EvolutionTracker
from .mental_model_trainer import MentalModelTrainer
from .self_repository_miner import EvolutionStage

#: 快照与阶段的对应方式：把快照序列按索引区间切给各个阶段。
#: 这是一种对齐假设，不是 git 意义上的阶段归属，会在报告里声明。
SNAPSHOT_ALIGNMENT = "snapshot_index_range"

#: 历史中没有记录的特征，用中性值填充；这是假设而不是观测值，会在报告里声明。
NEUTRAL_FEATURES = {"cross_domain_success": 0.5, "pattern_coverage": 0.5}

OVER_THRESHOLD = 0.05


@dataclass
class StageComparison:
    stage_id: int
    stage_name: str
    actual_success_rate: float
    predicted_success_rate: float
    delta: float
    accuracy: float
    over_under: str  # overestimated, underestimated, accurate
    evidence: str
    snapshot_count: int = 0
    predicted_source: str = "mental_model"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "actual_success_rate": round(self.actual_success_rate, 2),
            "predicted_success_rate": round(self.predicted_success_rate, 2),
            "delta": round(self.delta, 2),
            "accuracy": round(self.accuracy, 2),
            "over_under": self.over_under,
            "evidence": self.evidence,
            "snapshot_count": self.snapshot_count,
            "predicted_source": self.predicted_source,
        }


@dataclass
class ComparisonReport:
    comparisons: List[StageComparison] = field(default_factory=list)
    overall_accuracy: float = 0.0
    model_bias: str = ""
    alignment: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_accuracy": round(self.overall_accuracy, 2),
            "model_bias": self.model_bias,
            "alignment": self.alignment,
            "notes": self.notes,
        }


class EvolutionCompareEngine:
    def __init__(self, tracker: Optional[EvolutionTracker] = None, trainer: Optional[MentalModelTrainer] = None):
        self._tracker = tracker
        self._trainer = trainer

    def compare(self, stages: List[EvolutionStage]) -> ComparisonReport:
        report = ComparisonReport()

        if not stages:
            report.model_bias = "no_data"
            report.notes.append("没有可用的演化阶段（git 历史不可用）：本次不做任何阶段对比。")
            return report

        snapshots = self._tracker.get_recent_snapshots(50) if self._tracker else []
        if not snapshots:
            report.model_bias = "no_data"
            report.notes.append("没有真实快照可供对比：本次不做任何阶段对比，也不用任何演示数值填充结论。")
            return report

        model = self._trainer.get_model() if self._trainer else None
        if model is None:
            report.model_bias = "no_model"
            report.notes.append("心智模型没有可用模型，无法给出预测：本次不做阶段对比。")
            return report

        if len(snapshots) < len(stages):
            report.model_bias = "no_data"
            report.notes.append(
                f"快照数 {len(snapshots)} 少于阶段数 {len(stages)}，无法把快照对应到各个阶段："
                "本次不做任何阶段对比（不用默认成功率填充）。"
            )
            return report

        report.alignment = SNAPSHOT_ALIGNMENT
        report.notes.append(
            f"快照与阶段的对应方式：{len(snapshots)} 个快照按索引区间切分给 {len(stages)} 个阶段"
            f"（{SNAPSHOT_ALIGNMENT}），属于对齐假设而非 git 意义上的阶段归属。"
        )
        report.notes.append(
            "预测所需特征 cross_domain_success、pattern_coverage 在历史中没有记录，"
            f"以中性值 {NEUTRAL_FEATURES['cross_domain_success']} 填充，属于假设而非观测值。"
        )

        per_stage = max(1, len(snapshots) // len(stages))
        total_accuracy = 0.0

        for sid, stage in enumerate(stages):
            start = sid * per_stage
            end = len(snapshots) if sid == len(stages) - 1 else start + per_stage
            stage_snapshots = snapshots[start:end]
            actual = sum(s.success_rate for s in stage_snapshots) / len(stage_snapshots)
            predicted = float(
                self._trainer.predict_success_rate(
                    {"success_rate": actual, **NEUTRAL_FEATURES}, "recalibrate", "general", 0.0
                )
            )

            delta = predicted - actual
            accuracy = max(0.0, 1.0 - abs(delta))
            total_accuracy += accuracy

            if delta > OVER_THRESHOLD:
                over_under = "overestimated"
            elif delta < -OVER_THRESHOLD:
                over_under = "underestimated"
            else:
                over_under = "accurate"

            evidence = (
                f"阶段 {stage.stage_id}（{stage.name}）实测平均成功率 {actual:.0%}，"
                f"心智模型预测 {predicted:.0%}，偏差 {delta:+.0%}"
                f"（{len(stage_snapshots)} 个快照，对应方式 {SNAPSHOT_ALIGNMENT}）"
            )

            report.comparisons.append(
                StageComparison(
                    stage_id=stage.stage_id,
                    stage_name=stage.name,
                    actual_success_rate=actual,
                    predicted_success_rate=predicted,
                    delta=delta,
                    accuracy=accuracy,
                    over_under=over_under,
                    evidence=evidence,
                    snapshot_count=len(stage_snapshots),
                    predicted_source="mental_model",
                )
            )

        report.overall_accuracy = total_accuracy / len(report.comparisons) if report.comparisons else 0.0

        underestimates = [c for c in report.comparisons if c.over_under == "underestimated"]
        overestimates = [c for c in report.comparisons if c.over_under == "overestimated"]
        if underestimates and overestimates:
            report.model_bias = "mixed"
        elif underestimates:
            report.model_bias = "conservative"
        elif overestimates:
            report.model_bias = "optimistic"
        else:
            report.model_bias = "balanced"

        return report
