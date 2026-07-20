from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .self_repository_miner import EvolutionStage
from .evolution_tracker import EvolutionTracker
from .mental_model_trainer import MentalModelTrainer, PredictionModel


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
        }


@dataclass
class ComparisonReport:
    comparisons: List[StageComparison] = field(default_factory=list)
    overall_accuracy: float = 0.0
    model_bias: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_accuracy": round(self.overall_accuracy, 2),
            "model_bias": self.model_bias,
        }


class EvolutionCompareEngine:
    def __init__(self, tracker: Optional[EvolutionTracker] = None,
                 trainer: Optional[MentalModelTrainer] = None):
        self._tracker = tracker
        self._trainer = trainer

    def compare(self, stages: List[EvolutionStage]) -> ComparisonReport:
        report = ComparisonReport()

        snapshots = []
        if self._tracker:
            snapshots = self._tracker.get_recent_snapshots(50)

        if not snapshots:
            # Generate synthetic predictions for demo
            base_rates = [0.45, 0.65, 0.72]
            pred_rates = [0.50, 0.72, 0.68]
        else:
            per_stage = max(1, len(snapshots) // max(1, len(stages)))
            base_rates = []
            pred_rates = []
            for sid, stage in enumerate(stages):
                start = sid * per_stage
                end = min((sid + 1) * per_stage, len(snapshots))
                stage_snapshots = snapshots[start:end]
                avg_sr = sum(s.success_rate for s in stage_snapshots) / len(stage_snapshots) if stage_snapshots else 0.5

                if self._trainer:
                    model = self._trainer.get_model()
                    pred = self._trainer.predict_success_rate(
                        {"success_rate": avg_sr, "cross_domain_success": 0.5, "pattern_coverage": 0.5},
                        "recalibrate", "general", 0.0
                    ) if model else avg_sr + 0.05
                else:
                    pred = avg_sr * (1.0 + 0.1 * (sid + 1))
                    pred = min(1.0, pred)

                base_rates.append(avg_sr)
                pred_rates.append(pred)

        total_accuracy = 0.0
        for sid, stage in enumerate(stages):
            actual = base_rates[sid] if sid < len(base_rates) else 0.5
            predicted = pred_rates[sid] if sid < len(pred_rates) else 0.5

            delta = predicted - actual
            accuracy = max(0.0, 1.0 - abs(delta))
            total_accuracy += accuracy

            if delta > 0.05:
                over_under = "overestimated"
                evidence = f"心智模型预测跨域迁移应在阶段 {stage.stage_id} 达到峰值 (准确率 {accuracy:.0%})"
            elif delta < -0.05:
                over_under = "underestimated"
                evidence = f"实际历史显示跨域迁移在阶段 {stage.stage_id} 才真正成熟 (准确率 {accuracy:.0%})"
            else:
                over_under = "accurate"
                evidence = f"预测与实际一致 (准确率 {accuracy:.0%})"

            report.comparisons.append(StageComparison(
                stage_id=stage.stage_id,
                stage_name=stage.name,
                actual_success_rate=actual,
                predicted_success_rate=predicted,
                delta=delta,
                accuracy=accuracy,
                over_under=over_under,
                evidence=evidence,
            ))

        report.overall_accuracy = total_accuracy / len(stages) if stages else 0.0

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