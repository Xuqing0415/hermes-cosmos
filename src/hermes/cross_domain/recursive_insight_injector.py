from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .gap_calibrator import CalibrationSuggestion, GapCalibrator
from .evolution_compare_engine import ComparisonReport
from .mental_model_trainer import MentalModelTrainer
from .evolution_tracker import EvolutionTracker
from .self_improvement_policy import SelfImprovementPolicy
from .notification_dispatcher import notify


@dataclass
class CalibrationResult:
    calibrations_applied: int
    expected_accuracy_gain: float
    updated_training_samples: int
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrations_applied": self.calibrations_applied,
            "expected_accuracy_gain": round(self.expected_accuracy_gain, 2),
            "updated_training_samples": self.updated_training_samples,
            "message": self.message,
        }


class RecursiveInsightInjector:
    def __init__(self, tracker: Optional[EvolutionTracker] = None,
                 trainer: Optional[MentalModelTrainer] = None,
                 policy_loader: Optional[SelfImprovementPolicy] = None):
        self._tracker = tracker
        self._trainer = trainer
        self._policy_loader = policy_loader
        self._calibrator = GapCalibrator(trainer)

    def inject(self, comparison_report: ComparisonReport) -> CalibrationResult:
        suggestions = self._calibrator.analyze(comparison_report)
        applied = 0
        total_gain = 0.0

        if self._trainer:
            model = self._trainer.get_model()
            if model:
                for suggestion in suggestions[:3]:
                    if suggestion.target in model.feature_importance and suggestion.adjusted_value > 0:
                        old_val = model.feature_importance.get(suggestion.target, 0.5)
                        new_val = suggestion.adjusted_value
                        model.feature_importance[suggestion.target] = new_val
                        applied += 1
                        total_gain += suggestion.expected_accuracy_gain
                        notify(
                            event_type="model_calibrated",
                            title=f"校准: {suggestion.target}",
                            message=f"{suggestion.description[:60]}..."
                                    f" (预计准确率 +{suggestion.expected_accuracy_gain:.0%})",
                            severity="info",
                            metadata={"target": suggestion.target,
                                      "old_value": old_val, "new_value": new_val},
                        )

                if applied > 0:
                    model.accuracy = min(0.95, model.accuracy + total_gain * 0.3)

        if self._policy_loader:
            policy = self._policy_loader.load_policy()
            for suggestion in suggestions:
                if suggestion.target == "success_rate_learning_rate":
                    policy.mutation_rate = min(0.3, policy.mutation_rate + 0.02)
                elif suggestion.target == "decay_factor":
                    policy.exploration_factor = min(0.5, policy.exploration_factor + 0.03)

            policy.update_count += 1
            policy.last_updated = datetime.now(timezone.utc)
            self._policy_loader.save_policy(policy)

        sample_count = 0
        if self._tracker:
            sample_count = self._tracker.count_snapshots()

        message = f"已校准心智模型。应用 {applied} 项校准，预计准确率提升 {total_gain:.0%}。"

        return CalibrationResult(
            calibrations_applied=applied,
            expected_accuracy_gain=total_gain,
            updated_training_samples=sample_count,
            message=message,
        )