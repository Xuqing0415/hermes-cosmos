from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
import math

from .evolution_tracker import EvolutionTracker, EvolutionSnapshot


@dataclass
class PredictionModel:
    accuracy: float
    feature_importance: Dict[str, float]
    training_samples: int
    last_trained: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 2),
            "feature_importance": {
                k: round(v, 3) for k, v in
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            },
            "training_samples": self.training_samples,
            "last_trained": self.last_trained.isoformat(),
        }


class MentalModelTrainer:
    FEATURES = [
        "success_rate", "cross_domain_success", "pattern_coverage",
        "total_patterns", "total_relationships",
        "policy_mutation_rate", "policy_exploration_factor",
    ]

    def __init__(self, tracker: Optional[EvolutionTracker] = None):
        self._tracker = tracker
        self._model: Optional[PredictionModel] = None

    def train(self) -> PredictionModel:
        snapshots = []
        if self._tracker:
            snapshots = self._tracker.get_recent_snapshots(100)

        if len(snapshots) < 5:
            self._model = PredictionModel(
                accuracy=0.75,
                feature_importance={f: 1.0 / len(self.FEATURES) for f in self.FEATURES},
                training_samples=len(snapshots),
            )
            return self._model

        X, y = self._prepare_training_data(snapshots)
        accuracy = self._evaluate(X, y)

        self._model = PredictionModel(
            accuracy=accuracy,
            feature_importance=self._compute_importance(X, y),
            training_samples=len(snapshots),
        )
        return self._model

    def get_model(self) -> Optional[PredictionModel]:
        return self._model

    def predict_success_rate(self, current_state: Dict[str, float],
                             strategy_type: str, target: str,
                             adjustment: float) -> float:
        """Predict the success rate change given a strategy."""
        if self._model is None:
            return 0.05

        base = current_state.get("success_rate", 0.7)

        # Simulate feature changes based on strategy type
        predicted_delta = self._estimate_delta(
            base, current_state, strategy_type, target, adjustment
        )

        predicted = min(1.0, max(0.0, base + predicted_delta))
        return predicted

    def _estimate_delta(self, base_sr: float,
                        state: Dict[str, float],
                        strategy_type: str, target: str,
                        adjustment: float) -> float:
        """Estimate the delta based on strategy type and current state."""
        sr = state.get("success_rate", base_sr)
        cd = state.get("cross_domain_success", 0.5)
        pc = state.get("pattern_coverage", 0.5)

        delta = 0.0

        if strategy_type == "reset_similarity":
            base_benefit = 0.10
            decay = max(0, sr - 0.6) * 0.15
            delta = max(0.01, base_benefit - decay)
            if cd < 0.4:
                delta += 0.02

        elif strategy_type == "increase_weight":
            base_benefit = 0.05
            weight_delta = adjustment
            efficiency = max(0.3, 1.0 - weight_delta * 1.5)
            delta = base_benefit * efficiency

        elif strategy_type == "cross_domain_explore":
            base_benefit = 0.06
            if cd < 0.3:
                delta = base_benefit + 0.03
            elif cd > 0.7:
                delta = base_benefit - 0.02
            else:
                delta = base_benefit

        elif strategy_type == "pattern_refinement":
            base_benefit = 0.04
            if pc < 0.5:
                delta = base_benefit + 0.03
            else:
                delta = base_benefit * (1.0 - (pc - 0.5) * 0.5)

        elif strategy_type == "recalibrate":
            delta = 0.03 + (1.0 - sr) * 0.1

        else:
            delta = 0.02

        # Apply model accuracy as confidence scaling
        if self._model:
            delta *= (0.5 + self._model.accuracy * 0.5)

        return delta

    def _prepare_training_data(self, snapshots: List[EvolutionSnapshot]) -> Tuple[List, List]:
        X = []
        y = []

        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]

            features = [
                prev.success_rate,
                prev.cross_domain_success,
                prev.pattern_coverage,
                float(prev.total_patterns) / 10.0,
                float(prev.total_relationships) / 10.0,
                prev.policy_mutation_rate,
                prev.policy_exploration_factor,
            ]
            X.append(features)

            sr_change = curr.success_rate - prev.success_rate
            y.append(sr_change)

        return X, y

    def _evaluate(self, X: List, y: List) -> float:
        if len(X) < 3:
            return 0.7

        n = len(X)
        test_size = max(1, n // 5)

        correct = 0
        for i in range(0, n, test_size):
            test_indices = list(range(i, min(i + test_size, n)))
            train_indices = [j for j in range(n) if j not in test_indices]

            if not train_indices or not test_indices:
                continue

            train_y = [y[j] for j in train_indices]
            avg_change = sum(train_y) / len(train_y)

            for j in test_indices:
                predicted = avg_change
                actual = y[j]
                if (predicted >= 0 and actual >= 0) or (predicted < 0 and actual < 0):
                    correct += 1

        loo_accuracy = correct / n if n > 0 else 0.7
        return max(0.5, min(0.95, loo_accuracy))

    def _compute_importance(self, X: List, y: List) -> Dict[str, float]:
        if not X or len(X) < 2:
            return {f: 1.0 / len(self.FEATURES) for f in self.FEATURES}

        n_features = len(self.FEATURES)
        n_samples = len(X)
        base_pred = sum(y) / n_samples

        importances = {}
        for fi, fname in enumerate(self.FEATURES):
            col_values = [row[fi] for row in X]
            if max(col_values) == min(col_values):
                importances[fname] = 0.05
                continue
            corr_numerator = sum((X[j][fi] - sum(col_values) / n_samples) * (y[j] - base_pred) for j in range(n_samples))
            x_var = sum((X[j][fi] - sum(col_values) / n_samples) ** 2 for j in range(n_samples))
            y_var = sum((y[j] - base_pred) ** 2 for j in range(n_samples))
            if x_var * y_var > 0:
                corr = corr_numerator / math.sqrt(x_var * y_var)
                importances[fname] = max(0.0, abs(corr))
            else:
                importances[fname] = 0.05

        total = sum(importances.values())
        if total > 0:
            for k in importances:
                importances[k] /= total
        else:
            for f in self.FEATURES:
                importances[f] = 1.0 / n_features

        return importances