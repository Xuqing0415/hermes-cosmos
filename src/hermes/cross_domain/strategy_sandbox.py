from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .mental_simulator import MentalSimulator, SimulationResult
from .mental_model_trainer import MentalModelTrainer


@dataclass
class SandboxReport:
    rankings: List[SimulationResult] = field(default_factory=list)
    best_candidate: Optional[SimulationResult] = None
    spread: float = 0.0
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rankings": [r.to_dict() for r in self.rankings],
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "spread": round(self.spread, 3),
            "recommended_action": self.recommended_action,
        }


class StrategySandbox:
    def __init__(self, simulator: MentalSimulator):
        self._simulator = simulator

    def evaluate_all(self, candidates: Optional[List[Dict]] = None) -> SandboxReport:
        results = self._simulator.simulate_candidates(candidates)

        report = SandboxReport()
        report.rankings = results

        if results:
            report.best_candidate = results[0]
            best_delta = results[0].predicted_delta
            worst_delta = results[-1].predicted_delta
            report.spread = best_delta - worst_delta

            best = results[0]
            if best.confidence > 0.7 and best.predicted_delta > 0.02:
                report.recommended_action = (
                    f"Recommended: {best.description} "
                    f"(expected +{best.predicted_delta:.0%}, "
                    f"confidence {best.confidence:.0%})"
                )
            elif best.predicted_delta > 0:
                report.recommended_action = (
                    f"Best available: {best.description} "
                    f"(modest gain +{best.predicted_delta:.0%})"
                )
            else:
                report.recommended_action = (
                    "No strategy expected to yield positive gain. "
                    "Recommend running more experiments to gather data."
                )

        return report

    def compare(self, strategy_a: str, strategy_b: str) -> Dict[str, Any]:
        """Compare two specific strategies by name/target."""
        results = self._simulator.simulate_candidates()

        a_results = [r for r in results if strategy_a in r.description or strategy_a in r.target]
        b_results = [r for r in results if strategy_b in r.description or strategy_b in r.target]

        return {
            "strategy_a": [r.to_dict() for r in a_results] if a_results else [],
            "strategy_b": [r.to_dict() for r in b_results] if b_results else [],
            "a_better": (a_results[0].predicted_delta > b_results[0].predicted_delta)
            if a_results and b_results else None,
        }