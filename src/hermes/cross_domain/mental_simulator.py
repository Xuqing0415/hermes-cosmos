from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .mental_model_trainer import MentalModelTrainer, PredictionModel
from .evolution_tracker import EvolutionTracker


@dataclass
class SimulationResult:
    candidate_id: str
    strategy_type: str
    target: str
    description: str
    adjustment: float
    predicted_success_rate: float
    predicted_cross_domain: float
    predicted_coverage: float
    predicted_delta: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_type": self.strategy_type,
            "target": self.target,
            "description": self.description,
            "predicted_success_rate": round(self.predicted_success_rate, 3),
            "predicted_cross_domain": round(self.predicted_cross_domain, 3),
            "predicted_coverage": round(self.predicted_coverage, 3),
            "predicted_delta": round(self.predicted_delta, 4),
            "confidence": round(self.confidence, 2),
            "details": self.details,
        }


CANDIDATE_STRATEGIES = [
    {
        "strategy_type": "reset_similarity",
        "target": "type_mismatch",
        "description": "Reset type_mismatch similarity matrix to baseline",
        "adjustment": 0.0,
    },
    {
        "strategy_type": "increase_weight",
        "target": "resource_limit_missing",
        "description": "Increase resource_limit_missing sampling weight",
        "adjustment": 0.15,
    },
    {
        "strategy_type": "cross_domain_explore",
        "target": "boundary_check_missing",
        "description": "Explore boundary_check_missing in MLIR domain",
        "adjustment": 0.0,
    },
    {
        "strategy_type": "pattern_refinement",
        "target": "memory_leak",
        "description": "Add cross-domain samples for memory_leak refinement",
        "adjustment": 0.0,
    },
    {
        "strategy_type": "recalibrate",
        "target": "general",
        "description": "Recalibrate all similarity matrices with latest data",
        "adjustment": 0.0,
    },
]


class MentalSimulator:
    def __init__(self, trainer: MentalModelTrainer,
                 tracker: Optional[EvolutionTracker] = None):
        self._trainer = trainer
        self._tracker = tracker

    def simulate(self, strategy_type: str, target: str,
                 adjustment: float = 0.0) -> SimulationResult:
        current_state = self._get_current_state()

        predicted_sr = self._trainer.predict_success_rate(
            current_state, strategy_type, target, adjustment
        )

        base_sr = current_state.get("success_rate", 0.7)
        delta = predicted_sr - base_sr

        model = self._trainer.get_model()
        confidence = model.accuracy if model else 0.7

        # Predict cross-domain and coverage effects
        cd_effect = delta * (0.6 + current_state.get("cross_domain_success", 0.5) * 0.3)
        pc_effect = delta * 0.3

        return SimulationResult(
            candidate_id=f"sim-{strategy_type}-{target}",
            strategy_type=strategy_type,
            target=target,
            description=self._describe(strategy_type, target),
            adjustment=adjustment,
            predicted_success_rate=predicted_sr,
            predicted_cross_domain=min(1.0, max(0.0, base_sr + cd_effect)),
            predicted_coverage=min(1.0, max(0.0, current_state.get("pattern_coverage", 0.5) + pc_effect)),
            predicted_delta=delta,
            confidence=confidence,
            details={
                "base_success_rate": base_sr,
                "strategy_type": strategy_type,
                "target": target,
            },
        )

    def simulate_candidates(self, candidates: Optional[List[Dict]] = None) -> List[SimulationResult]:
        if candidates is None:
            candidates = CANDIDATE_STRATEGIES

        results = []
        for cand in candidates:
            result = self.simulate(
                cand["strategy_type"],
                cand["target"],
                cand.get("adjustment", 0.0),
            )
            result.description = cand["description"]
            results.append(result)

        results.sort(key=lambda r: r.predicted_delta, reverse=True)
        return results

    def _get_current_state(self) -> Dict[str, float]:
        state = {
            "success_rate": 0.7,
            "cross_domain_success": 0.5,
            "pattern_coverage": 0.5,
            "total_patterns": 0,
            "total_relationships": 0,
            "policy_mutation_rate": 0.1,
            "policy_exploration_factor": 0.15,
        }

        if self._tracker:
            snapshots = self._tracker.get_recent_snapshots(1)
            if snapshots:
                latest = snapshots[0]
                state.update({
                    "success_rate": latest.success_rate,
                    "cross_domain_success": latest.cross_domain_success,
                    "pattern_coverage": latest.pattern_coverage,
                    "total_patterns": float(latest.total_patterns),
                    "total_relationships": float(latest.total_relationships),
                    "policy_mutation_rate": latest.policy_mutation_rate,
                    "policy_exploration_factor": latest.policy_exploration_factor,
                })

        return state

    def _describe(self, strategy_type: str, target: str) -> str:
        descs = {
            "reset_similarity": f"重置 {target} 相似度矩阵",
            "increase_weight": f"增加 {target} 采样权重",
            "cross_domain_explore": f"探索 {target} 跨领域迁移",
            "pattern_refinement": f"优化 {target} 模式检测",
            "recalibrate": "重新校准所有相似度矩阵",
        }
        return descs.get(strategy_type, f"{strategy_type}: {target}")