from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .mental_model_trainer import MentalModelTrainer
from .mental_simulator import MentalSimulator
from .strategy_sandbox import StrategySandbox, SandboxReport
from .evolution_tracker import EvolutionTracker
from .knowledge_amalgamator import KnowledgeAmalgamator
from .self_improvement_policy import SelfImprovementPolicy
from .notification_dispatcher import notify


@dataclass
class CognitivePlan:
    plan_id: str
    selected_strategy: str
    target: str
    expected_gain: float
    confidence: float
    predicted_success_rate: float
    sandbox_report: SandboxReport
    executed: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "selected_strategy": self.selected_strategy,
            "target": self.target,
            "expected_gain": round(self.expected_gain, 3),
            "confidence": round(self.confidence, 2),
            "predicted_success_rate": round(self.predicted_success_rate, 3),
            "executed": self.executed,
            "timestamp": self.timestamp.isoformat(),
            "sandbox_report": self.sandbox_report.to_dict(),
        }


class CognitivePlanner:
    def __init__(self, tracker: Optional[EvolutionTracker] = None,
                 amalgamator: Optional[KnowledgeAmalgamator] = None,
                 policy_loader: Optional[SelfImprovementPolicy] = None):
        self._tracker = tracker
        self._amalgamator = amalgamator
        self._policy_loader = policy_loader
        self._trainer = MentalModelTrainer(tracker)
        self._simulator = MentalSimulator(self._trainer, tracker)
        self._sandbox = StrategySandbox(self._simulator)

    def plan(self, candidates: Optional[List[Dict]] = None) -> CognitivePlan:
        import hashlib
        import time

        # Step 1: Train/update prediction model
        model = self._trainer.train()

        # Step 2: Simulate all candidates
        sandbox_report = self._sandbox.evaluate_all(candidates)

        # Step 3: Select best
        best = sandbox_report.best_candidate
        if not best:
            plan_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            return CognitivePlan(
                plan_id=plan_id,
                selected_strategy="none",
                target="none",
                expected_gain=0.0,
                confidence=0.0,
                predicted_success_rate=0.0,
                sandbox_report=sandbox_report,
            )

        plan_id = hashlib.md5(
            f"{best.strategy_type}-{best.target}-{time.time()}".encode()
        ).hexdigest()[:8]

        plan = CognitivePlan(
            plan_id=plan_id,
            selected_strategy=best.strategy_type,
            target=best.target,
            expected_gain=best.predicted_delta,
            confidence=best.confidence,
            predicted_success_rate=best.predicted_success_rate,
            sandbox_report=sandbox_report,
        )

        # Step 4: Execute best strategy in real environment
        self._execute_plan(plan)

        return plan

    def _execute_plan(self, plan: CognitivePlan):
        if plan.expected_gain <= 0:
            return

        if plan.selected_strategy == "increase_weight" and self._policy_loader:
            policy = self._policy_loader.load_policy()
            ptype = plan.target
            if ptype in policy.pattern_weights:
                old_weight = policy.pattern_weights[ptype]
                new_weight = min(1.0, old_weight + 0.15)
                policy.pattern_weights[ptype] = new_weight

                total = sum(policy.pattern_weights.values())
                if total > 1.0:
                    for k in policy.pattern_weights:
                        policy.pattern_weights[k] /= total

                policy.update_count += 1
                policy.last_updated = datetime.now(timezone.utc)
                self._policy_loader.save_policy(policy)

                plan.executed = True
                notify(
                    event_type="plan_executed",
                    title=f"执行认知计划: {plan.selected_strategy}",
                    message=f"目标: {plan.target}, 权重 {old_weight:.2f} -> {new_weight:.2f}, "
                            f"预期增益: {plan.expected_gain:.0%}",
                    severity="info",
                    metadata=plan.to_dict(),
                )

        elif plan.selected_strategy == "reset_similarity" and self._amalgamator:
            graph = self._amalgamator.get_graph()
            if graph:
                for edge in graph.edges:
                    if plan.target in str(edge.source_id) or plan.target in str(edge.target_id):
                        edge.similarity = max(0.5, edge.similarity)
                self._amalgamator.save()
                plan.executed = True
                notify(
                    event_type="plan_executed",
                    title=f"执行认知计划: {plan.selected_strategy}",
                    message=f"目标: {plan.target}",
                    severity="info",
                    metadata=plan.to_dict(),
                )

        elif plan.selected_strategy in ("cross_domain_explore", "pattern_refinement", "recalibrate"):
            plan.executed = True
            notify(
                event_type="plan_executed",
                title=f"执行认知计划: {plan.selected_strategy}",
                message=f"目标: {plan.target}, 预期增益: {plan.expected_gain:.0%}",
                severity="info",
                metadata=plan.to_dict(),
            )