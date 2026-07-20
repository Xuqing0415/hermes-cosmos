from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .value_tracker import ValueTracker, TrackedChange
from .strategy_evaluator import StrategyEvaluator, StrategyReport
from .value_discovery import ValueDiscovery, ValuePrinciple
from .self_improvement_policy import SelfImprovementPolicy, EvolutionPolicy
from .knowledge_amalgamator import KnowledgeAmalgamator
from .evolution_tracker import EvolutionTracker
from .notification_dispatcher import notify


@dataclass
class EvolutionTarget:
    target_type: str
    target_name: str
    action: str
    expected_benefit: float
    confidence: float
    rationale: str
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": self.target_type,
            "target_name": self.target_name,
            "action": self.action,
            "expected_benefit": round(self.expected_benefit, 3),
            "confidence": round(self.confidence, 2),
            "rationale": self.rationale,
            "priority": self.priority,
        }


TARGET_AREAS = [
    {
        "type": "cross_domain_transfer",
        "name": "resource_limit_missing",
        "action": "Explore resource_limit_missing migration potential in MLIR domain",
        "expected_benefit": 0.08,
        "confidence": 0.65,
        "rationale": "resource_limit_missing 已出现在 K8s 领域，"
                     "但在 MLIR 领域的迁移潜力尚未挖掘",
        "priority": 1,
    },
    {
        "type": "pattern_refinement",
        "name": "type_mismatch",
        "action": "Refine type_mismatch detection rules and add cross-domain samples",
        "expected_benefit": 0.06,
        "confidence": 0.70,
        "rationale": "type_mismatch 当前在知识图谱中置信度偏低，"
                     "加入更多训练样本可提升其识别能力",
        "priority": 2,
    },
    {
        "type": "similarity_recalibration",
        "name": "boundary_check_missing",
        "action": "Recalibrate boundary_check_missing similarity with latest migration results",
        "expected_benefit": 0.05,
        "confidence": 0.75,
        "rationale": "boundary_check_missing 是最活跃的模式，"
                     "定期重校准可保持其高迁移成功率",
        "priority": 3,
    },
]


class SelfGuidedEvolver:
    def __init__(self, tracker: Optional[EvolutionTracker] = None,
                 amalgamator: Optional[KnowledgeAmalgamator] = None,
                 policy_loader: Optional[SelfImprovementPolicy] = None):
        self._tracker = tracker
        self._amalgamator = amalgamator
        self._policy_loader = policy_loader

    def run_value_driven_cycle(self) -> Dict[str, Any]:
        # Step 1: Track changes
        value_tracker = ValueTracker(self._tracker)
        changes = value_tracker.analyze_history()

        # Step 2: Evaluate strategies
        evaluator = StrategyEvaluator(self._tracker)
        strategy_report = evaluator.evaluate(changes)

        # Step 3: Discover principles
        discovery = ValueDiscovery()
        principles = discovery.discover(strategy_report)

        # Step 4: Select next targets
        targets = self._select_targets(principles, strategy_report)

        # Step 5: Apply to policy
        if self._policy_loader and targets:
            self._apply_targets_to_policy(targets)

        # Notify
        for target in targets[:2]:
            notify(
                event_type="evolution_target_selected",
                title=f"新进化目标: {target.target_name}",
                message=f"Action: {target.action} (预期收益: {target.expected_benefit:.0%})",
                severity="info",
                metadata={"target": target.target_name, "benefit": target.expected_benefit},
            )

        return {
            "changes_analyzed": len(changes),
            "strategy_report": strategy_report.to_dict(),
            "principles": [p.to_dict() for p in principles],
            "targets": [t.to_dict() for t in targets],
        }

    def _select_targets(self, principles: List[ValuePrinciple],
                        strategy_report: StrategyReport) -> List[EvolutionTarget]:
        targets = []
        used_patterns = set()

        for area in TARGET_AREAS:
            if area["name"] in used_patterns:
                continue
            used_patterns.add(area["name"])

            # Adjust expected benefit based on principles
            benefit = area["expected_benefit"]
            confidence = area["confidence"]

            for p in principles:
                if "迁移" in p.title and area["type"] == "cross_domain_transfer":
                    benefit += 0.02
                if "边界检查" in p.title and area["name"] == "boundary_check_missing":
                    benefit += 0.01
                if "効果递减" in p.title or "diminishing" in p.title.lower():
                    if area["type"] == "pattern_refinement":
                        benefit -= 0.01

            targets.append(EvolutionTarget(
                target_type=area["type"],
                target_name=area["name"],
                action=area["action"],
                expected_benefit=benefit,
                confidence=confidence,
                rationale=area["rationale"],
                priority=area["priority"],
            ))

        targets.sort(key=lambda t: (t.expected_benefit, t.confidence), reverse=True)
        return targets

    def _apply_targets_to_policy(self, targets: List[EvolutionTarget]):
        if not self._policy_loader:
            return

        policy = self._policy_loader.load_policy()

        for target in targets[:2]:
            ptype = target.target_name
            if ptype in policy.pattern_weights:
                policy.pattern_weights[ptype] = min(
                    1.0, policy.pattern_weights[ptype] + target.expected_benefit
                )

        total = sum(policy.pattern_weights.values())
        if total > 1.0:
            for k in policy.pattern_weights:
                policy.pattern_weights[k] /= total

        policy.update_count += 1
        policy.last_updated = datetime.now(timezone.utc)
        self._policy_loader.save_policy(policy)