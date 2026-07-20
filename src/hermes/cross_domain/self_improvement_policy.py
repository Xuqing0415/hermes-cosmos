from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os

from .gap_analyzer import GapReport


@dataclass
class EvolutionPolicy:
    pattern_weights: Dict[str, float] = field(default_factory=lambda: {
        "boundary_check_missing": 0.2,
        "resource_limit_missing": 0.2,
        "input_validation_missing": 0.15,
        "api_version_deprecated": 0.1,
        "performance_bottleneck": 0.1,
        "division_by_zero": 0.15,
        "uninitialized_variable": 0.1,
    })
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    exploration_factor: float = 0.15
    min_confidence_threshold: float = 0.3
    max_iterations_per_cycle: int = 100
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_weights": self.pattern_weights,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "exploration_factor": self.exploration_factor,
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_iterations_per_cycle": self.max_iterations_per_cycle,
            "last_updated": self.last_updated.isoformat(),
            "update_count": self.update_count,
        }


class SelfImprovementPolicy:
    def __init__(self, storage_path: str = "loop_output/self_improvement_policy.json"):
        self._storage_path = storage_path
        self._policy_history: List[Dict[str, Any]] = []

    def load_policy(self) -> EvolutionPolicy:
        if os.path.exists(self._storage_path):
            try:
                with open(self._storage_path, 'r') as f:
                    data = json.load(f)
                policy = EvolutionPolicy(
                    pattern_weights=data.get("pattern_weights", {}),
                    mutation_rate=data.get("mutation_rate", 0.1),
                    crossover_rate=data.get("crossover_rate", 0.3),
                    exploration_factor=data.get("exploration_factor", 0.15),
                    min_confidence_threshold=data.get("min_confidence_threshold", 0.3),
                    max_iterations_per_cycle=data.get("max_iterations_per_cycle", 100),
                    last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now(timezone.utc).isoformat())),
                    update_count=data.get("update_count", 0),
                )
                return policy
            except Exception:
                pass
        return EvolutionPolicy()

    def save_policy(self, policy: EvolutionPolicy):
        directory = os.path.dirname(self._storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(self._storage_path, 'w') as f:
            json.dump(policy.to_dict(), f, indent=2)

    def update_from_gap_report(self, gap_report: GapReport) -> EvolutionPolicy:
        policy = self.load_policy()

        for priority_item in gap_report.suggested_priority:
            ptype = priority_item["pattern_type"]
            new_weight = priority_item.get("new_weight", 0.3)

            if new_weight > 1.0:
                new_weight = 1.0

            policy.pattern_weights[ptype] = round(new_weight, 2)

        self._normalize_weights(policy)

        weakness_count = len(gap_report.suggested_priority)
        if weakness_count > 0:
            policy.exploration_factor = min(0.5, 0.15 + weakness_count * 0.05)
            policy.mutation_rate = min(0.3, 0.1 + weakness_count * 0.02)

        if gap_report.overall_success_rate > 90.0:
            policy.min_confidence_threshold = min(0.6, policy.min_confidence_threshold + 0.05)
        elif gap_report.overall_success_rate < 40.0:
            policy.min_confidence_threshold = max(0.1, policy.min_confidence_threshold - 0.05)

        policy.update_count += 1
        policy.last_updated = datetime.now(timezone.utc)

        self._policy_history.append({
            "timestamp": policy.last_updated.isoformat(),
            "update_count": policy.update_count,
            "overall_success_rate": gap_report.overall_success_rate,
            "changed_weights": {
                p["pattern_type"]: p["new_weight"]
                for p in gap_report.suggested_priority
            },
        })

        self.save_policy(policy)
        return policy

    def get_policy_summary(self, policy: EvolutionPolicy) -> str:
        high_priority = [
            (pt, w) for pt, w in sorted(policy.pattern_weights.items(), key=lambda x: x[1], reverse=True)
            if w > 0.2
        ]

        low_priority = [
            (pt, w) for pt, w in sorted(policy.pattern_weights.items(), key=lambda x: x[1])
            if w <= 0.2
        ]

        lines = ["Current Evolution Policy:"]
        lines.append(f"  Mutation rate: {policy.mutation_rate}")
        lines.append(f"  Crossover rate: {policy.crossover_rate}")
        lines.append(f"  Exploration factor: {policy.exploration_factor}")
        lines.append(f"  Confidence threshold: {policy.min_confidence_threshold}")
        lines.append(f"  Max iterations/cycle: {policy.max_iterations_per_cycle}")
        lines.append(f"  Update count: {policy.update_count}")

        if high_priority:
            lines.append("\n  Priority patterns (weight > 0.2):")
            for pt, w in high_priority:
                lines.append(f"    - {pt}: {w}")

        if low_priority:
            lines.append("\n  Standard patterns (weight <= 0.2):")
            for pt, w in low_priority:
                lines.append(f"    - {pt}: {w}")

        return "\n".join(lines)

    def _normalize_weights(self, policy: EvolutionPolicy):
        total = sum(policy.pattern_weights.values())
        if total > 1.0:
            for pt in policy.pattern_weights:
                policy.pattern_weights[pt] /= total

    def recommend_next_action(self, policy: EvolutionPolicy, gap_report: GapReport) -> str:
        if gap_report.overall_success_rate >= 90:
            return "成熟阶段：系统表现良好，进入微调优化模式"
        elif gap_report.overall_success_rate >= 60:
            weakest = gap_report.weakest_pattern
            return f"强化阶段：聚焦于薄弱模式 {weakest}，增加其采样和训练数据"
        else:
            return "基础建设阶段：需大幅提升跨领域知识质量，建议重新构建知识图谱基础"