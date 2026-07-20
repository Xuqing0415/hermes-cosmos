from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .evolution_tracker import EvolutionSnapshot
from .knowledge_amalgamator import KnowledgeAmalgamator


@dataclass
class RootCause:
    pattern_type: str
    metric_before: float
    metric_after: float
    delta: float
    cause_type: str
    description: str
    confidence: float
    affected_domains: List[str] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "metric_before": round(self.metric_before, 2),
            "metric_after": round(self.metric_after, 2),
            "delta": round(self.delta, 2),
            "cause_type": self.cause_type,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "affected_domains": self.affected_domains,
            "recommended_action": self.recommended_action,
        }


def _compute_pattern_success_from_snapshot(snapshot: EvolutionSnapshot,
                                            pattern_key: str) -> float:
    meta = snapshot.metadata or {}
    pattern_rates = meta.get("pattern_success_rates", {})
    return pattern_rates.get(pattern_key, 0.5)


class RootCauseAnalyzer:
    def __init__(self, knowledge_amalgamator: Optional[KnowledgeAmalgamator] = None):
        self._knowledge = knowledge_amalgamator

    def analyze(self, before_snapshot: EvolutionSnapshot,
                after_snapshot: EvolutionSnapshot) -> List[RootCause]:
        causes = []

        causes.extend(self._analyze_graph_structure(before_snapshot, after_snapshot))
        causes.extend(self._analyze_success_rates(before_snapshot, after_snapshot))
        causes.extend(self._analyze_policy_shift(before_snapshot, after_snapshot))

        if self._knowledge:
            causes.extend(self._analyze_knowledge_changes(before_snapshot, after_snapshot))

        if not causes:
            causes.append(RootCause(
                pattern_type="unknown",
                metric_before=0.0,
                metric_after=0.0,
                delta=after_snapshot.success_rate - before_snapshot.success_rate,
                cause_type="unidentified",
                description="Unable to identify root cause. May be environmental factors.",
                confidence=0.3,
                recommended_action="Run additional experiments to gather more data.",
            ))

        causes.sort(key=lambda c: c.confidence, reverse=True)
        return causes

    def _analyze_graph_structure(self, before: EvolutionSnapshot,
                                 after: EvolutionSnapshot) -> List[RootCause]:
        causes = []

        pattern_delta = after.total_patterns - before.total_patterns
        rel_delta = after.total_relationships - before.total_relationships

        if pattern_delta > 2:
            causes.append(RootCause(
                pattern_type="graph_expansion",
                metric_before=float(before.total_patterns),
                metric_after=float(after.total_patterns),
                delta=float(pattern_delta),
                cause_type="knowledge_inflation",
                description=f"知识图谱快速膨胀: 新增{pattern_delta}个模式节点",
                confidence=0.6,
                affected_domains=[],
                recommended_action="Review new pattern quality. Consider re-clustering.",
            ))

        if rel_delta > 5:
            causes.append(RootCause(
                pattern_type="graph_complexity",
                metric_before=float(before.total_relationships),
                metric_after=float(after.total_relationships),
                delta=float(rel_delta),
                cause_type="edge_explosion",
                description=f"知识图谱关系激增: 新增{rel_delta}条边",
                confidence=0.55,
                affected_domains=[],
                recommended_action="Prune low-confidence edges. Tighten similarity threshold.",
            ))

        return causes

    def _analyze_success_rates(self, before: EvolutionSnapshot,
                               after: EvolutionSnapshot) -> List[RootCause]:
        causes = []
        sr_delta = after.success_rate - before.success_rate

        if sr_delta < -0.05:
            causes.append(RootCause(
                pattern_type="success_rate_decline",
                metric_before=before.success_rate,
                metric_after=after.success_rate,
                delta=sr_delta,
                cause_type="overall_degradation",
                description=f"总体成功率下降 {abs(sr_delta):.1%}",
                confidence=0.7,
                recommended_action="Triage worst-performing patterns and reset their similarities.",
            ))

        cd_delta = after.cross_domain_success - before.cross_domain_success
        if cd_delta < -0.05:
            causes.append(RootCause(
                pattern_type="cross_domain_decline",
                metric_before=before.cross_domain_success,
                metric_after=after.cross_domain_success,
                delta=cd_delta,
                cause_type="transfer_degradation",
                description=f"跨领域迁移成功率下降 {abs(cd_delta):.1%}",
                confidence=0.65,
                recommended_action="Review cross-domain edges. Re-calibrate similarity matrix.",
            ))

        return causes

    def _analyze_policy_shift(self, before: EvolutionSnapshot,
                              after: EvolutionSnapshot) -> List[RootCause]:
        causes = []

        mutation_shift = after.policy_mutation_rate - before.policy_mutation_rate
        if abs(mutation_shift) > 0.1:
            causes.append(RootCause(
                pattern_type="policy_drift",
                metric_before=before.policy_mutation_rate,
                metric_after=after.policy_mutation_rate,
                delta=mutation_shift,
                cause_type="policy_instability",
                description=f"进化策略变异率大幅变化: {before.policy_mutation_rate} → {after.policy_mutation_rate}",
                confidence=0.5,
                recommended_action="Stabilize evolution policy. Reduce mutation rate variance.",
            ))

        return causes

    def _analyze_knowledge_changes(self, before: EvolutionSnapshot,
                                   after: EvolutionSnapshot) -> List[RootCause]:
        causes = []
        graph = self._knowledge.get_graph() if hasattr(self._knowledge, 'get_graph') else None
        if graph is None:
            return causes

        meta_after = after.metadata or {}
        pattern_rates_after = meta_after.get("pattern_success_rates", {})
        meta_before = before.metadata or {}
        pattern_rates_before = meta_before.get("pattern_success_rates", {})

        all_patterns = set(pattern_rates_before.keys()) | set(pattern_rates_after.keys())

        for ptype in all_patterns:
            rate_before = pattern_rates_before.get(ptype, 0.5)
            rate_after = pattern_rates_after.get(ptype, 0.5)
            delta = rate_after - rate_before

            if delta < -0.2:
                domains = []
                for node in graph.nodes:
                    if node.pattern_type.value == ptype:
                        domains = node.domains
                        break

                causes.append(RootCause(
                    pattern_type=ptype,
                    metric_before=rate_before,
                    metric_after=rate_after,
                    delta=delta,
                    cause_type="similarity_drift",
                    description=f"Similarity matrix drift in '{ptype}' pattern ({rate_before:.2f} → {rate_after:.2f})",
                    confidence=0.8,
                    affected_domains=domains,
                    recommended_action=f"Resetting similarity matrix for '{ptype}' to baseline ({rate_before:.2f})",
                ))

        return causes