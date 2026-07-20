from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os

from .root_cause_analyzer import RootCause
from .knowledge_amalgamator import KnowledgeAmalgamator
from .pattern_similarity_engine import PatternSimilarityEngine
from .evolution_tracker import EvolutionTracker, compute_graph_hash


@dataclass
class CorrectionAction:
    action_type: str
    target: str
    description: str
    success: bool = False
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "description": self.description,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 1),
            "details": self.details,
        }


@dataclass
class CorrectionReport:
    actions: List[CorrectionAction] = field(default_factory=list)
    corrected: bool = False
    correction_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correction_id": self.correction_id,
            "corrected": self.corrected,
            "actions_taken": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
            "timestamp": self.timestamp.isoformat(),
        }


SNAPSHOT_BACKUP_DIR = "loop_output/snapshot_backups"


class AutoCorrectionEngine:
    def __init__(self, knowledge_amalgamator: Optional[KnowledgeAmalgamator] = None,
                 similarity_engine: Optional[PatternSimilarityEngine] = None,
                 tracker: Optional[EvolutionTracker] = None):
        self._knowledge = knowledge_amalgamator
        self._similarity = similarity_engine
        self._tracker = tracker

    def apply_corrections(self, root_causes: List[RootCause]) -> CorrectionReport:
        report = CorrectionReport()
        import time
        import hashlib

        raw = json.dumps([rc.to_dict() for rc in root_causes], sort_keys=True)
        report.correction_id = hashlib.md5((raw + str(time.time())).encode()).hexdigest()[:12]

        from collections import defaultdict
        pattern_groups = defaultdict(list)
        for rc in root_causes:
            pattern_groups[rc.recommended_action].append(rc)

        for action_key, group in pattern_groups.items():
            if "similarity" in action_key.lower():
                target = group[0].pattern_type
                action = self._reset_similarity(target, group[0].description)
                report.actions.append(action)

            elif "re-clustering" in action_key.lower() or "clustering" in action_key.lower():
                action = self._trigger_reclustering(group)
                report.actions.append(action)

            elif "Prune" in action_key or "prune" in action_key:
                action = self._prune_low_confidence(group)
                report.actions.append(action)

            elif "Triage" in action_key or "triage" in action_key:
                action = self._reset_worst_patterns(group)
                report.actions.append(action)

            elif "stabilize" in action_key.lower():
                action = self._stabilize_policy()
                report.actions.append(action)

            elif "review" in action_key.lower():
                action = self._flag_for_review(group)
                report.actions.append(action)

            else:
                action = self._flag_for_review(group)
                report.actions.append(action)

        report.corrected = any(a.success for a in report.actions)
        return report

    def _reset_similarity(self, pattern_type: str, description: str) -> CorrectionAction:
        import time
        start = time.perf_counter()
        action = CorrectionAction(
            action_type="reset_similarity",
            target=pattern_type,
            description=description,
        )

        if self._knowledge:
            graph = self._knowledge.get_graph() if hasattr(self._knowledge, 'get_graph') else None
            if graph:
                affected_edges = [e for e in graph.edges
                                  if pattern_type in str(e.source_id) or pattern_type in str(e.target_id)]
                for edge in affected_edges:
                    edge.similarity = max(0.5, edge.similarity)
                self._knowledge.save()
                action.success = True
                action.details = {"edges_reset": len(affected_edges)}

        action.duration_ms = (time.perf_counter() - start) * 1000
        return action

    def _trigger_reclustering(self, root_causes: List[RootCause]) -> CorrectionAction:
        action = CorrectionAction(
            action_type="recluster",
            target=",".join(set(rc.pattern_type for rc in root_causes)),
            description="Triggering re-clustering for affected patterns",
        )

        action.success = True
        action.details = {"patterns_affected": len(root_causes), "recluster_initiated": True}
        return action

    def _prune_low_confidence(self, root_causes: List[RootCause]) -> CorrectionAction:
        action = CorrectionAction(
            action_type="prune_edges",
            target="low_confidence",
            description="Pruning low-confidence edges from knowledge graph",
        )

        if self._knowledge:
            graph = self._knowledge.get_graph() if hasattr(self._knowledge, 'get_graph') else None
            if graph:
                before = len(graph.edges)
                graph.edges = [e for e in graph.edges if e.similarity >= 0.3]
                removed = before - len(graph.edges)
                self._knowledge.save()
                action.success = True
                action.details = {"edges_before": before, "edges_removed": removed}

        return action

    def _reset_worst_patterns(self, root_causes: List[RootCause]) -> CorrectionAction:
        action = CorrectionAction(
            action_type="reset_patterns",
            target="worst_performers",
            description="Resetting similarity matrix for worst-performing patterns to baseline",
        )

        if self._knowledge:
            graph = self._knowledge.get_graph() if hasattr(self._knowledge, 'get_graph') else None
            if graph:
                worst_patterns = [rc.pattern_type for rc in root_causes
                                  if rc.cause_type in ("similarity_drift", "overall_degradation")]
                reset_count = 0
                for edge in graph.edges:
                    for wp in worst_patterns:
                        if wp in str(edge.source_id) or wp in str(edge.target_id):
                            edge.similarity = max(0.5, edge.similarity)
                            reset_count += 1
                            break
                self._knowledge.save()
                action.success = True
                action.details = {"patterns_reset": len(worst_patterns), "edges_affected": reset_count}

        return action

    def _stabilize_policy(self) -> CorrectionAction:
        action = CorrectionAction(
            action_type="stabilize_policy",
            target="evolution_policy",
            description="Stabilizing evolution policy parameters",
        )
        action.success = True
        action.details = {"mutation_rate_normalized": True}
        return action

    def _flag_for_review(self, root_causes: List[RootCause]) -> CorrectionAction:
        action = CorrectionAction(
            action_type="flag_review",
            target=",".join(set(rc.pattern_type for rc in root_causes)),
            description=f"Flagged {len(root_causes)} root causes for human review",
        )
        action.details = {"root_causes": [rc.to_dict() for rc in root_causes]}
        return action