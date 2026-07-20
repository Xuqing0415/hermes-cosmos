from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .evolution_tracker import EvolutionTracker, EvolutionSnapshot


@dataclass
class TrackedChange:
    change_type: str
    target: str
    description: str
    applied_at_snapshot: int
    before_value: float
    after_value: float
    actual_benefit: float
    effective: bool
    evaluation_window: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type,
            "target": self.target,
            "description": self.description,
            "applied_at_snapshot": self.applied_at_snapshot,
            "before_value": round(self.before_value, 2),
            "after_value": round(self.after_value, 2),
            "actual_benefit": round(self.actual_benefit, 4),
            "effective": self.effective,
            "evaluation_window": self.evaluation_window,
        }


CHANGE_PATTERNS = {
    "reset_similarity": {
        "keywords": ["reset", "similarity"],
        "metric": "success_rate",
        "expected_direction": "up",
    },
    "increase_weight": {
        "keywords": ["weight", "sampling"],
        "metric": "success_rate",
        "expected_direction": "up",
    },
    "inject_examples": {
        "keywords": ["inject", "examples", "cross-domain"],
        "metric": "cross_domain_success",
        "expected_direction": "up",
    },
    "correction": {
        "keywords": ["correction", "fix", "reset"],
        "metric": "success_rate",
        "expected_direction": "up",
    },
}


class ValueTracker:
    def __init__(self, tracker: Optional[EvolutionTracker] = None):
        self._tracker = tracker
        self._changes: List[TrackedChange] = []

    def analyze_history(self, window: int = 3) -> List[TrackedChange]:
        self._changes.clear()
        if not self._tracker:
            return []

        snapshots = self._tracker.get_recent_snapshots(100)
        if len(snapshots) < window + 2:
            return []

        # Detect changes from snapshots metadata
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]

            # Check for metadata changes
            prev_meta = prev.metadata or {}
            curr_meta = curr.metadata or {}

            # Detect corrections
            if curr_meta.get("corrected"):
                sr_before = prev.success_rate
                sr_after = min(sr_before + 0.08, 1.0) if i + 1 < len(snapshots) else curr.success_rate
                if i + 1 < len(snapshots):
                    sr_after = snapshots[i + 1].success_rate

                benefit = sr_after - sr_before
                self._changes.append(TrackedChange(
                    change_type="correction",
                    target="auto_correction",
                    description="Auto-correction triggered",
                    applied_at_snapshot=curr.snapshot_index,
                    before_value=sr_before,
                    after_value=sr_after,
                    actual_benefit=benefit,
                    effective=benefit > 0.02,
                    evaluation_window=window,
                ))

            # Detect policy changes from metadata
            prev_weights = prev_meta.get("pattern_success_rates", {})
            curr_weights = curr_meta.get("pattern_success_rates", {})

            for ptype in set(list(prev_weights.keys()) + list(curr_weights.keys())):
                pw = prev_weights.get(ptype, 0.5)
                cw = curr_weights.get(ptype, 0.5)
                diff = cw - pw

                if diff > 0.1:
                    self._changes.append(TrackedChange(
                        change_type="increase_weight",
                        target=ptype,
                        description=f"Sampling weight increased for {ptype}",
                        applied_at_snapshot=curr.snapshot_index,
                        before_value=pw,
                        after_value=pw + diff,
                        actual_benefit=diff * 0.3,
                        effective=diff > 0.15,
                        evaluation_window=window,
                    ))

            # Detect success rate change
            sr_diff = curr.success_rate - prev.success_rate
            if abs(sr_diff) > 0.03 and not self._changes:
                pass  # Natural variation, not a controlled change

        return self._changes

    def get_effective_changes(self, threshold: float = 0.02) -> List[TrackedChange]:
        return [c for c in self._changes if c.actual_benefit >= threshold]

    def get_ineffective_changes(self) -> List[TrackedChange]:
        return [c for c in self._changes if c.actual_benefit < 0.02]

    def get_change_by_type(self, change_type: str) -> List[TrackedChange]:
        return [c for c in self._changes if c.change_type == change_type]