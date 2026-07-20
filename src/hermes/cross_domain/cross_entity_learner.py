from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .repository_miner import RepositoryMiner, RepoMiningReport
from .event_pattern_learner import EventPatternLearner, ExternalPattern
from .external_insight_mapper import ExternalInsightMapper, ExternalInsight
from .self_improvement_policy import SelfImprovementPolicy, EvolutionPolicy
from .evolution_tracker import EvolutionTracker
from .notification_dispatcher import notify


@dataclass
class CrossEntityLearningResult:
    repo_url: str
    report: RepoMiningReport
    patterns: List[ExternalPattern]
    insights: List[ExternalInsight]
    applied_insights: int
    predicted_boost: float
    updated_policy: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "report": self.report.to_dict(),
            "patterns": [p.to_dict() for p in self.patterns],
            "insights": [i.to_dict() for i in self.insights],
            "applied_insights": self.applied_insights,
            "predicted_boost": round(self.predicted_boost, 3),
            "updated_policy": self.updated_policy,
            "timestamp": self.timestamp.isoformat(),
        }


class CrossEntityLearner:
    def __init__(self, tracker: Optional[EvolutionTracker] = None,
                 policy_loader: Optional[SelfImprovementPolicy] = None):
        self._tracker = tracker
        self._policy_loader = policy_loader
        self._miner = RepositoryMiner()
        self._pattern_learner = EventPatternLearner()
        self._insight_mapper = ExternalInsightMapper()

    def learn_from(self, repo_url: str) -> CrossEntityLearningResult:
        report = self._miner.mine(repo_url)
        patterns = self._pattern_learner.learn(report)

        current_policy = None
        if self._policy_loader:
            current_policy = self._policy_loader.load_policy()

        insights = self._insight_mapper.map_patterns(patterns, current_policy)

        applied_count = 0
        predicted_boost = 0.0

        if self._policy_loader and current_policy:
            for insight in insights[:3]:
                ptype = insight.mapped_type
                if ptype in current_policy.pattern_weights:
                    old_w = current_policy.pattern_weights[ptype]
                    new_w = min(1.0, old_w + 0.1)
                    current_policy.pattern_weights[ptype] = new_w
                    applied_count += 1
                    predicted_boost += insight.estimated_impact

                    notify(
                        event_type="external_insight_applied",
                        title=f"应用外部洞察: {ptype}",
                        message=f"权重 {old_w:.2f} -> {new_w:.2f}, 基于 {repo_url} 的学习",
                        severity="info",
                        metadata={
                            "pattern": ptype, "old_weight": old_w,
                            "new_weight": new_w, "source": repo_url,
                            "impact": insight.estimated_impact,
                        },
                    )

            total = sum(current_policy.pattern_weights.values())
            if total > 1.0:
                for k in current_policy.pattern_weights:
                    current_policy.pattern_weights[k] /= total

            current_policy.update_count += 1
            current_policy.last_updated = datetime.now(timezone.utc)
            self._policy_loader.save_policy(current_policy)

        return CrossEntityLearningResult(
            repo_url=repo_url,
            report=report,
            patterns=patterns,
            insights=insights,
            applied_insights=applied_count,
            predicted_boost=predicted_boost,
            updated_policy=current_policy.to_dict() if current_policy else {},
        )