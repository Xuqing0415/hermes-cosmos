from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .report_interpreter import ReportInterpreter, SignalReport
from .insight_extractor import InsightExtractor, Insight
from .self_improvement_proposal import SelfImprovementProposal, Proposal
from .self_improvement_policy import SelfImprovementPolicy, EvolutionPolicy
from .report_generator import ReportGenerator
from .evolution_tracker import EvolutionTracker
from .knowledge_amalgamator import KnowledgeAmalgamator
from .notification_dispatcher import get_dispatcher, notify


@dataclass
class CognitiveLoopResult:
    snapshot_index: Optional[int]
    signal_report: SignalReport
    insights: List[Insight]
    proposals: List[Proposal]
    applied_proposals: List[Proposal]
    policy_before: Dict[str, Any]
    policy_after: Dict[str, Any]
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    loop_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_index": self.snapshot_index,
            "signals": self.signal_report.to_dict(),
            "insights": [i.to_dict() for i in self.insights],
            "proposals": [p.to_dict() for p in self.proposals],
            "applied_proposals": [p.to_dict() for p in self.applied_proposals],
            "policy_before": self.policy_before,
            "policy_after": self.policy_after,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "loop_summary": self.loop_summary,
        }


class CognitiveLoopEngine:
    def __init__(self, tracker: Optional[EvolutionTracker] = None,
                 amalgamator: Optional[KnowledgeAmalgamator] = None,
                 policy_loader: Optional[SelfImprovementPolicy] = None,
                 report_generator: Optional[ReportGenerator] = None):
        self._tracker = tracker
        self._amalgamator = amalgamator
        self._policy_loader = policy_loader
        self._report_generator = report_generator

    def run_loop(self, report_path: str = "evolution_report.md",
                 auto_apply: bool = True) -> CognitiveLoopResult:
        # Step 0: Load current policy
        current_policy = EvolutionPolicy()
        if self._policy_loader:
            current_policy = self._policy_loader.load_policy()
        policy_before = current_policy.to_dict()

        # Step 1: Interpret report -> signals
        interpreter = ReportInterpreter()
        signal_report = interpreter.interpret_path(report_path)

        # Step 2: Signals -> insights
        extractor = InsightExtractor(self._tracker)
        insights = extractor.extract(signal_report)

        # Step 3: Insights -> proposals
        proposer = SelfImprovementProposal(current_policy)
        proposals = proposer.generate_from_insights(insights)

        save_proposals = list(proposals)

        # Step 4: Apply proposals
        applied = []
        if auto_apply and proposals:
            for prop in proposals[:2]:
                if prop.action_type == "increase_weight":
                    current_policy.pattern_weights[prop.target] = prop.proposed_value
                    from .self_improvement_policy import EvolutionPolicy as EP
                    if hasattr(current_policy, '_normalize_weights'):
                        current_policy._normalize_weights(current_policy, EP())
                    # Direct normalization
                    total = sum(current_policy.pattern_weights.values())
                    if total > 1.0:
                        for k in current_policy.pattern_weights:
                            current_policy.pattern_weights[k] /= total

                    current_policy.update_count += 1
                    current_policy.last_updated = datetime.now(timezone.utc)
                    applied.append(prop)

                elif prop.action_type == "reset_similarity":
                    applied.append(prop)

            if self._policy_loader:
                self._policy_loader.save_policy(current_policy)

        if self._amalgamator and applied:
            self._amalgamator.save()

        policy_after = current_policy.to_dict()

        # Step 5: Notify
        snapshot_index = signal_report.report_snapshot_index
        summary_parts = []

        if signal_report.down_signals > 0:
            summary_parts.append(f"{signal_report.down_signals}个下降信号")
        if signal_report.up_signals > 0:
            summary_parts.append(f"{signal_report.up_signals}个上升信号")
        if signal_report.anomalies > 0:
            summary_parts.append(f"{signal_report.anomalies}个异常")

        if applied:
            summary_parts.append(f"已应用{len(applied)}个提案")
            for prop in applied:
                notify(
                    event_type="proposal_applied",
                    title=f"应用提案: {prop.title}",
                    message=f"{prop.description} (预期收益: {prop.expected_benefit:.0%})",
                    severity="info",
                    metadata={"proposal_id": prop.proposal_id, "target": prop.target},
                )

        summary = " | ".join(summary_parts) if summary_parts else "未检测到显著变化"

        return CognitiveLoopResult(
            snapshot_index=snapshot_index,
            signal_report=signal_report,
            insights=insights,
            proposals=proposals,
            applied_proposals=applied,
            policy_before=policy_before,
            policy_after=policy_after,
            success=len(applied) > 0 or signal_report.up_signals > 0,
            loop_summary=summary,
        )

    def reflect(self, report_path: str = "evolution_report.md") -> CognitiveLoopResult:
        return self.run_loop(report_path, auto_apply=True)