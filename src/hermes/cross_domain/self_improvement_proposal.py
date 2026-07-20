from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os

from .insight_extractor import Insight
from .self_improvement_policy import EvolutionPolicy


@dataclass
class Proposal:
    proposal_id: str
    title: str
    description: str
    action_type: str
    target: str
    current_value: float
    proposed_value: float
    expected_benefit: float
    confidence: float
    risk: str  # low, medium, high
    implementation_steps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "description": self.description,
            "action_type": self.action_type,
            "target": self.target,
            "current_value": round(self.current_value, 2),
            "proposed_value": round(self.proposed_value, 2),
            "expected_benefit": round(self.expected_benefit, 3),
            "confidence": round(self.confidence, 2),
            "risk": self.risk,
            "implementation_steps": self.implementation_steps,
            "created_at": self.created_at.isoformat(),
        }


PROPOSAL_TEMPLATES = {
    "reset_similarity": {
        "title": "重置 {pattern} 相似度矩阵",
        "description": "将 {pattern} 的相似度矩阵重置为基线值 {baseline}，"
                       "以消除可能积累的漂移误差",
        "steps": [
            "Identify all edges related to {pattern} in knowledge graph",
            "Reset edge similarity values to baseline {baseline}",
            "Recalculate {pattern} cluster centroids",
            "Validate with sample test cases",
        ],
    },
    "increase_weight": {
        "title": "增加 {pattern} 采样权重",
        "description": "将 {pattern} 的采样权重从 {current} 提升至 {proposed}，"
                       "以提高该模式的训练覆盖率",
        "steps": [
            "Update evolution policy pattern_weights[{pattern}]",
            "Normalize weights to ensure sum <= 1.0",
            "Increase cross-domain sampling ratio for {pattern}",
        ],
    },
    "inject_examples": {
        "title": "为 {pattern} 注入跨领域样本",
        "description": "从 {source_domain} 领域收集 {pattern} 的示例，"
                       "注入到 {target_domain} 领域的训练集中",
        "steps": [
            "Extract {pattern} examples from {source_domain}",
            "Translate examples to {target_domain} format",
            "Add to test case library",
            "Run validation benchmark",
        ],
    },
    "enable_monitoring": {
        "title": "启用 {pattern} 专项监控",
        "description": "为 {pattern} 模式启用专项监控，跟踪其成功率变化趋势",
        "steps": [
            "Create dedicated metric tracker for {pattern}",
            "Set alert threshold at {threshold:.0%} decline",
            "Integrate with notification dispatcher",
        ],
    },
}


class SelfImprovementProposal:
    def __init__(self, policy: Optional[EvolutionPolicy] = None,
                 storage_path: str = "loop_output/proposals.json"):
        self._policy = policy or EvolutionPolicy()
        self._storage_path = storage_path
        self._proposal_counter = 0

    def generate_from_insights(self, insights: List[Insight]) -> List[Proposal]:
        proposals: List[Proposal] = []

        for insight in insights:
            proposals.extend(self._insight_to_proposals(insight))

        proposals.sort(key=lambda p: p.expected_benefit, reverse=True)
        return proposals

    def _insight_to_proposals(self, insight: Insight) -> List[Proposal]:
        proposals = []
        ptype = insight.affected_pattern or "unknown"

        if "相似度" in insight.description or "matrix" in insight.suggested_action:
            self._proposal_counter += 1
            baseline = max(0.5, self._policy.pattern_weights.get(ptype, 0.5))
            proposals.append(Proposal(
                proposal_id=f"prop-{self._proposal_counter:04d}",
                title=PROPOSAL_TEMPLATES["reset_similarity"]["title"].format(
                    pattern=ptype, baseline=baseline
                ),
                description=PROPOSAL_TEMPLATES["reset_similarity"]["description"].format(
                    pattern=ptype, baseline=baseline
                ),
                action_type="reset_similarity",
                target=ptype,
                current_value=0.0,
                proposed_value=baseline,
                expected_benefit=insight.data.get("decline", 0.08) * 0.8,
                confidence=0.7,
                risk="low",
                implementation_steps=[
                    s.format(pattern=ptype, baseline=baseline)
                    for s in PROPOSAL_TEMPLATES["reset_similarity"]["steps"]
                ],
            ))

        if "weight" in insight.suggested_action.lower() or "sampling" in insight.suggested_action.lower():
            self._proposal_counter += 1
            current_w = self._policy.pattern_weights.get(ptype, 0.1)
            proposed_w = min(1.0, current_w + 0.15)
            proposals.append(Proposal(
                proposal_id=f"prop-{self._proposal_counter:04d}",
                title=PROPOSAL_TEMPLATES["increase_weight"]["title"].format(
                    pattern=ptype, current=current_w, proposed=proposed_w
                ),
                description=PROPOSAL_TEMPLATES["increase_weight"]["description"].format(
                    pattern=ptype, current=current_w, proposed=proposed_w
                ),
                action_type="increase_weight",
                target=ptype,
                current_value=current_w,
                proposed_value=proposed_w,
                expected_benefit=0.05,
                confidence=0.6,
                risk="low",
                implementation_steps=[
                    s.format(pattern=ptype)
                    for s in PROPOSAL_TEMPLATES["increase_weight"]["steps"]
                ],
            ))

        if "inject" in insight.suggested_action.lower() or "cross-domain" in insight.suggested_action.lower():
            self._proposal_counter += 1
            proposals.append(Proposal(
                proposal_id=f"prop-{self._proposal_counter:04d}",
                title=PROPOSAL_TEMPLATES["inject_examples"]["title"].format(
                    pattern=ptype, source_domain="mlir", target_domain="k8s"
                ),
                description=PROPOSAL_TEMPLATES["inject_examples"]["description"].format(
                    pattern=ptype, source_domain="MLIR", target_domain="K8s"
                ),
                action_type="inject_examples",
                target=ptype,
                current_value=0.0,
                proposed_value=1.0,
                expected_benefit=insight.confidence * 0.1,
                confidence=0.5,
                risk="medium",
                implementation_steps=[
                    s.format(pattern=ptype, source_domain="MLIR", target_domain="K8s")
                    for s in PROPOSAL_TEMPLATES["inject_examples"]["steps"]
                ],
            ))

        return proposals

    def save_proposals(self, proposals: List[Proposal]):
        directory = os.path.dirname(self._storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        existing = []
        if os.path.exists(self._storage_path):
            try:
                with open(self._storage_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                pass

        new_data = [p.to_dict() for p in proposals]
        existing.extend(new_data)

        with open(self._storage_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def load_proposals(self) -> List[Dict[str, Any]]:
        if os.path.exists(self._storage_path):
            try:
                with open(self._storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return []