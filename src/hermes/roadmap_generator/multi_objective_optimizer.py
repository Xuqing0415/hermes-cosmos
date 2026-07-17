from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import random

from .features_extractor import FeatureCandidate
from .impact_estimator import ImpactScore
from .workload_estimator import WorkloadEstimate


class PriorityLevel(Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


@dataclass
class PrioritizedFeature:
    candidate: FeatureCandidate
    impact: ImpactScore
    workload: WorkloadEstimate
    priority: PriorityLevel
    overall_score: float = 0.0
    roi: float = 0.0
    risk_level: str = "low"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.candidate.title,
            "description": self.candidate.description,
            "category": self.candidate.category.value,
            "source": self.candidate.source.value,
            "priority": self.priority.value,
            "overall_score": round(self.overall_score, 2),
            "roi": round(self.roi, 2),
            "risk_level": self.risk_level,
            "impact": self.impact.to_dict(),
            "workload": self.workload.to_dict(),
            "suggested_actions": self.candidate.suggested_actions
        }


class MultiObjectiveOptimizer:
    WEIGHTS = {
        "impact": 0.5,
        "workload": 0.3,
        "risk": 0.2,
    }
    
    RISK_THRESHOLDS = {
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3,
    }
    
    def __init__(self, max_workload: float = 20):
        self.max_workload = max_workload
    
    def prioritize(
        self,
        candidates: List[FeatureCandidate],
        impacts: Dict[str, ImpactScore],
        workloads: Dict[str, WorkloadEstimate]
    ) -> List[PrioritizedFeature]:
        print("[多目标优化器] 开始优先级排序...")
        
        prioritized = []
        
        for candidate in candidates:
            impact = impacts.get(candidate.title)
            workload = workloads.get(candidate.title)
            
            if impact is None or workload is None:
                continue
            
            roi = self._calculate_roi(impact, workload)
            risk = self._calculate_risk(candidate, impact, workload)
            overall_score = self._calculate_overall_score(impact, workload, risk)
            priority = self._assign_priority(overall_score, risk)
            
            prioritized.append(PrioritizedFeature(
                candidate=candidate,
                impact=impact,
                workload=workload,
                priority=priority,
                overall_score=overall_score,
                roi=roi,
                risk_level=risk
            ))
        
        prioritized.sort(key=lambda x: (-x.overall_score, x.workload.estimate))
        
        print(f"[多目标优化器] 完成排序，共 {len(prioritized)} 个特性")
        
        return prioritized
    
    def _calculate_roi(self, impact: ImpactScore, workload: WorkloadEstimate) -> float:
        if workload.estimate <= 0:
            return 0.0
        
        base_roi = impact.overall / workload.estimate
        
        confidence_bonus = impact.confidence * 0.5 + workload.confidence * 0.5
        base_roi *= (1 + confidence_bonus)
        
        return base_roi
    
    def _calculate_risk(self, candidate: FeatureCandidate, impact: ImpactScore, workload: WorkloadEstimate) -> str:
        risk_score = 0.0
        
        if workload.confidence < 0.5:
            risk_score += 0.3
        
        if impact.confidence < 0.5:
            risk_score += 0.2
        
        if candidate.source == FeatureSource.TREND_ANALYSIS:
            risk_score += 0.2
        
        if len(candidate.suggested_actions) == 0:
            risk_score += 0.15
        
        if len(workload.dependencies) > 2:
            risk_score += 0.15
        
        if workload.max_estimate > workload.estimate * 2:
            risk_score += 0.1
        
        for threshold, level in sorted(self.RISK_THRESHOLDS.items(), key=lambda x: -x[1]):
            if risk_score >= level:
                return threshold
        
        return "low"
    
    def _calculate_overall_score(self, impact: ImpactScore, workload: WorkloadEstimate, risk: str) -> float:
        impact_score = impact.overall
        workload_score = max(0, 100 - workload.estimate * 5)
        risk_score = self._risk_to_score(risk)
        
        return (
            self.WEIGHTS["impact"] * impact_score +
            self.WEIGHTS["workload"] * workload_score +
            self.WEIGHTS["risk"] * risk_score
        )
    
    def _risk_to_score(self, risk: str) -> float:
        mapping = {
            "critical": 20,
            "high": 40,
            "medium": 60,
            "low": 80,
        }
        return mapping.get(risk, 60)
    
    def _assign_priority(self, score: float, risk: str) -> PriorityLevel:
        if score >= 80 or risk == "critical":
            return PriorityLevel.P0
        elif score >= 60:
            return PriorityLevel.P1
        elif score >= 40:
            return PriorityLevel.P2
        else:
            return PriorityLevel.P3
    
    def pareto_frontier(
        self,
        candidates: List[FeatureCandidate],
        impacts: Dict[str, ImpactScore],
        workloads: Dict[str, WorkloadEstimate]
    ) -> List[PrioritizedFeature]:
        prioritized = self.prioritize(candidates, impacts, workloads)
        
        frontier = []
        for pf in prioritized:
            dominated = False
            for existing in frontier:
                if (existing.impact.overall >= pf.impact.overall and
                    existing.workload.estimate <= pf.workload.estimate and
                    (existing.impact.overall > pf.impact.overall or 
                     existing.workload.estimate < pf.workload.estimate)):
                    dominated = True
                    break
            
            if not dominated:
                frontier.append(pf)
                frontier = [
                    f for f in frontier
                    if not (pf.impact.overall >= f.impact.overall and
                            pf.workload.estimate <= f.workload.estimate and
                            (pf.impact.overall > f.impact.overall or
                             pf.workload.estimate < f.workload.estimate))
                ]
        
        frontier.sort(key=lambda x: -x.overall_score)
        
        return frontier
    
    def select_budget_portfolio(
        self,
        candidates: List[FeatureCandidate],
        impacts: Dict[str, ImpactScore],
        workloads: Dict[str, WorkloadEstimate],
        budget: float = None
    ) -> Tuple[List[PrioritizedFeature], float]:
        prioritized = self.prioritize(candidates, impacts, workloads)
        
        if budget is None:
            budget = self.max_workload
        
        portfolio = []
        total_workload = 0.0
        
        for pf in prioritized:
            if total_workload + pf.workload.estimate <= budget:
                portfolio.append(pf)
                total_workload += pf.workload.estimate
        
        return portfolio, round(total_workload, 1)
    
    def get_features_by_priority(
        self,
        prioritized: List[PrioritizedFeature],
        priority: PriorityLevel
    ) -> List[PrioritizedFeature]:
        return [pf for pf in prioritized if pf.priority == priority]


from .features_extractor import FeatureSource