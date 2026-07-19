from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .features_extractor import FeatureCandidate, FeatureCategory, FeatureSource


class ImpactCategory(Enum):
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SECURITY = "security"
    USABILITY = "usability"
    MAINTAINABILITY = "maintainability"
    BUSINESS = "business"
    COMMUNITY = "community"


@dataclass
class ImpactScore:
    overall: float = 0.0
    category_scores: Dict[ImpactCategory, float] = field(default_factory=dict)
    description: str = ""
    confidence: float = 0.0
    metrics_improvement: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "category_scores": {k.value: v for k, v in self.category_scores.items()},
            "description": self.description,
            "confidence": self.confidence,
            "metrics_improvement": self.metrics_improvement
        }


class ImpactEstimator:
    CATEGORY_MULTIPLIERS = {
        FeatureCategory.SECURITY: {ImpactCategory.SECURITY: 3.0, ImpactCategory.BUSINESS: 2.0},
        FeatureCategory.PERFORMANCE: {ImpactCategory.PERFORMANCE: 3.0, ImpactCategory.QUALITY: 2.0},
        FeatureCategory.RELIABILITY: {ImpactCategory.QUALITY: 3.0, ImpactCategory.BUSINESS: 2.0},
        FeatureCategory.MAINTAINABILITY: {ImpactCategory.MAINTAINABILITY: 3.0, ImpactCategory.QUALITY: 1.5},
        FeatureCategory.TECH_DEBT: {ImpactCategory.MAINTAINABILITY: 2.5, ImpactCategory.QUALITY: 1.5},
        FeatureCategory.FEATURE_ENHANCEMENT: {ImpactCategory.BUSINESS: 2.0, ImpactCategory.USABILITY: 1.5},
        FeatureCategory.OBSERVABILITY: {ImpactCategory.QUALITY: 2.0, ImpactCategory.MAINTAINABILITY: 1.5},
        FeatureCategory.DEVOPS: {ImpactCategory.BUSINESS: 2.0, ImpactCategory.MAINTAINABILITY: 1.5},
        FeatureCategory.COMPLIANCE: {ImpactCategory.BUSINESS: 3.0, ImpactCategory.SECURITY: 2.0},
        FeatureCategory.USABILITY: {ImpactCategory.USABILITY: 3.0, ImpactCategory.BUSINESS: 1.5},
    }
    
    SOURCE_BONUSES = {
        FeatureSource.SELF_AUDIT: {ImpactCategory.QUALITY: 0.1, ImpactCategory.MAINTAINABILITY: 0.1},
        FeatureSource.TREND_ANALYSIS: {ImpactCategory.COMMUNITY: 0.2, ImpactCategory.BUSINESS: 0.1},
        FeatureSource.PROOF_FAILURE: {ImpactCategory.QUALITY: 0.2, ImpactCategory.SECURITY: 0.1},
        FeatureSource.ARCHITECTURE_SMELL: {ImpactCategory.MAINTAINABILITY: 0.2, ImpactCategory.QUALITY: 0.1},
        FeatureSource.TASK_HISTORY: {ImpactCategory.QUALITY: 0.15, ImpactCategory.BUSINESS: 0.1},
    }
    
    METRICS_TEMPLATES = {
        FeatureCategory.SECURITY: ["漏洞数量减少", "安全评分提升"],
        FeatureCategory.PERFORMANCE: ["执行速度提升", "内存占用降低"],
        FeatureCategory.RELIABILITY: ["任务成功率提升", "故障次数减少"],
        FeatureCategory.MAINTAINABILITY: ["代码复杂度降低", "维护成本减少"],
        FeatureCategory.TECH_DEBT: ["技术债务减少", "代码质量提升"],
        FeatureCategory.FEATURE_ENHANCEMENT: ["功能覆盖率提升", "用户体验改善"],
        FeatureCategory.OBSERVABILITY: ["可观测性提升", "故障定位时间缩短"],
        FeatureCategory.DEVOPS: ["部署时间缩短", "CI/CD效率提升"],
    }
    
    def __init__(self):
        pass
    
    def estimate(self, candidates: List[FeatureCandidate]) -> Dict[str, ImpactScore]:
        print("[影响估算器] 开始估算各候选特性的预期收益...")
        
        results = {}
        
        for candidate in candidates:
            impact = self._estimate_single(candidate)
            results[candidate.title] = impact
        
        print(f"[影响估算器] 完成 {len(results)} 个特性的影响估算")
        
        return results
    
    def _estimate_single(self, candidate: FeatureCandidate) -> ImpactScore:
        scores = {cat: 0.0 for cat in ImpactCategory}
        
        base_multipliers = self.CATEGORY_MULTIPLIERS.get(candidate.category, {})
        for impact_cat, multiplier in base_multipliers.items():
            scores[impact_cat] += candidate.priority_score / 100 * multiplier * 20
        
        source_bonus = self.SOURCE_BONUSES.get(candidate.source, {})
        for impact_cat, bonus in source_bonus.items():
            if impact_cat in scores:
                scores[impact_cat] += bonus * 10
        
        if candidate.category == FeatureCategory.SECURITY:
            severity = candidate.metadata.get("severity", "medium")
            if severity == "critical":
                scores[ImpactCategory.SECURITY] += 30
            elif severity == "high":
                scores[ImpactCategory.SECURITY] += 20
        
        if candidate.category == FeatureCategory.PERFORMANCE:
            stars = candidate.metadata.get("stars", 0)
            if stars > 10000:
                scores[ImpactCategory.PERFORMANCE] += 15
        
        overall_score = min(sum(scores.values()) / len(scores), 100)
        confidence = self._calculate_confidence(candidate)
        
        metrics = self._generate_metrics(candidate, scores)
        description = self._generate_description(candidate, scores)
        
        return ImpactScore(
            overall=round(overall_score, 1),
            category_scores={k: round(v, 1) for k, v in scores.items()},
            description=description,
            confidence=round(confidence, 2),
            metrics_improvement=metrics
        )
    
    def _calculate_confidence(self, candidate: FeatureCandidate) -> float:
        confidence = 0.5
        
        if candidate.source == FeatureSource.SELF_AUDIT:
            confidence += 0.3
        
        if candidate.source == FeatureSource.PROOF_FAILURE:
            confidence += 0.25
        
        if candidate.source == FeatureSource.ARCHITECTURE_SMELL:
            confidence += 0.2
        
        if candidate.source == FeatureSource.TREND_ANALYSIS:
            stars = candidate.metadata.get("stars", 0)
            if stars > 10000:
                confidence += 0.2
        
        if candidate.suggested_actions:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_metrics(self, candidate: FeatureCandidate, scores: Dict[ImpactCategory, float]) -> Dict[str, str]:
        templates = self.METRICS_TEMPLATES.get(candidate.category, [])
        metrics = {}
        
        high_score_cats = [cat for cat, score in scores.items() if score > 15]
        
        for i, cat in enumerate(high_score_cats[:2]):
            if i < len(templates):
                percentage = int(scores[cat] / 100 * 50) + 10
                metrics[templates[i]] = f"+{percentage}%"
        
        return metrics
    
    def _generate_description(self, candidate: FeatureCandidate, scores: Dict[ImpactCategory, float]) -> str:
        high_score_cats = sorted(
            [(cat, score) for cat, score in scores.items() if score > 10],
            key=lambda x: -x[1]
        )[:3]
        
        parts = []
        for cat, score in high_score_cats:
            if cat == ImpactCategory.PERFORMANCE:
                parts.append(f"性能提升")
            elif cat == ImpactCategory.SECURITY:
                parts.append(f"安全性增强")
            elif cat == ImpactCategory.QUALITY:
                parts.append(f"质量改善")
            elif cat == ImpactCategory.MAINTAINABILITY:
                parts.append(f"可维护性提升")
            elif cat == ImpactCategory.BUSINESS:
                parts.append(f"业务价值")
            elif cat == ImpactCategory.COMMUNITY:
                parts.append(f"社区吸引力")
        
        if parts:
            return f"预期收益: {', '.join(parts)}"
        return "预期收益: 多方面改进"
    
    def get_top_impact_features(self, impacts: Dict[str, ImpactScore], n: int = 5) -> List[str]:
        return sorted(impacts.keys(), key=lambda k: -impacts[k].overall)[:n]