from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .features_extractor import FeatureCandidate, FeatureCategory, FeatureSource


class WorkloadUnit(Enum):
    PERSON_DAYS = "person_days"
    AGENT_ROUNDS = "agent_rounds"
    STORY_POINTS = "story_points"


@dataclass
class WorkloadEstimate:
    estimate: float = 0.0
    unit: WorkloadUnit = WorkloadUnit.PERSON_DAYS
    min_estimate: float = 0.0
    max_estimate: float = 0.0
    confidence: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "unit": self.unit.value,
            "min_estimate": self.min_estimate,
            "max_estimate": self.max_estimate,
            "confidence": self.confidence,
            "breakdown": self.breakdown,
            "dependencies": self.dependencies
        }


class WorkloadEstimator:
    BASE_ESTIMATES = {
        FeatureCategory.SECURITY: {"base": 3, "var": 1.5},
        FeatureCategory.PERFORMANCE: {"base": 5, "var": 2},
        FeatureCategory.RELIABILITY: {"base": 4, "var": 1.5},
        FeatureCategory.MAINTAINABILITY: {"base": 2, "var": 1},
        FeatureCategory.TECH_DEBT: {"base": 6, "var": 3},
        FeatureCategory.FEATURE_ENHANCEMENT: {"base": 4, "var": 2},
        FeatureCategory.OBSERVABILITY: {"base": 3, "var": 1.5},
        FeatureCategory.DEVOPS: {"base": 2, "var": 1},
        FeatureCategory.COMPLIANCE: {"base": 5, "var": 2},
        FeatureCategory.USABILITY: {"base": 3, "var": 1.5},
    }
    
    SOURCE_FACTORS = {
        FeatureSource.SELF_AUDIT: 1.0,
        FeatureSource.TREND_ANALYSIS: 1.5,
        FeatureSource.PROOF_FAILURE: 0.8,
        FeatureSource.ARCHITECTURE_SMELL: 1.2,
        FeatureSource.TASK_HISTORY: 1.0,
    }
    
    COMPLEXITY_INDICATORS = {
        "rust": 2.0,
        "c++": 2.0,
        "kubernetes": 1.5,
        "distributed": 1.8,
        "security": 1.5,
        "performance": 1.5,
        "machine learning": 1.8,
        "neural": 1.8,
        "ai": 1.5,
        "gpu": 1.5,
        "database": 1.2,
        "api": 0.8,
        "web": 0.8,
        "testing": 0.6,
        "monitoring": 0.7,
        "logging": 0.5,
    }
    
    def __init__(self):
        pass
    
    def estimate(self, candidates: List[FeatureCandidate]) -> Dict[str, WorkloadEstimate]:
        print("[工作量估算器] 开始预估各候选特性的实现工作量...")
        
        results = {}
        
        for candidate in candidates:
            workload = self._estimate_single(candidate)
            results[candidate.title] = workload
        
        print(f"[工作量估算器] 完成 {len(results)} 个特性的工作量估算")
        
        return results
    
    def _estimate_single(self, candidate: FeatureCandidate) -> WorkloadEstimate:
        base_config = self.BASE_ESTIMATES.get(candidate.category, {"base": 3, "var": 1.5})
        source_factor = self.SOURCE_FACTORS.get(candidate.source, 1.0)
        
        base_estimate = base_config["base"] * source_factor
        
        complexity_factor = self._calculate_complexity_factor(candidate)
        base_estimate *= complexity_factor
        
        priority_factor = 1.0
        if candidate.priority_score > 80:
            priority_factor = 1.2
        elif candidate.priority_score < 30:
            priority_factor = 0.8
        base_estimate *= priority_factor
        
        var_range = base_config["var"] * complexity_factor
        min_estimate = max(base_estimate - var_range, 0.5)
        max_estimate = base_estimate + var_range * 2
        
        confidence = self._calculate_confidence(candidate)
        
        breakdown = self._generate_breakdown(candidate, base_estimate)
        dependencies = self._identify_dependencies(candidate)
        
        return WorkloadEstimate(
            estimate=round(base_estimate, 1),
            unit=WorkloadUnit.PERSON_DAYS,
            min_estimate=round(min_estimate, 1),
            max_estimate=round(max_estimate, 1),
            confidence=round(confidence, 2),
            breakdown={k: round(v, 1) for k, v in breakdown.items()},
            dependencies=dependencies
        )
    
    def _calculate_complexity_factor(self, candidate: FeatureCandidate) -> float:
        factor = 1.0
        text = f"{candidate.title} {candidate.description}".lower()
        
        for keyword, multiplier in self.COMPLEXITY_INDICATORS.items():
            if keyword in text:
                factor *= multiplier
        
        for tag in candidate.metadata.get("tags", []):
            tag_lower = tag.lower()
            if tag_lower in self.COMPLEXITY_INDICATORS:
                factor *= self.COMPLEXITY_INDICATORS[tag_lower]
        
        if factor > 3.0:
            factor = 3.0
        
        return factor
    
    def _calculate_confidence(self, candidate: FeatureCandidate) -> float:
        confidence = 0.5
        
        if candidate.source == FeatureSource.SELF_AUDIT:
            confidence += 0.2
        
        if candidate.source == FeatureSource.PROOF_FAILURE:
            confidence += 0.25
        
        if candidate.suggested_actions and len(candidate.suggested_actions) >= 2:
            confidence += 0.15
        
        if candidate.metadata.get("file_path"):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_breakdown(self, candidate: FeatureCandidate, total: float) -> Dict[str, float]:
        breakdown = {}
        
        if candidate.category == FeatureCategory.SECURITY:
            breakdown["安全评估"] = total * 0.3
            breakdown["实现修复"] = total * 0.5
            breakdown["测试验证"] = total * 0.2
        elif candidate.category == FeatureCategory.PERFORMANCE:
            breakdown["性能分析"] = total * 0.3
            breakdown["优化实现"] = total * 0.4
            breakdown["基准测试"] = total * 0.3
        elif candidate.category == FeatureCategory.MAINTAINABILITY:
            breakdown["代码分析"] = total * 0.2
            breakdown["重构实现"] = total * 0.6
            breakdown["回归测试"] = total * 0.2
        elif candidate.category == FeatureCategory.FEATURE_ENHANCEMENT:
            breakdown["需求分析"] = total * 0.2
            breakdown["功能实现"] = total * 0.5
            breakdown["测试文档"] = total * 0.3
        elif candidate.category == FeatureCategory.TECH_DEBT:
            breakdown["债务评估"] = total * 0.2
            breakdown["重构规划"] = total * 0.2
            breakdown["重构实施"] = total * 0.4
            breakdown["验证测试"] = total * 0.2
        else:
            breakdown["分析设计"] = total * 0.2
            breakdown["实现开发"] = total * 0.5
            breakdown["测试验证"] = total * 0.3
        
        return breakdown
    
    def _identify_dependencies(self, candidate: FeatureCandidate) -> List[str]:
        dependencies = []
        text = f"{candidate.title} {candidate.description}".lower()
        
        if "kubernetes" in text or "k8s" in text:
            dependencies.append("基础设施部署")
        
        if "api" in text or "web" in text:
            dependencies.append("API 网关")
        
        if "database" in text or "redis" in text:
            dependencies.append("数据存储")
        
        if "security" in text or "auth" in text:
            dependencies.append("安全模块")
        
        if "monitoring" in text or "observability" in text:
            dependencies.append("监控系统")
        
        return dependencies
    
    def convert_units(self, estimate: WorkloadEstimate, target_unit: WorkloadUnit) -> WorkloadEstimate:
        conversion_rates = {
            (WorkloadUnit.PERSON_DAYS, WorkloadUnit.AGENT_ROUNDS): 5,
            (WorkloadUnit.PERSON_DAYS, WorkloadUnit.STORY_POINTS): 1,
            (WorkloadUnit.AGENT_ROUNDS, WorkloadUnit.PERSON_DAYS): 0.2,
            (WorkloadUnit.AGENT_ROUNDS, WorkloadUnit.STORY_POINTS): 0.2,
            (WorkloadUnit.STORY_POINTS, WorkloadUnit.PERSON_DAYS): 1,
            (WorkloadUnit.STORY_POINTS, WorkloadUnit.AGENT_ROUNDS): 5,
        }
        
        if estimate.unit == target_unit:
            return estimate
        
        rate = conversion_rates.get((estimate.unit, target_unit), 1.0)
        
        return WorkloadEstimate(
            estimate=round(estimate.estimate * rate, 1),
            unit=target_unit,
            min_estimate=round(estimate.min_estimate * rate, 1),
            max_estimate=round(estimate.max_estimate * rate, 1),
            confidence=estimate.confidence,
            breakdown={k: round(v * rate, 1) for k, v in estimate.breakdown.items()},
            dependencies=estimate.dependencies
        )