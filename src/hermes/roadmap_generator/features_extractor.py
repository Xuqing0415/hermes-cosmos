from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import json

from hermes.llm_requirement.self_audit import SelfAuditor, AuditResult, AuditIssue, AuditIssueType, IssueSeverity
from hermes.llm_requirement.trend_analyzer import TrendAnalyzer, TrendAnalysisResult, TrendItem, TrendCategory
from hermes.self_refactor.smell_detector import SmellDetector, SmellInstance, SmellType, SmellSeverity
from hermes.agent_civilization.task_board import TaskBoard, TaskStatus
from hermes.neural_symbolic.proof_generator import ProofGenerator, ProofReport


class FeatureSource(Enum):
    SELF_AUDIT = "self_audit"
    TREND_ANALYSIS = "trend_analysis"
    PROOF_FAILURE = "proof_failure"
    ARCHITECTURE_SMELL = "architecture_smell"
    TASK_HISTORY = "task_history"


class FeatureCategory(Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    MAINTAINABILITY = "maintainability"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    TECH_DEBT = "tech_debt"
    COMPLIANCE = "compliance"
    OBSERVABILITY = "observability"
    DEVOPS = "devops"


@dataclass
class FeatureCandidate:
    title: str
    description: str
    source: FeatureSource
    category: FeatureCategory
    priority_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_issues: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source.value,
            "category": self.category.value,
            "priority_score": self.priority_score,
            "metadata": self.metadata,
            "related_issues": self.related_issues,
            "suggested_actions": self.suggested_actions
        }


class FeatureExtractor:
    def __init__(self):
        self.self_auditor = SelfAuditor()
        self.trend_analyzer = TrendAnalyzer()
        self.smell_detector = SmellDetector()
        self.task_board = TaskBoard()
    
    def extract_all(self, existing_data: Optional[Dict[str, Any]] = None) -> List[FeatureCandidate]:
        print("[特征提取器] 开始从各数据源提取候选特性...")
        
        candidates = []
        
        audit_result = self._extract_from_self_audit(existing_data)
        candidates.extend(audit_result)
        print(f"[特征提取器] 从自我审计提取: {len(audit_result)} 个候选特性")
        
        trend_result = self._extract_from_trends(existing_data)
        candidates.extend(trend_result)
        print(f"[特征提取器] 从趋势分析提取: {len(trend_result)} 个候选特性")
        
        smell_result = self._extract_from_smells(existing_data)
        candidates.extend(smell_result)
        print(f"[特征提取器] 从架构异味提取: {len(smell_result)} 个候选特性")
        
        task_result = self._extract_from_task_history(existing_data)
        candidates.extend(task_result)
        print(f"[特征提取器] 从任务历史提取: {len(task_result)} 个候选特性")
        
        proof_result = self._extract_from_proof_failures(existing_data)
        candidates.extend(proof_result)
        print(f"[特征提取器] 从证明失败提取: {len(proof_result)} 个候选特性")
        
        candidates = self._deduplicate(candidates)
        print(f"[特征提取器] 去重后: {len(candidates)} 个候选特性")
        
        return candidates
    
    def _extract_from_self_audit(self, existing_data: Optional[Dict[str, Any]]) -> List[FeatureCandidate]:
        candidates = []
        
        if existing_data and "audit_result" in existing_data:
            audit_result = existing_data["audit_result"]
        else:
            try:
                audit_result = self.self_auditor.audit()
            except Exception as e:
                print(f"[特征提取器] 自我审计失败: {e}")
                return candidates
        
        if isinstance(audit_result, AuditResult):
            issues = audit_result.issues
        elif isinstance(audit_result, dict):
            issues = [
                AuditIssue(
                    issue_type=AuditIssueType(i["issue_type"]),
                    severity=IssueSeverity(i["severity"]),
                    title=i["title"],
                    description=i["description"],
                    file_path=i["file_path"],
                    line_number=i["line_number"],
                    suggestions=i.get("suggestions", []),
                    metadata=i.get("metadata", {})
                ) for i in audit_result.get("issues", [])
            ]
        else:
            issues = []
        
        for issue in issues:
            category = self._map_audit_issue_to_category(issue)
            priority = self._calculate_audit_priority(issue)
            
            candidate = FeatureCandidate(
                title=issue.title,
                description=issue.description,
                source=FeatureSource.SELF_AUDIT,
                category=category,
                priority_score=priority,
                metadata={
                    "severity": issue.severity.value,
                    "issue_type": issue.issue_type.value,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number
                },
                related_issues=[issue.title],
                suggested_actions=issue.suggestions
            )
            candidates.append(candidate)
        
        return candidates
    
    def _extract_from_trends(self, existing_data: Optional[Dict[str, Any]]) -> List[FeatureCandidate]:
        candidates = []
        
        if existing_data and "trend_result" in existing_data:
            trend_result = existing_data["trend_result"]
        else:
            try:
                trend_result = self.trend_analyzer.analyze()
            except Exception as e:
                print(f"[特征提取器] 趋势分析失败: {e}")
                return candidates
        
        if isinstance(trend_result, TrendAnalysisResult):
            trends = trend_result.top_recommendations
        elif isinstance(trend_result, dict):
            trends = [
                TrendItem(
                    source=TrendSource(t["source"]),
                    name=t["name"],
                    description=t["description"],
                    url=t["url"],
                    stars=t.get("stars", 0),
                    category=TrendCategory(t.get("category", "other")),
                    relevance_score=t.get("relevance_score", 0.0),
                    tags=t.get("tags", []),
                    metadata=t.get("metadata", {})
                ) for t in trend_result.get("top_recommendations", [])
            ]
        else:
            trends = []
        
        for trend in trends:
            category = self._map_trend_to_category(trend)
            priority = min(trend.relevance_score * 50 + (trend.stars / 10000) * 50, 100)
            
            candidate = FeatureCandidate(
                title=f"集成 {trend.name}",
                description=f"{trend.description} - 来源: {trend.url}",
                source=FeatureSource.TREND_ANALYSIS,
                category=category,
                priority_score=priority,
                metadata={
                    "stars": trend.stars,
                    "relevance_score": trend.relevance_score,
                    "url": trend.url,
                    "tags": trend.tags,
                    "source_type": trend.source.value
                },
                related_issues=[],
                suggested_actions=[
                    f"评估 {trend.name} 的适用性",
                    f"研究 {trend.name} 的集成方案",
                    f"制定迁移计划"
                ]
            )
            candidates.append(candidate)
        
        return candidates
    
    def _extract_from_smells(self, existing_data: Optional[Dict[str, Any]]) -> List[FeatureCandidate]:
        candidates = []
        
        if existing_data and "smells" in existing_data:
            smells = existing_data["smells"]
        else:
            try:
                smells = self.smell_detector.detect_all()
            except Exception as e:
                print(f"[特征提取器] 异味检测失败: {e}")
                return candidates
        
        smell_map = {
            SmellType.CYCLE_DEPENDENCY: "打破循环依赖",
            SmellType.DUPLICATE_CODE: "消除重复代码",
            SmellType.GOD_MODULE: "拆分上帝模块",
            SmellType.LONG_FUNCTION: "重构过长函数",
            SmellType.DATA_CLUMP: "消除数据团",
            SmellType.SHOTGUN_SURGERY: "消除霰弹式修改",
            SmellType.FEATURE_ENVY: "修复特性羡慕"
        }
        
        for smell in smells:
            title = smell_map.get(smell.smell_type, "架构重构")
            category = FeatureCategory.MAINTAINABILITY
            
            if smell.smell_type in [SmellType.CYCLE_DEPENDENCY, SmellType.GOD_MODULE]:
                category = FeatureCategory.TECH_DEBT
            
            priority = smell.priority
            
            candidate = FeatureCandidate(
                title=title,
                description=smell.description,
                source=FeatureSource.ARCHITECTURE_SMELL,
                category=category,
                priority_score=priority,
                metadata={
                    "smell_type": smell.smell_type.value,
                    "severity": smell.severity.value,
                    "locations": [
                        {"file_path": loc.file_path, "line_start": loc.line_start, "line_end": loc.line_end}
                        for loc in smell.locations
                    ]
                },
                related_issues=[smell.description],
                suggested_actions=self._get_smell_actions(smell)
            )
            candidates.append(candidate)
        
        return candidates
    
    def _extract_from_task_history(self, existing_data: Optional[Dict[str, Any]]) -> List[FeatureCandidate]:
        candidates = []
        
        if existing_data and "task_stats" in existing_data:
            task_stats = existing_data["task_stats"]
            type_stats = existing_data.get("task_type_stats", {})
        else:
            task_stats = self.task_board.get_task_stats()
            type_stats = self.task_board.get_task_type_stats()
        
        if task_stats.get("failed", 0) > 0:
            failure_rate = task_stats["failed"] / (task_stats.get("completed", 1) + task_stats.get("failed", 1))
            
            if failure_rate > 0.1:
                candidate = FeatureCandidate(
                    title="提高任务执行成功率",
                    description=f"当前任务失败率 {failure_rate:.1%}，需要优化任务执行流程",
                    source=FeatureSource.TASK_HISTORY,
                    category=FeatureCategory.RELIABILITY,
                    priority_score=min(failure_rate * 100, 80),
                    metadata={
                        "total_tasks": task_stats.get("total", 0),
                        "failed_tasks": task_stats.get("failed", 0),
                        "success_rate": 1 - failure_rate
                    },
                    related_issues=["任务失败率过高"],
                    suggested_actions=[
                        "分析失败任务的共同特征",
                        "优化任务调度策略",
                        "增加任务重试机制"
                    ]
                )
                candidates.append(candidate)
        
        for task_type, stats in type_stats.items():
            if stats.get("success_rate", 1.0) < 0.8:
                candidate = FeatureCandidate(
                    title=f"优化 {task_type} 类型任务",
                    description=f"{task_type} 任务成功率 {stats['success_rate']:.1%}，低于目标",
                    source=FeatureSource.TASK_HISTORY,
                    category=FeatureCategory.RELIABILITY,
                    priority_score=min((1 - stats["success_rate"]) * 80, 60),
                    metadata={
                        "task_type": task_type,
                        "completed": stats.get("completed", 0),
                        "failed": stats.get("failed", 0),
                        "success_rate": stats.get("success_rate", 0.0)
                    },
                    related_issues=[f"{task_type} 任务失败"],
                    suggested_actions=[
                        f"分析 {task_type} 任务失败原因",
                        f"改进 {task_type} 任务处理逻辑"
                    ]
                )
                candidates.append(candidate)
        
        return candidates
    
    def _extract_from_proof_failures(self, existing_data: Optional[Dict[str, Any]]) -> List[FeatureCandidate]:
        candidates = []
        
        if existing_data and "proof_report" in existing_data:
            report = existing_data["proof_report"]
        else:
            return candidates
        
        disproven_count = report.get("disproven", 0)
        unknown_count = report.get("unknown", 0)
        timeout_count = report.get("timeout", 0)
        
        if disproven_count > 0:
            candidate = FeatureCandidate(
                title="修复被证伪的属性",
                description=f"有 {disproven_count} 个属性被证伪，需要修复代码缺陷",
                source=FeatureSource.PROOF_FAILURE,
                category=FeatureCategory.RELIABILITY,
                priority_score=min(disproven_count * 20, 80),
                metadata={
                    "disproven_count": disproven_count,
                    "unknown_count": unknown_count,
                    "timeout_count": timeout_count
                },
                related_issues=["属性被证伪"],
                suggested_actions=[
                    "分析证伪的反例",
                    "修复相关代码缺陷",
                    "重新验证修复后的属性"
                ]
            )
            candidates.append(candidate)
        
        if unknown_count > disproven_count:
            candidate = FeatureCandidate(
                title="提升证明能力",
                description=f"有 {unknown_count} 个属性无法证明，需要增强证明策略",
                source=FeatureSource.PROOF_FAILURE,
                category=FeatureCategory.FEATURE_ENHANCEMENT,
                priority_score=min(unknown_count * 15, 60),
                metadata={
                    "unknown_count": unknown_count,
                    "timeout_count": timeout_count
                },
                related_issues=["证明超时/未知"],
                suggested_actions=[
                    "优化证明策略",
                    "增加证明超时时间",
                    "引入更强的证明引擎"
                ]
            )
            candidates.append(candidate)
        
        return candidates
    
    def _map_audit_issue_to_category(self, issue: AuditIssue) -> FeatureCategory:
        mapping = {
            AuditIssueType.SECURITY_VULNERABILITY: FeatureCategory.SECURITY,
            AuditIssueType.PERFORMANCE_BOTTLENECK: FeatureCategory.PERFORMANCE,
            AuditIssueType.CODE_SMELL: FeatureCategory.MAINTAINABILITY,
            AuditIssueType.FUNCTIONAL_GAP: FeatureCategory.FEATURE_ENHANCEMENT,
            AuditIssueType.ARCHITECTURE_ISSUE: FeatureCategory.TECH_DEBT,
        }
        return mapping.get(issue.issue_type, FeatureCategory.MAINTAINABILITY)
    
    def _map_trend_to_category(self, trend: TrendItem) -> FeatureCategory:
        mapping = {
            TrendCategory.WEB_FRAMEWORK: FeatureCategory.FEATURE_ENHANCEMENT,
            TrendCategory.DATABASE: FeatureCategory.PERFORMANCE,
            TrendCategory.ML_AI: FeatureCategory.FEATURE_ENHANCEMENT,
            TrendCategory.DEVOPS: FeatureCategory.DEVOPS,
            TrendCategory.SECURITY: FeatureCategory.SECURITY,
            TrendCategory.PERFORMANCE: FeatureCategory.PERFORMANCE,
            TrendCategory.TESTING: FeatureCategory.RELIABILITY,
            TrendCategory.OBSERVABILITY: FeatureCategory.OBSERVABILITY,
        }
        return mapping.get(trend.category, FeatureCategory.FEATURE_ENHANCEMENT)
    
    def _calculate_audit_priority(self, issue: AuditIssue) -> float:
        severity_weights = {
            IssueSeverity.CRITICAL: 100,
            IssueSeverity.HIGH: 75,
            IssueSeverity.MEDIUM: 50,
            IssueSeverity.LOW: 25
        }
        type_weights = {
            AuditIssueType.SECURITY_VULNERABILITY: 1.5,
            AuditIssueType.PERFORMANCE_BOTTLENECK: 1.2,
            AuditIssueType.ARCHITECTURE_ISSUE: 1.1,
            AuditIssueType.CODE_SMELL: 1.0,
            AuditIssueType.FUNCTIONAL_GAP: 0.9,
        }
        return severity_weights[issue.severity] * type_weights.get(issue.issue_type, 1.0)
    
    def _get_smell_actions(self, smell: SmellInstance) -> List[str]:
        actions = {
            SmellType.CYCLE_DEPENDENCY: [
                "识别循环依赖路径",
                "引入中间层解耦",
                "重新组织模块结构"
            ],
            SmellType.DUPLICATE_CODE: [
                "提取公共函数/类",
                "创建共享模块",
                "消除重复逻辑"
            ],
            SmellType.GOD_MODULE: [
                "识别职责边界",
                "拆分为多个小模块",
                "应用单一职责原则"
            ],
            SmellType.LONG_FUNCTION: [
                "提取子函数",
                "简化条件逻辑",
                "应用策略模式"
            ],
            SmellType.DATA_CLUMP: [
                "创建数据类封装",
                "使用对象替代参数列表",
                "减少参数传递"
            ],
            SmellType.SHOTGUN_SURGERY: [
                "集中常量定义",
                "使用配置文件",
                "创建单一数据源"
            ],
            SmellType.FEATURE_ENVY: [
                "移动方法到正确的类",
                "重新设计类职责",
                "应用迪米特法则"
            ]
        }
        return actions.get(smell.smell_type, ["分析问题", "制定修复方案"])
    
    def _deduplicate(self, candidates: List[FeatureCandidate]) -> List[FeatureCandidate]:
        seen_titles = set()
        unique = []
        
        for candidate in sorted(candidates, key=lambda c: -c.priority_score):
            normalized_title = candidate.title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique.append(candidate)
        
        return unique