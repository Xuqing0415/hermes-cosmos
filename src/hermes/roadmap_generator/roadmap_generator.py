from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import os
import json
import urllib.request
import urllib.parse
import ssl

from .features_extractor import FeatureExtractor, FeatureCandidate
from .impact_estimator import ImpactEstimator, ImpactScore
from .workload_estimator import WorkloadEstimator, WorkloadEstimate
from .multi_objective_optimizer import MultiObjectiveOptimizer, PrioritizedFeature, PriorityLevel


@dataclass
class RoadmapConfig:
    output_dir: str = "./roadmaps"
    generate_md: bool = True
    create_github_issue: bool = False
    github_repo: str = ""
    github_token: str = ""
    max_features: int = 15
    budget_days: float = 20
    include_archived: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "generate_md": self.generate_md,
            "create_github_issue": self.create_github_issue,
            "github_repo": self.github_repo,
            "max_features": self.max_features,
            "budget_days": self.budget_days,
            "include_archived": self.include_archived
        }


@dataclass
class RoadmapReport:
    version: str = ""
    generated_at: str = ""
    features: List[PrioritizedFeature] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    budget_allocation: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "features": [f.to_dict() for f in self.features],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "budget_allocation": self.budget_allocation
        }


class RoadmapGenerator:
    VERSION = "3.1"
    
    def __init__(self, config: Optional[RoadmapConfig] = None):
        self.config = config or RoadmapConfig()
        self.feature_extractor = FeatureExtractor()
        self.impact_estimator = ImpactEstimator()
        self.workload_estimator = WorkloadEstimator()
        self.optimizer = MultiObjectiveOptimizer(max_workload=self.config.budget_days)
        
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
    
    def generate(self, existing_data: Optional[Dict[str, Any]] = None) -> RoadmapReport:
        print("[路线图生成器] 开始生成进化路线图...")
        
        candidates = self.feature_extractor.extract_all(existing_data)
        
        if len(candidates) == 0:
            print("[路线图生成器] 未提取到候选特性，使用默认数据")
            candidates = self._generate_default_candidates()
        
        candidates = candidates[:self.config.max_features]
        
        impacts = self.impact_estimator.estimate(candidates)
        workloads = self.workload_estimator.estimate(candidates)
        
        prioritized = self.optimizer.prioritize(candidates, impacts, workloads)
        
        portfolio, total_workload = self.optimizer.select_budget_portfolio(
            candidates, impacts, workloads, self.config.budget_days
        )
        
        report = self._create_report(prioritized, portfolio, total_workload)
        
        if self.config.generate_md:
            md_path = self._generate_markdown_report(report)
            print(f"[路线图生成器] Markdown 报告已保存: {md_path}")
        
        if self.config.create_github_issue and self.config.github_repo:
            self._create_github_issue(report)
        
        print("[路线图生成器] 路线图生成完成")
        
        return report
    
    def _generate_default_candidates(self) -> List[FeatureCandidate]:
        from .features_extractor import FeatureSource, FeatureCategory
        
        return [
            FeatureCandidate(
                title="优化符号执行核心性能",
                description="当前符号执行在复杂代码路径上性能较慢，需要优化算法和数据结构",
                source=FeatureSource.SELF_AUDIT,
                category=FeatureCategory.PERFORMANCE,
                priority_score=85.0,
                suggested_actions=["分析性能瓶颈", "优化约束求解", "引入并行处理"]
            ),
            FeatureCandidate(
                title="增强安全漏洞检测能力",
                description="当前安全扫描规则有限，需要扩展检测规则库",
                source=FeatureSource.SELF_AUDIT,
                category=FeatureCategory.SECURITY,
                priority_score=80.0,
                suggested_actions=["扩展安全规则", "集成 SAST 工具", "添加 CVE 数据库"]
            ),
            FeatureCandidate(
                title="集成 OpenTelemetry 分布式追踪",
                description="添加完整的分布式追踪能力，提升可观测性",
                source=FeatureSource.TREND_ANALYSIS,
                category=FeatureCategory.OBSERVABILITY,
                priority_score=75.0,
                metadata={"stars": 4500, "tags": ["python", "observability", "monitoring"]},
                suggested_actions=["集成 OpenTelemetry SDK", "配置 exporters", "创建追踪仪表盘"]
            ),
            FeatureCandidate(
                title="消除循环依赖",
                description="检测到模块间存在循环依赖，影响可维护性",
                source=FeatureSource.ARCHITECTURE_SMELL,
                category=FeatureCategory.TECH_DEBT,
                priority_score=70.0,
                suggested_actions=["识别循环路径", "引入中间层", "重构模块结构"]
            ),
            FeatureCandidate(
                title="支持 Kubernetes Helm Chart 自动生成",
                description="自动为生成的服务创建 Helm Chart，简化部署流程",
                source=FeatureSource.TREND_ANALYSIS,
                category=FeatureCategory.DEVOPS,
                priority_score=65.0,
                metadata={"stars": 5000, "tags": ["python", "kubernetes"]},
                suggested_actions=["设计 Chart 模板", "集成到部署流程", "添加值配置"]
            ),
        ]
    
    def _create_report(self, prioritized: List[PrioritizedFeature], 
                       portfolio: List[PrioritizedFeature], 
                       total_workload: float) -> RoadmapReport:
        p0_count = sum(1 for f in prioritized if f.priority == PriorityLevel.P0)
        p1_count = sum(1 for f in prioritized if f.priority == PriorityLevel.P1)
        p2_count = sum(1 for f in prioritized if f.priority == PriorityLevel.P2)
        p3_count = sum(1 for f in prioritized if f.priority == PriorityLevel.P3)
        
        total_impact = sum(f.impact.overall for f in prioritized)
        avg_impact = total_impact / len(prioritized) if prioritized else 0
        
        short_term = [f for f in portfolio if f.priority in [PriorityLevel.P0, PriorityLevel.P1]][:3]
        long_term = [f for f in prioritized if f.priority in [PriorityLevel.P2, PriorityLevel.P3]][:3]
        
        recommendations = []
        if short_term:
            recommendations.append(f"下一阶段（2周）: {'、'.join(f.candidate.title for f in short_term)}")
        if long_term:
            recommendations.append(f"长期（1个月）: {'、'.join(f.candidate.title for f in long_term)}")
        
        budget_allocation = {}
        for pf in portfolio:
            cat = pf.candidate.category.value
            budget_allocation[cat] = budget_allocation.get(cat, 0) + pf.workload.estimate
        
        return RoadmapReport(
            version=self.VERSION,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            features=prioritized,
            summary={
                "total_features": len(prioritized),
                "p0_count": p0_count,
                "p1_count": p1_count,
                "p2_count": p2_count,
                "p3_count": p3_count,
                "total_workload": total_workload,
                "avg_impact": round(avg_impact, 1),
                "portfolio_size": len(portfolio)
            },
            recommendations=recommendations,
            budget_allocation=budget_allocation
        )
    
    def _generate_markdown_report(self, report: RoadmapReport) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"roadmap_{timestamp}.md"
        filepath = os.path.join(self.config.output_dir, filename)
        
        md = self._build_markdown(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
        
        latest_path = os.path.join(self.config.output_dir, "roadmap.md")
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        return filepath
    
    def _build_markdown(self, report: RoadmapReport) -> str:
        md_parts = []
        
        md_parts.append(f"# AutoTestGen {report.version} 进化路线图\n")
        md_parts.append(f"\n生成时间：{report.generated_at}\n")
        
        md_parts.append("\n## 执行摘要\n")
        md_parts.append(f"- **候选特性总数**: {report.summary['total_features']}\n")
        md_parts.append(f"- **P0 优先级**: {report.summary['p0_count']} 个\n")
        md_parts.append(f"- **P1 优先级**: {report.summary['p1_count']} 个\n")
        md_parts.append(f"- **P2 优先级**: {report.summary['p2_count']} 个\n")
        md_parts.append(f"- **P3 优先级**: {report.summary['p3_count']} 个\n")
        md_parts.append(f"- **推荐组合工作量**: {report.summary['total_workload']} 人天\n")
        
        md_parts.append("\n## 候选特性列表\n")
        md_parts.append("\n| 优先级 | 特性 | 预期收益 | 预估工作量 | ROI | 风险 |")
        md_parts.append("\n|--------|------|----------|------------|-----|------|")
        
        for pf in report.features[:10]:
            impact_str = ", ".join([f"{k}: {v}" for k, v in pf.impact.metrics_improvement.items()])
            if not impact_str:
                impact_str = pf.impact.description
            
            md_parts.append(f"\n| {pf.priority.value} | {pf.candidate.title} | {impact_str} | {pf.workload.estimate} 人天 | {pf.roi:.1f} | {pf.risk_level} |")
        
        md_parts.append("\n## 优先级说明\n")
        md_parts.append("\n| 优先级 | 含义 | 建议处理时间 |")
        md_parts.append("\n|--------|------|--------------|")
        md_parts.append("\n| P0 | 关键特性，必须优先处理 | 立即 |")
        md_parts.append("\n| P1 | 重要特性，近期处理 | 1-2周 |")
        md_parts.append("\n| P2 | 一般特性，规划处理 | 1个月 |")
        md_parts.append("\n| P3 | 低优先级，酌情处理 | 后续版本 |")
        
        md_parts.append("\n## 推荐优先实施\n")
        for rec in report.recommendations:
            md_parts.append(f"\n**{rec}**\n")
        
        p0_features = [pf for pf in report.features if pf.priority == PriorityLevel.P0]
        if p0_features:
            md_parts.append("\n### P0 特性详情\n")
            for pf in p0_features:
                md_parts.append(f"\n#### {pf.candidate.title}\n")
                md_parts.append(f"\n**描述**: {pf.candidate.description}\n")
                md_parts.append(f"\n**类别**: {pf.candidate.category.value}\n")
                md_parts.append(f"\n**预期收益**: {pf.impact.description}\n")
                md_parts.append(f"\n**预估工作量**: {pf.workload.estimate} 人天 ({pf.workload.min_estimate}-{pf.workload.max_estimate})\n")
                md_parts.append(f"\n**风险等级**: {pf.risk_level}\n")
                if pf.candidate.suggested_actions:
                    md_parts.append("\n**建议行动**:\n")
                    for i, action in enumerate(pf.candidate.suggested_actions, 1):
                        md_parts.append(f"- {i}. {action}\n")
        
        md_parts.append("\n## 工作量分配\n")
        total_budget = report.summary['total_workload']
        for cat, workload in report.budget_allocation.items():
            percentage = (workload / total_budget) * 100 if total_budget > 0 else 0
            md_parts.append(f"- {cat}: {workload} 人天 ({percentage:.1f}%)\n")
        
        md_parts.append("\n## 数据源\n")
        md_parts.append("- 自我审视报告\n")
        md_parts.append("- 技术趋势分析\n")
        md_parts.append("- 架构异味检测\n")
        md_parts.append("- 任务执行历史\n")
        md_parts.append("- 证明失败记录\n")
        
        md_parts.append("\n---\n")
        md_parts.append("*本路线图由 AutoTestGen 自动生成，等待人类审批*\n")
        
        return "".join(md_parts)
    
    def _create_github_issue(self, report: RoadmapReport):
        if not self.config.github_token:
            print("[路线图生成器] 未提供 GitHub Token，跳过创建 Issue")
            return
        
        title = f"[Roadmap] AutoTestGen {report.version} 进化路线图"
        body = self._build_markdown(report)
        
        url = f"https://api.github.com/repos/{self.config.github_repo}/issues"
        data = json.dumps({
            "title": title,
            "body": body,
            "labels": ["roadmap", "auto-generated"]
        }).encode()
        
        headers = {
            "Authorization": f"token {self.config.github_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                result = json.loads(resp.read())
                print(f"[路线图生成器] GitHub Issue 创建成功: {result.get('html_url')}")
        except Exception as e:
            print(f"[路线图生成器] 创建 GitHub Issue 失败: {e}")
    
    def save_json(self, report: RoadmapReport, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"roadmap_{timestamp}.json"
        
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        return filepath