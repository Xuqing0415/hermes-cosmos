from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os


class ArchitecturePattern(Enum):
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    HYBRID = "hybrid"


class ModuleType(Enum):
    CORE = "core"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    WORKER = "worker"
    SCHEDULER = "scheduler"
    OBSERVABILITY = "observability"
    SECURITY = "security"


@dataclass
class ArchitectureModule:
    name: str
    module_type: ModuleType
    responsibility: str
    technology: str
    interfaces: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    scalability: str = "horizontal"
    reliability: str = "high"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module_type": self.module_type.value,
            "responsibility": self.responsibility,
            "technology": self.technology,
            "interfaces": self.interfaces,
            "dependencies": self.dependencies,
            "scalability": self.scalability,
            "reliability": self.reliability
        }


@dataclass
class DataFlow:
    source: str
    destination: str
    data_type: str
    protocol: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "data_type": self.data_type,
            "protocol": self.protocol,
            "description": self.description
        }


@dataclass
class ArchitectureSpec:
    name: str
    version: str
    pattern: ArchitecturePattern
    modules: List[ArchitectureModule] = field(default_factory=list)
    data_flows: List[DataFlow] = field(default_factory=list)
    tech_stack: Dict[str, str] = field(default_factory=dict)
    design_principles: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    tla_spec: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "pattern": self.pattern.value,
            "modules": [m.to_dict() for m in self.modules],
            "data_flows": [df.to_dict() for df in self.data_flows],
            "tech_stack": self.tech_stack,
            "design_principles": self.design_principles,
            "improvement_areas": self.improvement_areas,
            "tla_spec_available": len(self.tla_spec) > 0
        }
    
    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "architecture_spec.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        if self.tla_spec:
            with open(os.path.join(output_dir, "architecture_spec.tla"), "w", encoding="utf-8") as f:
                f.write(self.tla_spec)


class ArchitectureDesigner:
    def __init__(self):
        self._current_spec: Optional[ArchitectureSpec] = None
    
    def design(self, audit_issues: List[Any], trend_results: Any) -> ArchitectureSpec:
        print("[架构设计] 开始设计下一代架构...")
        
        self._current_spec = ArchitectureSpec(
            name="AutoTestGen NextGen",
            version="3.0",
            pattern=self._determine_pattern(audit_issues, trend_results)
        )
        
        self._define_modules(audit_issues, trend_results)
        self._define_data_flows()
        self._define_tech_stack(trend_results)
        self._generate_tla_spec()
        
        self._print_summary(self._current_spec)
        
        return self._current_spec
    
    def _determine_pattern(self, issues: List[Any], trends: Any) -> ArchitecturePattern:
        has_performance_issues = any(
            issue.issue_type.value == "performance_bottleneck"
            for issue in issues
        )
        
        has_scalability_trends = any(
            trend.category.value == "devops" and "kubernetes" in trend.tags
            for trend in trends.top_recommendations
        )
        
        if has_performance_issues or has_scalability_trends:
            return ArchitecturePattern.MICROSERVICES
        
        return ArchitecturePattern.HYBRID
    
    def _define_modules(self, issues: List[Any], trends: Any):
        modules = []
        
        core_tech = "Python 3.11+"
        if any(t.name.lower() == "rust" or "rust" in t.tags for t in trends.top_recommendations):
            core_tech = "Rust + Python"
        
        modules.append(ArchitectureModule(
            name="requirement_engine",
            module_type=ModuleType.CORE,
            responsibility="自然语言需求解析、结构化",
            technology=core_tech,
            interfaces=["parse_requirement()", "refine_requirement()"],
            dependencies=[]
        ))
        
        modules.append(ArchitectureModule(
            name="spec_generator",
            module_type=ModuleType.CORE,
            responsibility="TLA+ 规范生成与验证",
            technology="Python + TLA+",
            interfaces=["generate_spec()", "verify_spec()"],
            dependencies=["requirement_engine"]
        ))
        
        modules.append(ArchitectureModule(
            name="code_generator",
            module_type=ModuleType.CORE,
            responsibility="多语言代码生成（Python/Go/Rust）",
            technology="Python + Jinja2 + polyglot",
            interfaces=["generate_code()", "generate_deployment()"],
            dependencies=["spec_generator", "requirement_engine"]
        ))
        
        modules.append(ArchitectureModule(
            name="deployment_engine",
            module_type=ModuleType.WORKER,
            responsibility="Kubernetes 部署与管理",
            technology="Python + kubernetes client",
            interfaces=["deploy()", "scale()", "rollback()"],
            dependencies=["code_generator"]
        ))
        
        modules.append(ArchitectureModule(
            name="runtime_monitor",
            module_type=ModuleType.OBSERVABILITY,
            responsibility="运行时监控与状态追踪",
            technology="Python + OpenTelemetry",
            interfaces=["record_transition()", "detect_violation()"],
            dependencies=["spec_generator"]
        ))
        
        modules.append(ArchitectureModule(
            name="anomaly_analyzer",
            module_type=ModuleType.CORE,
            responsibility="异常分析与根因定位",
            technology="Python",
            interfaces=["analyze()", "suggest_fix()"],
            dependencies=["runtime_monitor"]
        ))
        
        modules.append(ArchitectureModule(
            name="self_audit",
            module_type=ModuleType.CORE,
            responsibility="代码库自我审计",
            technology="Python + AST",
            interfaces=["audit()", "report_issues()"],
            dependencies=[]
        ))
        
        modules.append(ArchitectureModule(
            name="trend_analyzer",
            module_type=ModuleType.CORE,
            responsibility="技术趋势分析",
            technology="Python + HTTP",
            interfaces=["analyze_trends()", "score_relevance()"],
            dependencies=[]
        ))
        
        modules.append(ArchitectureModule(
            name="security_gateway",
            module_type=ModuleType.SECURITY,
            responsibility="安全验证与访问控制",
            technology="Python + OAuth2",
            interfaces=["validate_request()", "authenticate()"],
            dependencies=[]
        ))
        
        self._current_spec.modules = modules
    
    def _define_data_flows(self):
        flows = [
            DataFlow(
                source="requirement_engine",
                destination="spec_generator",
                data_type="ParsedRequirement",
                protocol="function call",
                description="结构化需求传递"
            ),
            DataFlow(
                source="spec_generator",
                destination="code_generator",
                data_type="TLASpec",
                protocol="function call",
                description="TLA+ 规范传递"
            ),
            DataFlow(
                source="code_generator",
                destination="deployment_engine",
                data_type="GeneratedService",
                protocol="file system",
                description="生成的代码工件"
            ),
            DataFlow(
                source="runtime_monitor",
                destination="anomaly_analyzer",
                data_type="ViolationReport",
                protocol="event",
                description="违规事件传递"
            ),
            DataFlow(
                source="self_audit",
                destination="architecture_designer",
                data_type="AuditResult",
                protocol="function call",
                description="审计结果传递"
            ),
            DataFlow(
                source="trend_analyzer",
                destination="architecture_designer",
                data_type="TrendAnalysisResult",
                protocol="function call",
                description="趋势分析结果传递"
            ),
            DataFlow(
                source="security_gateway",
                destination="all_modules",
                data_type="AuthToken",
                protocol="middleware",
                description="安全认证"
            )
        ]
        
        self._current_spec.data_flows = flows
    
    def _define_tech_stack(self, trends: Any):
        tech_stack = {
            "language": "Python 3.11+",
            "web_framework": "FastAPI",
            "database": "PostgreSQL",
            "cache": "Redis",
            "message_queue": "RabbitMQ",
            "observability": "OpenTelemetry + Prometheus",
            "container": "Docker",
            "orchestration": "Kubernetes"
        }
        
        for trend in trends.top_recommendations:
            if "rust" in trend.tags:
                tech_stack["language"] = "Rust + Python (hybrid)"
            if trend.category.value == "web_framework":
                tech_stack["web_framework"] = trend.name
            if trend.category.value == "database":
                tech_stack["database"] = trend.name
            if "async" in trend.tags:
                tech_stack["web_framework"] = "FastAPI (async)"
        
        self._current_spec.tech_stack = tech_stack
        
        design_principles = [
            "单一职责原则",
            "依赖倒置原则",
            "接口隔离原则",
            "开闭原则",
            "可观测性优先",
            "防御性编程",
            "渐进式演进"
        ]
        self._current_spec.design_principles = design_principles
    
    def _generate_tla_spec(self):
        spec = r"""---- MODULE AutoTestGenArchitecture ----
EXTENDS Naturals, TLC, Sequences

CONSTANTS ModuleNames, InterfaceNames

Modules == ModuleNames
Interfaces == InterfaceNames

(* 模块状态 *)
State == [
    modules: Modules -> {idle, working, error},
    data_flows: (Modules \X Modules) -> BOOLEAN,
    pending_tasks: Seq(InterfaceNames)
]

(* 初始状态 *)
Init == /\\
    \A m \in Modules: modules[m] = idle
    /\\
    \A m1, m2 \in Modules: data_flows[m1, m2] = FALSE
    /\\
    pending_tasks = <<>>

(* 启动模块 *)
StartModule(m) == /\\
    modules[m] = idle
    /\\
    modules' = [modules EXCEPT ![m] = working]

(* 停止模块 *)
StopModule(m) == /\\
    modules[m] = working
    /\\
    modules' = [modules EXCEPT ![m] = idle]

(* 数据流动 *)
DataFlow(m1, m2) == /\\
    modules[m1] = working
    /\\
    modules[m2] = working
    /\\
    data_flows[m1, m2] = FALSE
    /\\
    data_flows' = [data_flows EXCEPT ![m1, m2] = TRUE]

(* 完成数据流动 *)
CompleteFlow(m1, m2) == /\\
    data_flows[m1, m2] = TRUE
    /\\
    data_flows' = [data_flows EXCEPT ![m1, m2] = FALSE]

(* 添加任务 *)
AddTask(task) == /\\
    pending_tasks' = Append(pending_tasks, task)

(* 处理任务 *)
ProcessTask == /\\
    Len(pending_tasks) > 0
    /\\
    pending_tasks' = Tail(pending_tasks)

Next ==
    \E m \in Modules: StartModule(m)
    \/ \E m \in Modules: StopModule(m)
    \/ \E m1, m2 \in Modules: DataFlow(m1, m2)
    \/ \E m1, m2 \in Modules: CompleteFlow(m1, m2)
    \/ AddTask("")
    \/ ProcessTask

(* 安全性属性：无死锁 *)
NoDeadlock ==
    ~(modules = [m \in Modules |-> working] /\\ data_flows = [m1, m2 \in Modules |-> FALSE])

(* 活性属性：所有任务都会被处理 *)
AllTasksProcessed ==
    [] (Len(pending_tasks) > 0 => <><<ProcessTask>>>)

(* 模块可用性 *)
ModuleAvailable(m) ==
    [] (<>modules[m] = working)

====
"""
        self._current_spec.tla_spec = spec
    
    def _print_summary(self, spec: ArchitectureSpec):
        print(f"[架构设计] 架构设计完成: {spec.name} v{spec.version}")
        print(f"[架构设计] 架构模式: {spec.pattern.value}")
        print(f"\n[架构设计] 技术栈:")
        for key, value in spec.tech_stack.items():
            print(f"  {key}: {value}")
        
        print(f"\n[架构设计] 模块列表 ({len(spec.modules)} 个):")
        for module in spec.modules:
            print(f"  - {module.name} ({module.module_type.value})")
            print(f"     职责: {module.responsibility}")
            print(f"     技术: {module.technology}")
        
        print(f"\n[架构设计] 数据流 ({len(spec.data_flows)} 条):")
        for flow in spec.data_flows[:5]:
            print(f"  {flow.source} -> {flow.destination} [{flow.data_type}]")