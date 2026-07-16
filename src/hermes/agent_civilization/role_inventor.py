import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .types import AgentRole, TaskStatus


class FailurePatternType(Enum):
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CAPABILITY_MISSING = "capability_missing"
    LOGIC_ERROR = "logic_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    INVALID_INPUT = "invalid_input"
    INTERNAL_ERROR = "internal_error"


@dataclass
class FailurePattern:
    pattern_id: str
    task_type: str
    error_type: str
    error_message: str
    frequency: int = 0
    affected_tasks: List[str] = field(default_factory=list)
    suggested_role: Optional[str] = None
    
    def __post_init__(self):
        if self.pattern_id == "":
            self.pattern_id = str(uuid.uuid4())[:8]


class RoleTemplate(Enum):
    CROSS_LANG_VERIFIER = "cross_lang_verifier"
    PERFORMANCE_ANALYZER = "performance_analyzer"
    SECURITY_AUDITOR = "security_auditor"
    DATA_VALIDATOR = "data_validator"
    INFRASTRUCTURE_MANAGER = "infrastructure_manager"
    DOCUMENTATION_GENERATOR = "documentation_generator"
    COMPLIANCE_CHECKER = "compliance_checker"


class RoleCategory(Enum):
    TESTING = "testing"
    FIXING = "fixing"
    VERIFICATION = "verification"
    ANALYSIS = "analysis"
    MANAGEMENT = "management"
    CREATION = "creation"


@dataclass
class FailureAnalysis:
    task_type: str
    failure_pattern: FailurePattern
    failure_count: int = 0
    last_failure_time: float = 0.0
    affected_roles: List[str] = field(default_factory=list)
    suggested_roles: List[str] = field(default_factory=list)


@dataclass
class RoleDefinition:
    role_id: str
    name: str
    category: Optional[str]
    description: str
    responsibilities: List[str]
    capabilities: List[str]
    input_spec: Dict[str, Any]
    output_spec: Dict[str, Any]
    required_skills: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.role_id == "":
            self.role_id = str(uuid.uuid4())[:8]


@dataclass
class GeneratedRole:
    role_name: str
    role_id: str
    description: str
    capabilities: List[str]
    required_modules: List[str]
    dependent_roles: List[str]
    output_spec: Dict[str, Any]
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.role_id == "":
            self.role_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_name": self.role_name,
            "role_id": self.role_id,
            "description": self.description,
            "capabilities": self.capabilities,
            "required_modules": self.required_modules,
            "dependent_roles": self.dependent_roles,
            "output_spec": self.output_spec,
            "created_at": self.created_at
        }
    
    def to_role_definition(self) -> RoleDefinition:
        return RoleDefinition(
            role_id=self.role_id,
            name=self.role_name.replace('_', ' ').title(),
            category=None,
            description=self.description,
            responsibilities=self.capabilities,
            capabilities=self.capabilities,
            input_spec={},
            output_spec=self.output_spec,
            required_skills=self.required_modules
        )


class RoleDemandAnalyzer:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failure_history: List[Dict[str, Any]] = []
        self.analyses: Dict[str, FailureAnalysis] = {}
    
    def record_failure(self, task_type: str, failure_pattern: FailurePattern,
                       agent_role: str, details: Dict[str, Any]):
        failure_record = {
            "task_type": task_type,
            "failure_pattern": failure_pattern.error_type,
            "agent_role": agent_role,
            "details": details,
            "timestamp": time.time()
        }
        self.failure_history.append(failure_record)
        
        key = f"{task_type}_{failure_pattern.error_type}"
        if key not in self.analyses:
            self.analyses[key] = FailureAnalysis(
                task_type=task_type,
                failure_pattern=failure_pattern,
                affected_roles=[],
                suggested_roles=[]
            )
        
        analysis = self.analyses[key]
        analysis.failure_count += 1
        analysis.last_failure_time = time.time()
        
        if agent_role not in analysis.affected_roles:
            analysis.affected_roles.append(agent_role)
        
        self._update_suggested_roles(analysis)
    
    def _update_suggested_roles(self, analysis: FailureAnalysis):
        error_type = analysis.failure_pattern.error_type.lower()
        
        if "capability" in error_type or "missing" in error_type:
            missing_cap = analysis.failure_pattern.error_message
            if "cross" in missing_cap.lower() or "multi" in missing_cap.lower():
                analysis.suggested_roles.append(RoleTemplate.CROSS_LANG_VERIFIER.value)
            elif "security" in missing_cap.lower():
                analysis.suggested_roles.append(RoleTemplate.SECURITY_AUDITOR.value)
            elif "performance" in missing_cap.lower():
                analysis.suggested_roles.append(RoleTemplate.PERFORMANCE_ANALYZER.value)
            elif "data" in missing_cap.lower():
                analysis.suggested_roles.append(RoleTemplate.DATA_VALIDATOR.value)
        
        elif "timeout" in error_type:
            analysis.suggested_roles.append(RoleTemplate.PERFORMANCE_ANALYZER.value)
        
        elif "resource" in error_type or "exhausted" in error_type:
            analysis.suggested_roles.append(RoleTemplate.INFRASTRUCTURE_MANAGER.value)
        
        elif "logic" in error_type or "complexity" in error_type:
            analysis.suggested_roles.append(RoleTemplate.DATA_VALIDATOR.value)
        
        elif "security" in error_type:
            analysis.suggested_roles.append(RoleTemplate.SECURITY_AUDITOR.value)
    
    def get_role_demand(self) -> List[FailureAnalysis]:
        return [
            analysis for analysis in self.analyses.values()
            if analysis.failure_count >= self.failure_threshold
        ]
    
    def get_suggested_new_roles(self) -> List[str]:
        suggested = []
        for analysis in self.get_role_demand():
            suggested.extend(analysis.suggested_roles)
        return list(set(suggested))
    
    def get_failure_stats(self) -> Dict[str, Any]:
        stats = {"by_pattern": {}, "by_task_type": {}}
        
        for failure in self.failure_history:
            pattern = failure["failure_pattern"]
            task_type = failure["task_type"]
            
            stats["by_pattern"][pattern] = stats["by_pattern"].get(pattern, 0) + 1
            stats["by_task_type"][task_type] = stats["by_task_type"].get(task_type, 0) + 1
        
        return stats


class RoleDefinitionGenerator:
    _role_templates = {
        RoleTemplate.CROSS_LANG_VERIFIER: {
            "description": "跨语言验证器，负责验证不同编程语言代码的等价性",
            "capabilities": ["cross_lang_verify", "code_translation", "language_analysis"],
            "required_modules": ["polyglot", "autolang", "test_generator"],
            "dependent_roles": ["LANGUAGE_PRIEST", "MIGRATION_APOSTLE"],
            "output_spec": {
                "verified": bool,
                "equivalence_proof": str,
                "translation_result": str,
                "issues": list
            }
        },
        RoleTemplate.PERFORMANCE_ANALYZER: {
            "description": "性能分析器，负责分析代码执行性能和优化建议",
            "capabilities": ["performance_analysis", "benchmarking", "optimization_suggestion"],
            "required_modules": ["profiler", "mlir_verifier"],
            "dependent_roles": ["COMPILER_VERIFIER", "ARCHITECTURE_JUDGE"],
            "output_spec": {
                "analysis_report": str,
                "bottlenecks": list,
                "optimization_suggestions": list,
                "performance_metrics": dict
            }
        },
        RoleTemplate.SECURITY_AUDITOR: {
            "description": "安全审计员，负责检测代码中的安全漏洞",
            "capabilities": ["security_scan", "vulnerability_detection", "secure_coding"],
            "required_modules": ["security_scanner", "test_generator"],
            "dependent_roles": ["TEST_OFFICER", "ARCHITECTURE_JUDGE"],
            "output_spec": {
                "security_report": str,
                "vulnerabilities": list,
                "risk_level": str,
                "fix_suggestions": list
            }
        },
        RoleTemplate.DATA_VALIDATOR: {
            "description": "数据验证器，负责验证输入输出数据的正确性",
            "capabilities": ["data_validation", "constraint_checking", "type_analysis"],
            "required_modules": ["autolang", "test_generator"],
            "dependent_roles": ["TEST_OFFICER", "PROVER"],
            "output_spec": {
                "valid": bool,
                "validation_report": str,
                "errors": list,
                "constraints": list
            }
        },
        RoleTemplate.INFRASTRUCTURE_MANAGER: {
            "description": "基础设施管理器，负责管理计算资源和环境配置",
            "capabilities": ["resource_management", "environment_setup", "scaling"],
            "required_modules": ["resource_manager"],
            "dependent_roles": ["RESOURCE_OVERSEER"],
            "output_spec": {
                "resources_allocated": dict,
                "environment_ready": bool,
                "scaling_decision": str
            }
        },
        RoleTemplate.DOCUMENTATION_GENERATOR: {
            "description": "文档生成器，负责自动生成代码文档",
            "capabilities": ["documentation_generation", "code_explanation", "API_docs"],
            "required_modules": ["autolang"],
            "dependent_roles": ["LANGUAGE_PRIEST"],
            "output_spec": {
                "documentation": str,
                "api_reference": str,
                "examples": list
            }
        },
        RoleTemplate.COMPLIANCE_CHECKER: {
            "description": "合规检查器，负责检查代码是否符合规范和标准",
            "capabilities": ["compliance_check", "code_style", "standard_verification"],
            "required_modules": ["autolang"],
            "dependent_roles": ["ARCHITECTURE_JUDGE"],
            "output_spec": {
                "compliant": bool,
                "violations": list,
                "suggestions": list
            }
        }
    }
    
    def __init__(self):
        self.generated_roles: Dict[str, GeneratedRole] = {}
    
    def generate_role(self, role_template: RoleTemplate,
                      custom_params: Optional[Dict[str, Any]] = None) -> GeneratedRole:
        template = self._role_templates.get(role_template)
        if not template:
            raise ValueError(f"Unknown role template: {role_template}")
        
        params = custom_params or {}
        
        generated_role = GeneratedRole(
            role_name=params.get("role_name", role_template.value),
            role_id="",
            description=params.get("description", template["description"]),
            capabilities=params.get("capabilities", template["capabilities"]),
            required_modules=params.get("required_modules", template["required_modules"]),
            dependent_roles=params.get("dependent_roles", template["dependent_roles"]),
            output_spec=params.get("output_spec", template["output_spec"])
        )
        
        self.generated_roles[generated_role.role_id] = generated_role
        return generated_role
    
    def generate_from_failure(self, failure_analysis: FailureAnalysis) -> List[GeneratedRole]:
        generated = []
        for suggested_role in failure_analysis.suggested_roles:
            template = self._find_template_by_name(suggested_role)
            if template:
                generated.append(self.generate_role(template))
        return generated
    
    def _find_template_by_name(self, name: str) -> Optional[RoleTemplate]:
        for template in RoleTemplate:
            if template.value == name:
                return template
        return None
    
    def generate_custom_role(self, role_name: str, description: str,
                             capabilities: List[str], required_modules: List[str],
                             dependent_roles: List[str],
                             output_spec: Dict[str, Any]) -> GeneratedRole:
        generated_role = GeneratedRole(
            role_name=role_name,
            role_id="",
            description=description,
            capabilities=capabilities,
            required_modules=required_modules,
            dependent_roles=dependent_roles,
            output_spec=output_spec
        )
        
        self.generated_roles[generated_role.role_id] = generated_role
        return generated_role
    
    def get_generated_role(self, role_id: str) -> Optional[GeneratedRole]:
        return self.generated_roles.get(role_id)
    
    def get_all_generated_roles(self) -> List[GeneratedRole]:
        return list(self.generated_roles.values())
    
    def generate_role_code(self, generated_role: GeneratedRole) -> str:
        class_name = ''.join(word.capitalize() for word in generated_role.role_name.split('_'))
        
        capabilities_str = ',\n        '.join(f'"{cap}"' for cap in generated_role.capabilities)
        output_spec_str = ',\n            '.join(f'"{k}": {v.__name__}' for k, v in generated_role.output_spec.items())
        
        code = f"""from typing import Dict, Any, Optional
from ..base_agent import BaseAgent
from .types import AgentRole

class {class_name}(BaseAgent):
    def __init__(self, agent_id: str = ""):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RESOURCE_OVERSEER,
            name="{generated_role.role_name.replace('_', ' ').title()}",
            capabilities=[
                {capabilities_str}
            ]
        )
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_type = task.get("type", "")
        
        if task_type in self.capabilities:
            return await self._handle_{generated_role.role_name}(task)
        
        return await self._default_handle(task)
    
    async def _handle_{generated_role.role_name}(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {{
            "success": True,
            "role": "{generated_role.role_name}",
            "output": {{}}
        }}
    
    async def _default_handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {{
            "success": False,
            "error": f"Task type {{task.get('type')}} not supported",
            "role": "{generated_role.role_name}"
        }}
"""
        return code
    
    def save_role_code(self, generated_role: GeneratedRole, output_dir: str = "agents") -> str:
        import os
        filename = f"{generated_role.role_name}.py"
        filepath = os.path.join(output_dir, filename)
        code = self.generate_role_code(generated_role)
        
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        
        return filepath


class RoleInventor:
    def __init__(self):
        self.demand_analyzer = RoleDemandAnalyzer()
        self.definition_generator = RoleDefinitionGenerator()
        self.generated_role_definitions: List[RoleDefinition] = []
    
    def analyze_failures(self, task_history: List[Dict[str, Any]]) -> List[FailurePattern]:
        patterns = []
        
        error_type_counts: Dict[str, int] = {}
        for task in task_history:
            error_type = task.get("error_type", "unknown")
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        for error_type, frequency in error_type_counts.items():
            if frequency >= 3:
                pattern = FailurePattern(
                    pattern_id="",
                    task_type=task_history[0].get("task_type", "unknown"),
                    error_type=error_type,
                    error_message=task_history[0].get("error_message", ""),
                    frequency=frequency,
                    affected_tasks=[t.get("task_id") for t in task_history[:5]]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_failure_pattern(self, task: Dict[str, Any]) -> FailurePattern:
        error_msg = task.get("error_message", "").lower()
        
        if "timeout" in error_msg or "time limit" in error_msg:
            return FailurePattern(
                pattern_id="",
                task_type=task.get("type", "unknown"),
                error_type="timeout",
                error_message=task.get("error_message", "")
            )
        elif "memory" in error_msg or "resource" in error_msg:
            return FailurePattern(
                pattern_id="",
                task_type=task.get("type", "unknown"),
                error_type="resource_exhausted",
                error_message=task.get("error_message", "")
            )
        elif "capability" in error_msg or "not supported" in error_msg:
            return FailurePattern(
                pattern_id="",
                task_type=task.get("type", "unknown"),
                error_type="capability_missing",
                error_message=task.get("error_message", "")
            )
        elif "logic" in error_msg or "assertion" in error_msg:
            return FailurePattern(
                pattern_id="",
                task_type=task.get("type", "unknown"),
                error_type="logic_error",
                error_message=task.get("error_message", "")
            )
        elif "dependency" in error_msg or "import" in error_msg:
            return FailurePattern(
                pattern_id="",
                task_type=task.get("type", "unknown"),
                error_type="dependency_failure",
                error_message=task.get("error_message", "")
            )
        elif "invalid" in error_msg or "format" in error_msg:
            return FailurePattern(
                pattern_id="",
                task_type=task.get("type", "unknown"),
                error_type="invalid_input",
                error_message=task.get("error_message", "")
            )
        
        return FailurePattern(
            pattern_id="",
            task_type=task.get("type", "unknown"),
            error_type="internal_error",
            error_message=task.get("error_message", "")
        )
    
    def generate_role_definition(self, pattern: FailurePattern) -> Optional[RoleDefinition]:
        error_type = pattern.error_type.lower()
        
        if "performance" in error_type:
            return RoleDefinition(
                role_id="",
                name="PerformanceWatcher",
                category=RoleCategory.ANALYSIS.value,
                description="Monitors function execution time, identifies performance bottlenecks",
                responsibilities=["run_benchmark", "analyze_performance", "detect_bottlenecks", "suggest_optimizations"],
                capabilities=["performance_analysis", "benchmark_testing", "bottleneck_detection", "optimization_suggestion"],
                input_spec={"source_code": "", "baseline": {}},
                output_spec={"execution_time": 0.0, "bottlenecks": [], "suggestions": []},
                required_skills=["profiling", "benchmarking"]
            )
        
        elif "complexity" in error_type:
            return RoleDefinition(
                role_id="",
                name="ComplexityAnalyzer",
                category=RoleCategory.ANALYSIS.value,
                description="Analyzes code complexity and identifies problematic patterns",
                responsibilities=["analyze_complexity", "detect_god_functions", "suggest_refactoring"],
                capabilities=["complexity_analysis", "code_metrics", "refactoring_suggestion"],
                input_spec={"source_code": ""},
                output_spec={"complexity_score": 0.0, "issues": [], "suggestions": []},
                required_skills=["static_analysis", "metrics"]
            )
        
        elif "security" in error_type:
            return RoleDefinition(
                role_id="",
                name="SecurityGuard",
                category=RoleCategory.VERIFICATION.value,
                description="Detects security vulnerabilities and injection risks",
                responsibilities=["scan_vulnerabilities", "detect_injection", "verify_security"],
                capabilities=["security_scan", "vulnerability_detection", "secure_coding"],
                input_spec={"source_code": ""},
                output_spec={"vulnerabilities": [], "risk_level": "", "fix_suggestions": []},
                required_skills=["security", "penetration_testing"]
            )
        
        return None
    
    def invent_roles(self, task_history: List[Dict[str, Any]]) -> List[GeneratedRole]:
        demand = self.analyze_failures(task_history)
        generated = []
        
        for pattern in demand:
            role_def = self.generate_role_definition(pattern)
            if role_def:
                generated.append(GeneratedRole(
                    role_name=role_def.name.lower().replace(" ", "_"),
                    role_id=role_def.role_id,
                    description=role_def.description,
                    capabilities=role_def.capabilities,
                    required_modules=role_def.required_skills,
                    dependent_roles=[],
                    output_spec=role_def.output_spec
                ))
        
        return generated
    
    def get_demand_stats(self) -> Dict[str, Any]:
        return self.demand_analyzer.get_failure_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_failure_patterns": len(self.demand_analyzer.analyses),
            "total_role_definitions": len(self.generated_role_definitions),
            "failure_stats": self.demand_analyzer.get_failure_stats()
        }
    
    def generate_role_proposal(self, generated_role: GeneratedRole,
                               proposer_id: str, proposer_role: AgentRole,
                               rationale: str) -> Dict[str, Any]:
        return {
            "role_name": generated_role.role_name,
            "role_id": generated_role.role_id,
            "role_definition": generated_role.to_dict(),
            "proposer_id": proposer_id,
            "proposer_role": proposer_role.value,
            "rationale": rationale,
            "generated_code": self.definition_generator.generate_role_code(generated_role)
        }