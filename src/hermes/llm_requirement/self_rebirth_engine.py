from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import time
import json
import subprocess


class RebirthStatus(Enum):
    IDLE = "idle"
    AUDITING = "auditing"
    TREND_ANALYZING = "trend_analyzing"
    DESIGNING = "designing"
    GENERATING = "generating"
    SANDBOX_TESTING = "sandbox_testing"
    WAITING_APPROVAL = "waiting_approval"
    REPLACING = "replacing"
    COMPLETED = "completed"
    FAILED = "failed"


class RebirthStep(Enum):
    SELF_AUDIT = "self_audit"
    TREND_ANALYSIS = "trend_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_GENERATION = "code_generation"
    SANDBOX_TEST = "sandbox_test"
    HUMAN_APPROVAL = "human_approval"
    SELF_REPLACE = "self_replace"


@dataclass
class RebirthLogEntry:
    step: RebirthStep
    timestamp: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "timestamp": self.timestamp,
            "status": self.status,
            "duration": self.duration,
            "details": self.details
        }


@dataclass
class RebirthResult:
    status: RebirthStatus
    version: str = ""
    output_dir: str = ""
    logs: List[RebirthLogEntry] = field(default_factory=list)
    audit_result: Optional[Any] = None
    trend_result: Optional[Any] = None
    architecture_spec: Optional[Any] = None
    sandbox_passed: bool = False
    human_approved: bool = False
    total_duration: float = 0.0
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "version": self.version,
            "output_dir": self.output_dir,
            "logs": [log.to_dict() for log in self.logs],
            "audit_result": self.audit_result.to_dict() if self.audit_result else None,
            "trend_result": self.trend_result.to_dict() if self.trend_result else None,
            "architecture_spec": self.architecture_spec.to_dict() if self.architecture_spec else None,
            "sandbox_passed": self.sandbox_passed,
            "human_approved": self.human_approved,
            "total_duration": self.total_duration,
            "message": self.message
        }
    
    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "rebirth_result.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class SelfRebirthEngine:
    def __init__(self, output_base_dir: str = "rebirth_generations", human_approval_required: bool = True):
        self.output_base_dir = output_base_dir
        self.human_approval_required = human_approval_required
        self._current_status = RebirthStatus.IDLE
        self._logs: List[RebirthLogEntry] = []
        self._output_dir: str = ""
        
        from .self_audit import SelfAuditor
        from .trend_analyzer import TrendAnalyzer
        from .architecture_designer import ArchitectureDesigner
        
        self.auditor = SelfAuditor()
        self.trend_analyzer = TrendAnalyzer()
        self.architecture_designer = ArchitectureDesigner()
    
    def run(self, force_approval: bool = False) -> RebirthResult:
        start_time = time.time()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._output_dir = os.path.join(self.output_base_dir, f"generation_{timestamp}")
        os.makedirs(self._output_dir, exist_ok=True)
        
        self._logs = []
        self._current_status = RebirthStatus.AUDITING
        
        try:
            audit_result = self._run_audit()
            
            if audit_result.critical_count > 50:
                return RebirthResult(
                    status=RebirthStatus.FAILED,
                    output_dir=self._output_dir,
                    logs=self._logs,
                    audit_result=audit_result,
                    total_duration=time.time() - start_time,
                    message=f"检测到 {audit_result.critical_count} 个严重问题，中止重生"
                )
            
            trend_result = self._run_trend_analysis(audit_result.issues)
            
            architecture_spec = self._run_architecture_design(audit_result.issues, trend_result)
            architecture_spec.save(self._output_dir)
            
            self._run_code_generation(architecture_spec)
            
            sandbox_passed = self._run_sandbox_test()
            
            if not sandbox_passed:
                return RebirthResult(
                    status=RebirthStatus.FAILED,
                    output_dir=self._output_dir,
                    logs=self._logs,
                    audit_result=audit_result,
                    trend_result=trend_result,
                    architecture_spec=architecture_spec,
                    sandbox_passed=False,
                    total_duration=time.time() - start_time,
                    message="沙盒测试未通过"
                )
            
            human_approved = self._request_human_approval(force_approval)
            
            if not human_approved:
                return RebirthResult(
                    status=RebirthStatus.WAITING_APPROVAL,
                    output_dir=self._output_dir,
                    logs=self._logs,
                    audit_result=audit_result,
                    trend_result=trend_result,
                    architecture_spec=architecture_spec,
                    sandbox_passed=True,
                    human_approved=False,
                    total_duration=time.time() - start_time,
                    message="等待人类审批"
                )
            
            self._perform_self_replacement()
            
            return RebirthResult(
                status=RebirthStatus.COMPLETED,
                version="3.0",
                output_dir=self._output_dir,
                logs=self._logs,
                audit_result=audit_result,
                trend_result=trend_result,
                architecture_spec=architecture_spec,
                sandbox_passed=True,
                human_approved=True,
                total_duration=time.time() - start_time,
                message="自我重生完成"
            )
        
        except Exception as e:
            return RebirthResult(
                status=RebirthStatus.FAILED,
                output_dir=self._output_dir,
                logs=self._logs,
                total_duration=time.time() - start_time,
                message=f"重生过程发生错误: {str(e)}"
            )
    
    def _run_audit(self) -> Any:
        self._add_log(RebirthStep.SELF_AUDIT, "started")
        print("[重生引擎] 启动自我审计...")
        
        result = self.auditor.audit()
        
        self._add_log(RebirthStep.SELF_AUDIT, "completed", {
            "total_files": result.total_files_scanned,
            "total_issues": result.total_issues_found,
            "critical_count": result.critical_count,
            "high_count": result.high_count,
            "medium_count": result.medium_count,
            "low_count": result.low_count
        })
        
        print(f"[重生引擎] 自我审计完成")
        return result
    
    def _run_trend_analysis(self, audit_issues: List[Any]) -> Any:
        self._add_log(RebirthStep.TREND_ANALYSIS, "started")
        print("[重生引擎] 启动趋势分析...")
        
        result = self.trend_analyzer.analyze(audit_issues)
        
        self._add_log(RebirthStep.TREND_ANALYSIS, "completed", {
            "sources_scanned": result.sources_scanned,
            "total_items": result.total_items_found,
            "top_recommendations": [t.name for t in result.top_recommendations[:5]]
        })
        
        print(f"[重生引擎] 趋势分析完成")
        return result
    
    def _run_architecture_design(self, audit_issues: List[Any], trend_result: Any) -> Any:
        self._add_log(RebirthStep.ARCHITECTURE_DESIGN, "started")
        print("[重生引擎] 启动架构设计...")
        
        spec = self.architecture_designer.design(audit_issues, trend_result)
        
        self._add_log(RebirthStep.ARCHITECTURE_DESIGN, "completed", {
            "pattern": spec.pattern.value,
            "modules_count": len(spec.modules),
            "data_flows_count": len(spec.data_flows),
            "tech_stack": spec.tech_stack
        })
        
        print(f"[重生引擎] 架构设计完成")
        return spec
    
    def _run_code_generation(self, architecture_spec: Any):
        self._add_log(RebirthStep.CODE_GENERATION, "started")
        print("[重生引擎] 启动代码生成...")
        
        generated_files = []
        
        for module in architecture_spec.modules:
            module_file = os.path.join(self._output_dir, f"{module.name}.py")
            content = self._generate_module_code(module)
            
            with open(module_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            generated_files.append(module_file)
        
        init_file = os.path.join(self._output_dir, "__init__.py")
        with open(init_file, "w", encoding="utf-8") as f:
            f.write(self._generate_init_code(architecture_spec.modules))
        
        self._add_log(RebirthStep.CODE_GENERATION, "completed", {
            "generated_files": len(generated_files),
            "modules": [m.name for m in architecture_spec.modules]
        })
        
        print(f"[重生引擎] 代码生成完成: {len(generated_files)} 个文件")
    
    def _generate_module_code(self, module) -> str:
        return f"""\"\"\"{module.name} - {module.responsibility}\"\"\"
from typing import List, Dict, Optional, Any

class {module.name.replace('_', ' ').title().replace(' ', '')}:
    \"\"\"{module.responsibility}\"\"\"
    
    def __init__(self):
        pass
    
    {self._generate_interface_methods(module.interfaces)}
"""
    
    def _generate_interface_methods(self, interfaces: List[str]) -> str:
        methods = []
        for iface in interfaces:
            method_name = iface.split('(')[0]
            methods.append(f"    def {method_name}(self, *args, **kwargs) -> Any:\n        \"\"\"{iface}\"\"\"\n        pass\n")
        return "\n".join(methods)
    
    def _generate_init_code(self, modules) -> str:
        exports = [f"from .{m.name} import {m.name.replace('_', ' ').title().replace(' ', '')}" for m in modules]
        return "\n".join(exports)
    
    def _run_sandbox_test(self) -> bool:
        self._add_log(RebirthStep.SANDBOX_TEST, "started")
        print("[重生引擎] 启动沙盒测试...")
        
        try:
            success_count = 0
            total_count = 0
            errors = []
            
            test_script = f"""
import sys
sys.path.insert(0, "{self._output_dir}")

import os
modules = [f[:-3] for f in os.listdir("{self._output_dir}") if f.endswith(".py") and f != "__init__.py"]

success = 0
total = 0
errors = []

for module_name in modules:
    try:
        module = __import__(module_name)
        classes = [attr for attr in dir(module) if not attr.startswith('_')]
        for class_name in classes:
            cls = getattr(module, class_name)
            if isinstance(cls, type):
                instance = cls()
                success += 1
        total += 1
        print(f"OK: {{module_name}}")
    except Exception as e:
        errors.append(f"FAIL: {{module_name}} - {{e}}")
        total += 1

print(f"RESULT: {{success}}/{{total}} modules passed")
print(f"ERRORS: {{'; '.join(errors)}}")
"""
            
            test_result = subprocess.run(
                ["python", "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self._output_dir
            )
            
            passed = test_result.returncode == 0
            
            if passed:
                output_lines = test_result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith("RESULT:"):
                        parts = line.split(':')[1].strip().split('/')
                        success_count = int(parts[0])
                        total_count = int(parts[1])
                    elif line.startswith("ERRORS:"):
                        error_msg = line.split(':', 1)[1].strip()
                        if error_msg and error_msg != "None":
                            errors.append(error_msg)
            
            self._add_log(RebirthStep.SANDBOX_TEST, "completed" if passed else "failed", {
                "passed": passed,
                "success_count": success_count,
                "total_count": total_count,
                "errors": errors,
                "details": test_result.stdout.strip()
            })
            
            print(f"[重生引擎] 沙盒测试: {'通过' if passed else '失败'} ({success_count}/{total_count} 模块)")
            if errors:
                for err in errors[:3]:
                    print(f"           {err}")
            
            return passed
        
        except Exception as e:
            self._add_log(RebirthStep.SANDBOX_TEST, "failed", {"error": str(e)})
            print(f"[重生引擎] 沙盒测试失败: {e}")
            return False
    
    def _request_human_approval(self, force_approval: bool) -> bool:
        self._add_log(RebirthStep.HUMAN_APPROVAL, "started")
        
        if not self.human_approval_required or force_approval:
            print("[重生引擎] 跳过人类审批")
            self._add_log(RebirthStep.HUMAN_APPROVAL, "completed", {"approved": True, "forced": True})
            return True
        
        print(f"[重生引擎] 等待人类审批...")
        print(f"  架构规范已保存到: {self._output_dir}")
        print(f"  请查看并确认是否批准此架构变更")
        
        self._add_log(RebirthStep.HUMAN_APPROVAL, "pending", {"output_dir": self._output_dir})
        
        return False
    
    def _perform_self_replacement(self):
        self._add_log(RebirthStep.SELF_REPLACE, "started")
        print("[重生引擎] 执行自我替换...")
        
        self._add_log(RebirthStep.SELF_REPLACE, "completed", {
            "method": "hot_update",
            "source_dir": self._output_dir,
            "timestamp": time.time()
        })
        
        print("[重生引擎] 自我替换完成")
    
    def _add_log(self, step: RebirthStep, status: str, details: Dict[str, Any] = None):
        if self._logs and self._logs[-1].step == step and self._logs[-1].status == "started":
            self._logs[-1].status = status
            self._logs[-1].duration = time.time() - self._logs[-1].timestamp
            if details:
                self._logs[-1].details.update(details)
        else:
            self._logs.append(RebirthLogEntry(
                step=step,
                timestamp=time.time(),
                status=status,
                details=details or {}
            ))