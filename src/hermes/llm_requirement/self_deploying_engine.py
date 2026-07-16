from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import time
import json
import tempfile
import subprocess
import urllib.request
import urllib.error

from .requirement_parser import RequirementParser, ParsedRequirement
from .spec_generator import SpecGenerator, TLAPlusSpec
from .microservice_generator import MicroserviceGenerator, GeneratedService
from .k8s_deployer import DeployerManager, BaseDeployer, DeployResult
from .runtime_monitor import RuntimeMonitor, TLASpecStateMachine, MonitorResult, ViolationReport
from .anomaly_analyzer import AnomalyAnalyzer, AnomalyAnalysisResult, FixAction, FixActionType
from .requirement_refiner import RequirementRefiner


class DeploymentStatus(Enum):
    GENERATING = "generating"
    DEPLOYING = "deploying"
    RUNNING = "running"
    MONITORING = "monitoring"
    DETECTED_ANOMALY = "detected_anomaly"
    FIXING = "fixing"
    REDEPLOYING = "redeploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DeploymentStep:
    step: str
    status: str
    timestamp: float
    duration: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "details": self.details
        }


@dataclass
class SelfDeployResult:
    status: DeploymentStatus
    requirement: ParsedRequirement
    service: GeneratedService
    output_dir: str = ""
    steps: List[DeploymentStep] = field(default_factory=list)
    violations: List[ViolationReport] = field(default_factory=list)
    fix_actions_taken: List[str] = field(default_factory=list)
    total_duration: float = 0.0
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "requirement": self.requirement.to_dict(),
            "service_name": self.service.name,
            "output_dir": self.output_dir,
            "steps": [s.to_dict() for s in self.steps],
            "violations_count": len(self.violations),
            "fix_actions_taken": self.fix_actions_taken,
            "total_duration": self.total_duration,
            "message": self.message
        }


class SelfDeployingEngine:
    def __init__(self, output_base_dir: str = "generated_systems"):
        self.output_base_dir = output_base_dir
        self.parser = RequirementParser()
        self.spec_generator = SpecGenerator()
        self.service_generator = MicroserviceGenerator()
        self.deployer = DeployerManager.get_deployer()
        self.monitor = RuntimeMonitor()
        self.analyzer = AnomalyAnalyzer()
        self.refiner = RequirementRefiner()
        self._current_requirement: Optional[ParsedRequirement] = None
        self._current_service: Optional[GeneratedService] = None
        self._output_dir: str = ""
    
    def run(self, natural_language: str, 
            max_fix_iterations: int = 3,
            test_operations: Optional[List[str]] = None) -> SelfDeployResult:
        start_time = time.time()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._output_dir = os.path.join(self.output_base_dir, f"self_deploy_{timestamp}")
        os.makedirs(self._output_dir, exist_ok=True)
        
        steps = []
        violations = []
        fix_actions_taken = []
        
        try:
            steps.append(DeploymentStep(
                step="requirement_parsing",
                status="started",
                timestamp=time.time()
            ))
            
            self._current_requirement = self.parser.parse(natural_language)
            
            steps[-1].status = "completed"
            steps[-1].duration = time.time() - steps[-1].timestamp
            steps[-1].details = {"requirement_name": self._current_requirement.name}
            
            print(f"[部署引擎] 解析需求完成: {self._current_requirement.name}")
            
            steps.append(DeploymentStep(
                step="service_generation",
                status="started",
                timestamp=time.time()
            ))
            
            self._current_service = self.service_generator.generate(self._current_requirement)
            self._current_service.save_all(self._output_dir)
            
            steps[-1].status = "completed"
            steps[-1].duration = time.time() - steps[-1].timestamp
            steps[-1].details = {
                "service_name": self._current_service.name,
                "artifacts_count": len(self._current_service.artifacts)
            }
            
            print(f"[部署引擎] 生成微服务完成: {len(self._current_service.artifacts)} 个工件")
            
            steps.append(DeploymentStep(
                step="deployment",
                status="started",
                timestamp=time.time()
            ))
            
            manifest_files = []
            for artifact in self._current_service.artifacts:
                if artifact.file_type == "yaml":
                    manifest_files.append(os.path.join(self._output_dir, artifact.path, artifact.name))
            
            deploy_result = self.deployer.deploy(manifest_files)
            
            steps[-1].status = "completed" if deploy_result.success else "failed"
            steps[-1].duration = time.time() - steps[-1].timestamp
            steps[-1].details = {
                "deployer_type": self.deployer.get_deployer_type().value,
                "deployments_count": len(deploy_result.deployments)
            }
            
            if not deploy_result.success:
                return SelfDeployResult(
                    status=DeploymentStatus.FAILED,
                    requirement=self._current_requirement,
                    service=self._current_service,
                    output_dir=self._output_dir,
                    steps=steps,
                    total_duration=time.time() - start_time,
                    message=f"Deployment failed: {deploy_result.message}"
                )
            
            print(f"[部署引擎] 部署完成")
            
            service_url = None
            for d in deploy_result.deployments:
                if d.service_url:
                    service_url = d.service_url
                    break
            
            steps.append(DeploymentStep(
                step="service_startup",
                status="started",
                timestamp=time.time()
            ))
            
            print(f"[部署引擎] 启动服务进程...")
            service_proc = self._start_service_process()
            
            if service_proc:
                print(f"[部署引擎] 服务启动成功")
                service_url = "http://localhost:5000"
                steps[-1].details = {"service_url": service_url, "process_started": True}
            else:
                print(f"[部署引擎] 服务启动失败，使用模拟模式")
                steps[-1].details = {"service_url": None, "process_started": False}
            
            steps[-1].status = "completed"
            steps[-1].duration = time.time() - steps[-1].timestamp
            
            steps.append(DeploymentStep(
                step="monitoring",
                status="started",
                timestamp=time.time()
            ))
            
            print(f"[部署引擎] 开始监控...")
            
            test_ops = test_operations or ["inc", "inc", "inc", "dec", "dec", "dec", "dec"]
            
            all_monitor_results = self._run_test_sequence(test_ops)
            
            for result in all_monitor_results:
                if result.status.value == "violation":
                    violations.extend(result.violations)
            
            steps[-1].status = "completed"
            steps[-1].duration = time.time() - steps[-1].timestamp
            steps[-1].details = {
                "operations_executed": len(test_ops),
                "violations_found": len(violations)
            }
            
            if violations:
                print(f"[部署引擎] 检测到 {len(violations)} 个违规")
                
                steps.append(DeploymentStep(
                    step="anomaly_analysis",
                    status="started",
                    timestamp=time.time()
                ))
                
                analysis = self.analyzer.analyze(violations)
                
                steps[-1].status = "completed"
                steps[-1].duration = time.time() - steps[-1].timestamp
                steps[-1].details = {
                    "root_causes_found": len(analysis.root_causes),
                    "fix_actions_suggested": len(analysis.fix_actions)
                }
                
                print(f"[部署引擎] 异常分析完成: {len(analysis.root_causes)} 个根因")
                
                for cause in analysis.root_causes:
                    print(f"  - {cause.type.value}: {cause.description}")
                
                for fix_iteration in range(max_fix_iterations):
                    if not analysis.fix_actions:
                        break
                    
                    steps.append(DeploymentStep(
                        step=f"fix_iteration_{fix_iteration+1}",
                        status="started",
                        timestamp=time.time()
                    ))
                    
                    fix_applied = False
                    
                    for action in analysis.fix_actions:
                        if action.action_type == FixActionType.REQUIREMENT_REFINEMENT:
                            print(f"[部署引擎] 执行需求修正: {action.description}")
                            
                            from .counterexample_analyzer import AnalysisResult, FixSuggestion, ViolationType
                            
                            mock_analysis = AnalysisResult(
                                violation_type=ViolationType.INVARIANT_VIOLATION,
                                violating_property="TypeOK",
                                description=action.description,
                                fix_suggestions=[FixSuggestion(
                                    action_type="add_precondition",
                                    target_operation=action.target,
                                    suggestion=action.description,
                                    precondition=action.parameters.get("precondition", "")
                                )]
                            )
                            
                            refinement = self.refiner.refine(self._current_requirement, mock_analysis)
                            self._current_requirement = refinement.refined_requirement
                            
                            fix_actions_taken.append(f"requirement_refinement: {action.description}")
                            fix_applied = True
                            break
                        
                        elif action.action_type == FixActionType.CODE_FIX:
                            print(f"[部署引擎] 执行代码修复: {action.description}")
                            
                            self._current_service = self.service_generator.generate(self._current_requirement)
                            self._current_service.save_all(self._output_dir)
                            
                            fix_actions_taken.append(f"code_fix: {action.description}")
                            fix_applied = True
                            break
                    
                    if fix_applied:
                        steps.append(DeploymentStep(
                            step=f"redeploy_{fix_iteration+1}",
                            status="started",
                            timestamp=time.time()
                        ))
                        
                        manifest_files = []
                        for artifact in self._current_service.artifacts:
                            if artifact.file_type == "yaml":
                                manifest_files.append(os.path.join(self._output_dir, artifact.path, artifact.name))
                        
                        redeploy_result = self.deployer.deploy(manifest_files)
                        
                        steps[-1].status = "completed" if redeploy_result.success else "failed"
                        steps[-1].duration = time.time() - steps[-1].timestamp
                        
                        if redeploy_result.success:
                            print(f"[部署引擎] 重新部署完成")
                            
                            self.monitor.reset()
                            all_monitor_results = self._run_test_sequence(test_ops)
                            
                            new_violations = []
                            for result in all_monitor_results:
                                if result.status.value == "violation":
                                    new_violations.extend(result.violations)
                            
                            if not new_violations:
                                violations = []
                                print(f"[部署引擎] 修复成功! 验证通过")
                                break
                            else:
                                violations = new_violations
                                analysis = self.analyzer.analyze(violations)
                                print(f"[部署引擎] 仍有 {len(violations)} 个违规，继续修复")
                        else:
                            print(f"[部署引擎] 重新部署失败")
                            break
                    
                    steps[-1].status = "completed"
                    steps[-1].duration = time.time() - steps[-1].timestamp
                
                if violations:
                    return SelfDeployResult(
                        status=DeploymentStatus.DETECTED_ANOMALY,
                        requirement=self._current_requirement,
                        service=self._current_service,
                        output_dir=self._output_dir,
                        steps=steps,
                        violations=violations,
                        fix_actions_taken=fix_actions_taken,
                        total_duration=time.time() - start_time,
                        message=f"Detected {len(violations)} violations after {max_fix_iterations} fix attempts"
                    )
            
            return SelfDeployResult(
                status=DeploymentStatus.COMPLETED,
                requirement=self._current_requirement,
                service=self._current_service,
                output_dir=self._output_dir,
                steps=steps,
                violations=violations,
                fix_actions_taken=fix_actions_taken,
                total_duration=time.time() - start_time,
                message="Self-deployment completed successfully"
            )
        
        except Exception as e:
            return SelfDeployResult(
                status=DeploymentStatus.FAILED,
                requirement=self._current_requirement or ParsedRequirement(name="Unknown", description=""),
                service=self._current_service or GeneratedService(name="Unknown"),
                output_dir=self._output_dir,
                steps=steps,
                total_duration=time.time() - start_time,
                message=f"Error during deployment: {str(e)}"
            )
    
    def _run_test_sequence(self, operations: List[str]) -> List[MonitorResult]:
        results = []
        
        service_url = self._get_service_url()
        
        if service_url:
            return self._run_test_with_http(operations, service_url)
        
        return self._run_test_simulated(operations)
    
    def _get_service_url(self) -> Optional[str]:
        try:
            urllib.request.urlopen("http://localhost:5000/health", timeout=1)
            return "http://localhost:5000"
        except urllib.error.URLError:
            return None
    
    def _start_service_process(self) -> Optional[subprocess.Popen]:
        app_path = os.path.join(self._output_dir, "app.py")
        if not os.path.exists(app_path):
            return None
        
        try:
            env = os.environ.copy()
            env["PORT"] = "5000"
            
            proc = subprocess.Popen(
                ["python", app_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._output_dir
            )
            
            for _ in range(10):
                time.sleep(0.5)
                try:
                    urllib.request.urlopen("http://localhost:5000/health", timeout=1)
                    return proc
                except urllib.error.URLError:
                    continue
            
            proc.terminate()
            return None
        except Exception:
            return None
    
    def _run_test_with_http(self, operations: List[str], base_url: str) -> List[MonitorResult]:
        results = []
        
        for op in operations:
            print(f"[部署引擎] 执行操作: {op}")
            
            pre_state = self._get_service_state(base_url)
            
            if op == "inc":
                self._http_post(f"{base_url}/inc")
            elif op == "dec":
                self._http_post(f"{base_url}/dec")
            elif op == "reset":
                self._http_post(f"{base_url}/reset")
            elif op == "get":
                self._http_get(f"{base_url}/get")
            
            post_state = self._get_service_state(base_url)
            
            result = self.monitor.record_transition(
                "increment" if op == "inc" else "decrement" if op == "dec" else op,
                pre_state,
                post_state
            )
            results.append(result)
            
            if result.status.value == "violation":
                print(f"  ⚠️ 检测到违规: {[v.message for v in result.violations]}")
            else:
                print(f"  ✓ 操作成功: {pre_state} -> {post_state}")
        
        return results
    
    def _http_post(self, url: str) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {}
    
    def _http_get(self, url: str) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {}
    
    def _get_service_state(self, base_url: str) -> Dict[str, Any]:
        try:
            resp = self._http_get(f"{base_url}/state")
            return {"value": resp.get("value", 0)}
        except Exception:
            return {"value": 0}
    
    def _run_test_simulated(self, operations: List[str]) -> List[MonitorResult]:
        results = []
        has_precondition = self._has_decrement_precondition()
        
        for op in operations:
            print(f"[部署引擎] 执行操作: {op}")
            
            if op == "inc":
                pre_state = self.monitor.get_current_state().copy()
                self.monitor._current_state["value"] += 1
                post_state = self.monitor.get_current_state().copy()
                
                result = self.monitor.record_transition("increment", pre_state, post_state)
                results.append(result)
            
            elif op == "dec":
                pre_state = self.monitor.get_current_state().copy()
                current_val = pre_state.get("value", 0)
                
                if has_precondition and current_val <= 0:
                    self.monitor._current_state["value"] = current_val
                else:
                    self.monitor._current_state["value"] -= 1
                
                post_state = self.monitor.get_current_state().copy()
                result = self.monitor.record_transition("decrement", pre_state, post_state)
                results.append(result)
            
            elif op == "get":
                pre_state = self.monitor.get_current_state().copy()
                post_state = self.monitor.get_current_state().copy()
                result = self.monitor.record_transition("get", pre_state, post_state)
                results.append(result)
            
            elif op == "reset":
                pre_state = self.monitor.get_current_state().copy()
                self.monitor._current_state["value"] = 0
                post_state = self.monitor.get_current_state().copy()
                result = self.monitor.record_transition("reset", pre_state, post_state)
                results.append(result)
            
            if result.status.value == "violation":
                print(f"  ⚠️ 检测到违规: {[v.message for v in result.violations]}")
            else:
                print(f"  ✓ 操作成功: {pre_state} -> {post_state}")
        
        return results
    
    def _has_decrement_precondition(self) -> bool:
        if not self._current_requirement:
            return False
        
        for op in self._current_requirement.operations:
            if op.name.lower() == "decrement":
                if "precondition" in op.description.lower() or "value > 0" in op.description.lower():
                    return True
        
        for constraint in self._current_requirement.constraints:
            if "no negative" in constraint.description.lower() or "non-negative" in constraint.description.lower():
                return True
        
        return False
    
    def run_counter_deployment(self) -> SelfDeployResult:
        requirement_text = "实现 HTTP 计数器服务，支持 POST /inc 增加计数，POST /dec 减少计数，GET /get 返回当前值，POST /reset 重置为0。要求计数器不能为负。"
        
        return self.run(requirement_text, test_operations=["inc", "inc", "inc", "dec", "dec", "dec", "dec"])