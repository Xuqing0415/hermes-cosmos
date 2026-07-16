import os
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .requirement_parser import RequirementParser, ParsedRequirement
from .spec_generator import SpecGenerator, TLAPlusSpec
from .spec_validator import SpecValidator, ValidationResult, ValidationStatus
from .deploy_generator import DeployGenerator, DeployArtifact


class PipelineStep(Enum):
    REQUIREMENT_PARSING = "requirement_parsing"
    SPEC_GENERATION = "spec_generation"
    SPEC_VALIDATION = "spec_validation"
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    DEPLOY_GENERATION = "deploy_generation"
    TEST_EXECUTION = "test_execution"
    VERIFICATION = "verification"


@dataclass
class PipelineResult:
    step: str
    status: str
    output: Any = None
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0


class IntegrationOrchestrator:
    def __init__(self, output_base_dir: str = "generated_systems"):
        self.output_base_dir = output_base_dir
        self.parser = RequirementParser()
        self.spec_generator = SpecGenerator()
        self.spec_validator = SpecValidator()
        self.deploy_generator = DeployGenerator()
        self._pipeline_results: List[PipelineResult] = []
    
    def run_pipeline(self, natural_language: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_base_dir, timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        
        steps = [
            ("解析需求", PipelineStep.REQUIREMENT_PARSING, self._parse_requirement),
            ("生成规范", PipelineStep.SPEC_GENERATION, self._generate_spec),
            ("验证规范", PipelineStep.SPEC_VALIDATION, self._validate_spec),
            ("生成部署", PipelineStep.DEPLOY_GENERATION, self._generate_deploy),
            ("执行测试", PipelineStep.TEST_EXECUTION, self._execute_tests),
        ]
        
        context = {
            "natural_language": natural_language,
            "output_dir": output_dir,
            "requirement": None,
            "spec": None,
            "validation_result": None,
            "artifacts": [],
        }
        
        for step_name, step_enum, step_func in steps:
            step_start = time.time()
            print(f"[{step_name}] 开始...")
            
            try:
                result = step_func(context)
                duration = time.time() - step_start
                
                self._pipeline_results.append(PipelineResult(
                    step=step_enum.value,
                    status="success",
                    output=result,
                    duration=duration
                ))
                
                print(f"[{step_name}] 完成 ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - step_start
                error_msg = str(e)
                
                self._pipeline_results.append(PipelineResult(
                    step=step_enum.value,
                    status="failed",
                    errors=[error_msg],
                    duration=duration
                ))
                
                print(f"[{step_name}] 失败: {error_msg}")
                break
        
        total_duration = time.time() - start_time
        
        self._save_report(context, total_duration)
        
        return {
            "status": "success" if all(r.status == "success" for r in self._pipeline_results) else "failed",
            "total_duration": total_duration,
            "steps": [r.__dict__ for r in self._pipeline_results],
            "output_dir": output_dir,
            "requirement": context.get("requirement").to_dict() if context.get("requirement") else None,
            "spec": context.get("spec").to_dict() if context.get("spec") else None,
            "artifacts_count": len(context.get("artifacts", [])),
        }
    
    def _parse_requirement(self, context: Dict[str, Any]) -> ParsedRequirement:
        requirement = self.parser.parse(context["natural_language"])
        context["requirement"] = requirement
        
        requirement_json = requirement.to_json()
        requirement_path = os.path.join(context["output_dir"], "requirement.json")
        with open(requirement_path, "w", encoding="utf-8") as f:
            f.write(requirement_json)
        
        return requirement
    
    def _generate_spec(self, context: Dict[str, Any]) -> TLAPlusSpec:
        requirement = context["requirement"]
        spec = self.spec_generator.generate(requirement)
        context["spec"] = spec
        
        spec_path = os.path.join(context["output_dir"], f"{spec.name}.tla")
        spec.save(spec_path)
        
        context["artifacts"].append({"name": f"{spec.name}.tla", "type": "tla_spec"})
        
        return spec
    
    def _validate_spec(self, context: Dict[str, Any]) -> ValidationResult:
        spec = context["spec"]
        result = self.spec_validator.validate(spec)
        context["validation_result"] = result
        
        validation_path = os.path.join(context["output_dir"], "validation_result.json")
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return result
    
    def _generate_deploy(self, context: Dict[str, Any]) -> List[DeployArtifact]:
        requirement = context["requirement"]
        artifacts = self.deploy_generator.generate_full_project(
            requirement, context["output_dir"]
        )
        context["artifacts"] = [
            {"name": a.name, "type": a.file_type} for a in artifacts
        ]
        
        return artifacts
    
    def _execute_tests(self, context: Dict[str, Any]) -> Dict[str, Any]:
        test_path = os.path.join(context["output_dir"], "test_app.py")
        
        if not os.path.exists(test_path):
            return {"status": "skipped", "reason": "Test file not found"}
        
        import subprocess
        import sys
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            test_output = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "passed": result.returncode == 0
            }
            
            test_output_path = os.path.join(context["output_dir"], "test_output.json")
            with open(test_output_path, "w", encoding="utf-8") as f:
                json.dump(test_output, f, indent=2, ensure_ascii=False)
            
            return test_output
        
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "reason": "Test execution timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _save_report(self, context: Dict[str, Any], total_duration: float):
        steps_data = []
        for r in self._pipeline_results:
            step_dict = {
                "step": r.step,
                "status": r.status,
                "errors": r.errors,
                "duration": r.duration,
            }
            if r.output is not None:
                if hasattr(r.output, 'to_dict'):
                    step_dict["output"] = r.output.to_dict()
                elif isinstance(r.output, list):
                    step_dict["output"] = [
                        item.to_dict() if hasattr(item, 'to_dict') else str(item)
                        for item in r.output
                    ]
                elif isinstance(r.output, (dict, str, int, float, bool, type(None))):
                    step_dict["output"] = r.output
                else:
                    step_dict["output"] = str(r.output)
            steps_data.append(step_dict)
        
        report = {
            "timestamp": time.time(),
            "natural_language": context["natural_language"],
            "total_duration": total_duration,
            "output_dir": context["output_dir"],
            "steps": steps_data,
            "artifacts": context.get("artifacts", []),
            "requirement": context.get("requirement").to_dict() if context.get("requirement") else None,
            "spec": context.get("spec").to_dict() if context.get("spec") else None,
            "validation_result": context.get("validation_result").to_dict() if context.get("validation_result") else None,
        }
        
        report_path = os.path.join(context["output_dir"], "build_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def run_counter_example(self) -> Dict[str, Any]:
        requirement = "实现 HTTP 计数器服务，两个操作：POST /inc 增加计数；GET /get 返回当前值。要求并发调用不会丢失更新。"
        return self.run_pipeline(requirement)
    
    def run_stack_example(self) -> Dict[str, Any]:
        requirement = "实现一个线程安全的栈服务，支持 push、pop、peek 操作。要求保持后进先出顺序。"
        return self.run_pipeline(requirement)
    
    def run_queue_example(self) -> Dict[str, Any]:
        requirement = "实现一个消息队列服务，支持 enqueue 和 dequeue 操作。要求保持先进先出顺序，支持并发访问。"
        return self.run_pipeline(requirement)