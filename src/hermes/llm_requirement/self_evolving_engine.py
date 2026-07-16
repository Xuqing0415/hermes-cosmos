import os
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .requirement_parser import RequirementParser, ParsedRequirement
from .spec_generator import SpecGenerator, TLAPlusSpec
from .tlc_validator import TLCValidator, TLCResult, TLCStatus
from .counterexample_analyzer import CounterexampleAnalyzer, AnalysisResult
from .requirement_refiner import RequirementRefiner, RefinementResult


class EvolutionStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    TLC_UNAVAILABLE = "tlc_unavailable"


@dataclass
class EvolutionStep:
    iteration: int
    timestamp: float
    requirement_name: str
    spec_name: str
    tlc_status: str
    violation_type: Optional[str] = None
    changes_applied: int = 0
    changes: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "requirement_name": self.requirement_name,
            "spec_name": self.spec_name,
            "tlc_status": self.tlc_status,
            "violation_type": self.violation_type,
            "changes_applied": self.changes_applied,
            "changes": self.changes,
            "duration": self.duration
        }


@dataclass
class EvolutionResult:
    status: EvolutionStatus
    final_requirement: ParsedRequirement
    final_spec: TLAPlusSpec
    iterations: int
    total_duration: float
    steps: List[EvolutionStep] = field(default_factory=list)
    message: str = ""
    output_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "final_requirement": self.final_requirement.to_dict(),
            "final_spec": self.final_spec.to_dict(),
            "iterations": self.iterations,
            "total_duration": self.total_duration,
            "steps": [s.to_dict() for s in self.steps],
            "message": self.message,
            "output_dir": self.output_dir
        }


class SelfEvolvingEngine:
    def __init__(self, output_base_dir: str = "generated_systems",
                 tlc_path: str = "tlc",
                 java_path: str = "java",
                 tla2tools_jar: Optional[str] = None):
        self.output_base_dir = output_base_dir
        self.parser = RequirementParser()
        self.spec_generator = SpecGenerator()
        self.tlc_validator = TLCValidator(tlc_path, java_path, tla2tools_jar)
        self.analyzer = CounterexampleAnalyzer()
        self.refiner = RequirementRefiner()
    
    def run(self, natural_language: str, 
            max_iterations: int = 5,
            output_dir: Optional[str] = None) -> EvolutionResult:
        start_time = time.time()
        
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_base_dir, f"self_evolve_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        requirement = self.parser.parse(natural_language)
        steps = []
        last_spec = None
        
        if not self.tlc_validator.is_tlc_available():
            final_spec = self.spec_generator.generate(requirement)
            final_spec.save(os.path.join(output_dir, f"{final_spec.name}.tla"))
            
            return EvolutionResult(
                status=EvolutionStatus.TLC_UNAVAILABLE,
                final_requirement=requirement,
                final_spec=final_spec,
                iterations=0,
                total_duration=time.time() - start_time,
                steps=steps,
                message="TLC model checker not available. Generated specification without verification.",
                output_dir=output_dir
            )
        
        for iteration in range(1, max_iterations + 1):
            step_start = time.time()
            print(f"\n[迭代 {iteration}/{max_iterations}] 开始...")
            
            spec = self.spec_generator.generate(requirement)
            last_spec = spec
            spec_path = os.path.join(output_dir, f"{spec.name}_v{iteration}.tla")
            spec.save(spec_path)
            
            tlc_result = self.tlc_validator.validate(spec_path)
            
            if tlc_result.status == TLCStatus.SUCCESS:
                duration = time.time() - step_start
                steps.append(EvolutionStep(
                    iteration=iteration,
                    timestamp=time.time(),
                    requirement_name=requirement.name,
                    spec_name=spec.name,
                    tlc_status="success",
                    changes_applied=0,
                    duration=duration
                ))
                
                total_duration = time.time() - start_time
                print(f"[迭代 {iteration}] 验证通过!")
                
                self._save_evolution_report(output_dir, requirement, spec, steps, 
                                           EvolutionStatus.SUCCESS,
                                           f"Verified after {iteration} iterations",
                                           total_duration)
                
                return EvolutionResult(
                    status=EvolutionStatus.SUCCESS,
                    final_requirement=requirement,
                    final_spec=spec,
                    iterations=iteration,
                    total_duration=total_duration,
                    steps=steps,
                    message=f"Verified after {iteration} iterations",
                    output_dir=output_dir
                )
            
            if tlc_result.status == TLCStatus.FAILURE and tlc_result.counterexample:
                print(f"[迭代 {iteration}] 验证失败 - 发现反例")
                
                analysis = self.analyzer.analyze(tlc_result.counterexample)
                refinement = self.refiner.refine(requirement, analysis)
                
                requirement = refinement.refined_requirement
                
                duration = time.time() - step_start
                steps.append(EvolutionStep(
                    iteration=iteration,
                    timestamp=time.time(),
                    requirement_name=requirement.name,
                    spec_name=spec.name,
                    tlc_status="failure",
                    violation_type=analysis.violation_type.value,
                    changes_applied=len([c for c in refinement.changes if c.applied]),
                    changes=[c.to_dict() for c in refinement.changes],
                    duration=duration
                ))
                
                print(f"[迭代 {iteration}] 修正需求: {len(refinement.changes)} 处更改")
                for change in refinement.changes:
                    if change.applied:
                        print(f"  - {change.description}")
                
                if iteration == max_iterations:
                    total_duration = time.time() - start_time
                    print(f"[迭代 {iteration}] 达到最大迭代次数，退出")
                    
                    self._save_evolution_report(output_dir, requirement, spec, steps,
                                               EvolutionStatus.PARTIAL_SUCCESS,
                                               f"Partial success after {iteration} iterations",
                                               total_duration)
                    
                    return EvolutionResult(
                        status=EvolutionStatus.PARTIAL_SUCCESS,
                        final_requirement=requirement,
                        final_spec=spec,
                        iterations=iteration,
                        total_duration=total_duration,
                        steps=steps,
                        message=f"Partial success after {iteration} iterations",
                        output_dir=output_dir
                    )
            
            else:
                duration = time.time() - step_start
                steps.append(EvolutionStep(
                    iteration=iteration,
                    timestamp=time.time(),
                    requirement_name=requirement.name,
                    spec_name=spec.name,
                    tlc_status=str(tlc_result.status.value),
                    duration=duration
                ))
                
                total_duration = time.time() - start_time
                print(f"[迭代 {iteration}] 验证错误: {tlc_result.error_message}")
                
                self._save_evolution_report(output_dir, requirement, spec, steps,
                                           EvolutionStatus.FAILED,
                                           f"Failed: {tlc_result.error_message}",
                                           total_duration)
                
                return EvolutionResult(
                    status=EvolutionStatus.FAILED,
                    final_requirement=requirement,
                    final_spec=spec if last_spec else spec,
                    iterations=iteration,
                    total_duration=total_duration,
                    steps=steps,
                    message=f"Failed: {tlc_result.error_message}",
                    output_dir=output_dir
                )
        
        total_duration = time.time() - start_time
        
        return EvolutionResult(
            status=EvolutionStatus.FAILED,
            final_requirement=requirement,
            final_spec=last_spec if last_spec else self.spec_generator.generate(requirement),
            iterations=max_iterations,
            total_duration=total_duration,
            steps=steps,
            message="Failed to verify after max iterations",
            output_dir=output_dir
        )
    
    def _save_evolution_report(self, output_dir: str, requirement: ParsedRequirement,
                               spec: TLAPlusSpec, steps: List[EvolutionStep],
                               status: EvolutionStatus, message: str,
                               total_duration: float):
        report = {
            "timestamp": time.time(),
            "status": status.value,
            "message": message,
            "total_duration": total_duration,
            "iterations": len(steps),
            "final_requirement": requirement.to_dict(),
            "final_spec": spec.to_dict(),
            "evolution_steps": [s.to_dict() for s in steps]
        }
        
        report_path = os.path.join(output_dir, "evolution_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def run_counter_example(self) -> EvolutionResult:
        requirement = "实现 HTTP 计数器服务，支持 POST /inc 增加计数，POST /dec 减少计数，GET /get 返回当前值。要求并发调用不会丢失更新。"
        return self.run(requirement)
    
    def run_with_mock_counterexample(self, natural_language: str, 
                                     mock_violation: str = "TypeOK") -> EvolutionResult:
        from .tlc_validator import TLCCounterexample, TLCState
        
        start_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_base_dir, f"self_evolve_mock_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        requirement = self.parser.parse(natural_language)
        steps = []
        
        for iteration in range(1, 4):
            step_start = time.time()
            print(f"\n[模拟迭代 {iteration}/3] 开始...")
            
            spec = self.spec_generator.generate(requirement)
            spec_path = os.path.join(output_dir, f"{spec.name}_v{iteration}.tla")
            spec.save(spec_path)
            
            if iteration == 1:
                mock_counterexample = TLCCounterexample(
                    violating_property=mock_violation,
                    states=[
                        TLCState(step=0, action="Init", variables={"value": "0"}),
                        TLCState(step=1, action="Decrement", variables={"value": "-1"}),
                    ],
                    message=f"Mock violation: {mock_violation} violated"
                )
                
                analysis = self.analyzer.analyze(mock_counterexample)
                refinement = self.refiner.refine(requirement, analysis)
                requirement = refinement.refined_requirement
                
                duration = time.time() - step_start
                steps.append(EvolutionStep(
                    iteration=iteration,
                    timestamp=time.time(),
                    requirement_name=requirement.name,
                    spec_name=spec.name,
                    tlc_status="failure",
                    violation_type=analysis.violation_type.value,
                    changes_applied=len([c for c in refinement.changes if c.applied]),
                    changes=[c.to_dict() for c in refinement.changes],
                    duration=duration
                ))
                
                print(f"[模拟迭代 {iteration}] 发现反例，分析并修正")
                for change in refinement.changes:
                    if change.applied:
                        print(f"  - {change.description}")
            
            elif iteration == 2:
                mock_counterexample = TLCCounterexample(
                    violating_property="NoLostUpdates",
                    states=[
                        TLCState(step=0, action="Init", variables={"value": "0"}),
                        TLCState(step=1, action="Increment", variables={"value": "2"}),
                    ],
                    message="Mock violation: NoLostUpdates violated"
                )
                
                analysis = self.analyzer.analyze(mock_counterexample)
                refinement = self.refiner.refine(requirement, analysis)
                requirement = refinement.refined_requirement
                
                duration = time.time() - step_start
                steps.append(EvolutionStep(
                    iteration=iteration,
                    timestamp=time.time(),
                    requirement_name=requirement.name,
                    spec_name=spec.name,
                    tlc_status="failure",
                    violation_type=analysis.violation_type.value,
                    changes_applied=len([c for c in refinement.changes if c.applied]),
                    changes=[c.to_dict() for c in refinement.changes],
                    duration=duration
                ))
                
                print(f"[模拟迭代 {iteration}] 发现反例，分析并修正")
                for change in refinement.changes:
                    if change.applied:
                        print(f"  - {change.description}")
            
            else:
                duration = time.time() - step_start
                steps.append(EvolutionStep(
                    iteration=iteration,
                    timestamp=time.time(),
                    requirement_name=requirement.name,
                    spec_name=spec.name,
                    tlc_status="success",
                    changes_applied=0,
                    duration=duration
                ))
                
                print(f"[模拟迭代 {iteration}] 验证通过!")
                break
        
        total_duration = time.time() - start_time
        
        self._save_evolution_report(output_dir, requirement, spec, steps,
                                   EvolutionStatus.SUCCESS,
                                   f"Mock evolution completed after {len(steps)} iterations",
                                   total_duration)
        
        return EvolutionResult(
            status=EvolutionStatus.SUCCESS,
            final_requirement=requirement,
            final_spec=spec,
            iterations=len(steps),
            total_duration=total_duration,
            steps=steps,
            message=f"Mock evolution completed after {len(steps)} iterations",
            output_dir=output_dir
        )