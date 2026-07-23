import time
import os
import json
from typing import List, Dict, Optional

from .types import (
    SmellInstance,
    SmellType,
    SmellSeverity,
    RefactorPlan,
    RefactorResult,
    HealthReport,
    RefactorHistoryEntry
)
from .arch_analyzer import ArchAnalyzer
from .smell_detector import SmellDetector
from .refactor_plan_generator import RefactorPlanGenerator
from .behavior_verifier import BehaviorVerifier
from .refactor_rollback import RefactorRollback


class SelfRefactorEngine:
    def __init__(self, source_dir: str = "src/hermes", auto_apply: bool = False):
        self.source_dir = source_dir
        self.auto_apply = auto_apply
        
        self.analyzer = ArchAnalyzer(source_dir)
        self.detector = SmellDetector(self.analyzer)
        self.plan_generator = RefactorPlanGenerator(self.analyzer)
        self.verifier = BehaviorVerifier()
        self.rollback = RefactorRollback()
        
        self.last_health_report: Optional[HealthReport] = None

    def run(self, max_plans: int = 3) -> List[RefactorResult]:
        smells = self.detect_smells()
        plans = self.generate_plans(smells)
        results = []
        
        for plan in plans[:max_plans]:
            if plan.risk_level == "high" and not self.auto_apply:
                continue
            
            result = self.execute_plan(plan)
            results.append(result)
            
            if not result.success:
                break
        
        return results

    def detect_smells(self) -> List[SmellInstance]:
        return self.detector.detect_all()

    def generate_plans(self, smells: List[SmellInstance]) -> List[RefactorPlan]:
        return self.plan_generator.generate_plans(smells)

    def execute_plan(self, plan: RefactorPlan) -> RefactorResult:
        start_time = time.time()
        applied_actions = []
        failed_actions = []
        
        self.rollback.create_backup(plan)
        
        for action in plan.actions:
            try:
                self._apply_action(action)
                applied_actions.append(action.description)
            except Exception as e:
                failed_actions.append(f"{action.description}: {str(e)}")
        
        if failed_actions:
            self.rollback.rollback(plan.plan_id)
            execution_time = time.time() - start_time
            return RefactorResult(
                plan_id=plan.plan_id,
                success=False,
                applied_actions=applied_actions,
                failed_actions=failed_actions,
                verification_passed=False,
                rollback_required=True,
                execution_time=execution_time
            )
        
        verification_passed, verification_details = self.verifier.verify_equivalence(plan)
        
        if not verification_passed:
            self.rollback.rollback(plan.plan_id)
            execution_time = time.time() - start_time
            
            self.rollback._add_history_entry(
                plan_id=plan.plan_id,
                smell_type=plan.smell.smell_type.value,
                actions_taken=applied_actions,
                success=False,
                verification_result=json.dumps(verification_details),
                execution_time=execution_time,
                rollback_used=True,
                notes="Verification failed"
            )
            
            return RefactorResult(
                plan_id=plan.plan_id,
                success=False,
                applied_actions=applied_actions,
                failed_actions=[],
                verification_passed=False,
                rollback_required=True,
                execution_time=execution_time
            )
        
        execution_time = time.time() - start_time
        
        self.rollback._add_history_entry(
            plan_id=plan.plan_id,
            smell_type=plan.smell.smell_type.value,
            actions_taken=applied_actions,
            success=True,
            verification_result=json.dumps(verification_details),
            execution_time=execution_time,
            rollback_used=False
        )
        
        return RefactorResult(
            plan_id=plan.plan_id,
            success=True,
            applied_actions=applied_actions,
            failed_actions=[],
            verification_passed=True,
            rollback_required=False,
            execution_time=execution_time
        )

    def _apply_action(self, action):
        pass

    def generate_health_report(self) -> HealthReport:
        smells = self.detect_smells()
        module_metrics = self.analyzer.analyze_all()
        
        smells_by_type = {}
        smells_by_severity = {}
        
        for smell in SmellType:
            smells_by_type[smell] = 0
        for severity in SmellSeverity:
            smells_by_severity[severity] = 0
        
        for smell in smells:
            smells_by_type[smell.smell_type] += 1
            smells_by_severity[smell.severity] += 1
        
        total_smells = len(smells)
        
        complexity_score = 0
        coupling_score = 0
        maintainability_score = 0
        count = len(module_metrics)
        
        if count > 0:
            complexity_score = max(0, min(100, sum(100 - m.avg_complexity * 10 for m in module_metrics) / count))
            coupling_score = max(0, min(100, sum((1 - m.coupling_score) * 100 for m in module_metrics) / count))
            maintainability_score = max(0, min(100, sum(
                sum(f.maintainability_index for f in m.file_metrics) / len(m.file_metrics)
                if m.file_metrics else 0
                for m in module_metrics
            ) / count))
        
        health_score = max(0.0, min(1.0, (complexity_score * 0.3 + coupling_score * 0.3 + maintainability_score * 0.4) / 100))
        
        recommendations = self._generate_recommendations(smells)
        
        report = HealthReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_smells=total_smells,
            smells_by_type=smells_by_type,
            smells_by_severity=smells_by_severity,
            module_metrics=module_metrics,
            overall_health_score=health_score,
            recommendations=recommendations
        )
        
        self.last_health_report = report
        return report

    def _generate_recommendations(self, smells: List[SmellInstance]) -> List[str]:
        recommendations = []
        
        top_smells = self.detector.get_top_smells(5)
        
        if any(s.smell_type == SmellType.CYCLE_DEPENDENCY for s in top_smells):
            recommendations.append("Break circular dependencies by extracting shared interfaces or restructuring module imports")
        
        if any(s.smell_type == SmellType.DUPLICATE_CODE for s in top_smells):
            recommendations.append("Extract duplicate code into shared utility functions or base classes")
        
        if any(s.smell_type == SmellType.GOD_MODULE for s in top_smells):
            recommendations.append("Split god module into smaller, single-responsibility modules")
        
        if any(s.smell_type == SmellType.LONG_FUNCTION for s in top_smells):
            recommendations.append("Refactor long functions by extracting helper functions for each logical block")
        
        return recommendations

    def get_top_smells(self, n: int = 10) -> List[SmellInstance]:
        return self.detector.get_top_smells(n)

    def get_refactor_history(self) -> List[RefactorHistoryEntry]:
        return self.rollback.get_history()

    def print_health_report(self):
        if not self.last_health_report:
            self.generate_health_report()
        
        report = self.last_health_report
        
        print(f"\n{'='*60}")
        print(f"AutoTestGen ")
        print(f": {report.timestamp}")
        print(f"{'='*60}")
        
        print(f"\n: {report.overall_health_score:.2%}")
        
        print(f"\n:")
        print(f"  : {report.total_smells}")
        
        print(f"\n:")
        for smell_type, count in report.smells_by_type.items():
            if count > 0:
                print(f"  {smell_type.value}: {count}")
        
        print(f"\n:")
        for severity, count in report.smells_by_severity.items():
            if count > 0:
                marker = "CRITICAL" if severity == SmellSeverity.CRITICAL else \
                        "HIGH" if severity == SmellSeverity.HIGH else \
                        "MEDIUM" if severity == SmellSeverity.MEDIUM else "LOW"
                print(f"  [{marker}] {severity.value}: {count}")
        
        print(f"\n:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\n{'='*60}")

    def save_health_report(self, file_path: str = "health_report.json"):
        if not self.last_health_report:
            self.generate_health_report()
        
        report = self.last_health_report
        
        data = {
            'timestamp': report.timestamp,
            'overall_health_score': report.overall_health_score,
            'total_smells': report.total_smells,
            'smells_by_type': {k.value: v for k, v in report.smells_by_type.items()},
            'smells_by_severity': {k.value: v for k, v in report.smells_by_severity.items()},
            'recommendations': report.recommendations,
            'module_metrics': []
        }
        
        for mm in report.module_metrics:
            module_data = {
                'module_name': mm.module_name,
                'total_loc': mm.total_loc,
                'total_functions': mm.total_functions,
                'total_classes': mm.total_classes,
                'avg_complexity': mm.avg_complexity,
                'coupling_score': mm.coupling_score,
                'cohesion_score': mm.cohesion_score
            }
            data['module_metrics'].append(module_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def rollback_plan(self, plan_id: str) -> bool:
        return self.rollback.rollback(plan_id)

    def set_auto_apply(self, auto_apply: bool):
        self.auto_apply = auto_apply