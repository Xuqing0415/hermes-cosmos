import ast
import os
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .types import (
    SmellInstance,
    SmellType,
    RefactorPlan,
    RefactorAction,
    RefactorOperation,
    CodeLocation
)
from .arch_analyzer import ArchAnalyzer


class RefactorPlanGenerator:
    def __init__(self, analyzer: Optional[ArchAnalyzer] = None):
        self.analyzer = analyzer or ArchAnalyzer()

    def generate_plans(self, smells: List[SmellInstance]) -> List[RefactorPlan]:
        plans = []
        
        for smell in smells:
            plan = self._generate_plan_for_smell(smell)
            if plan:
                plans.append(plan)
        
        return sorted(plans, key=lambda p: p.smell.priority, reverse=True)

    def _generate_plan_for_smell(self, smell: SmellInstance) -> Optional[RefactorPlan]:
        plan_id = str(uuid.uuid4())[:8]
        
        if smell.smell_type == SmellType.CYCLE_DEPENDENCY:
            return self._generate_break_cycle_plan(plan_id, smell)
        elif smell.smell_type == SmellType.DUPLICATE_CODE:
            return self._generate_remove_duplication_plan(plan_id, smell)
        elif smell.smell_type == SmellType.GOD_MODULE:
            return self._generate_split_module_plan(plan_id, smell)
        elif smell.smell_type == SmellType.LONG_FUNCTION:
            return self._generate_extract_function_plan(plan_id, smell)
        elif smell.smell_type == SmellType.DATA_CLUMP:
            return self._generate_introduce_class_plan(plan_id, smell)
        elif smell.smell_type == SmellType.SHOTGUN_SURGERY:
            return self._generate_move_constant_plan(plan_id, smell)
        elif smell.smell_type == SmellType.FEATURE_ENVY:
            return self._generate_move_method_plan(plan_id, smell)
        
        return None

    def _generate_break_cycle_plan(self, plan_id: str, smell: SmellInstance) -> RefactorPlan:
        cycle = smell.metadata.get('cycle', [])
        actions = []
        
        if len(cycle) >= 2:
            middle_module = cycle[1] if len(cycle) > 2 else cycle[0]
            target_module = cycle[-2] if len(cycle) > 2 else cycle[1]
            
            actions.append(RefactorAction(
                operation=RefactorOperation.BREAK_CYCLE,
                description=f" {middle_module}  {target_module} ",
                affected_files=[self._module_to_path(m) for m in cycle]
            ))
            
            actions.append(RefactorAction(
                operation=RefactorOperation.INTRODUCE_INTERFACE,
                description=f"",
                new_name=f"{middle_module}_interface",
                affected_files=[self._module_to_path(m) for m in cycle]
            ))
        
        return RefactorPlan(
            plan_id=plan_id,
            smell=smell,
            actions=actions,
            estimated_effort=8.0,
            risk_level="high",
            expected_improvement={'coupling_reduction': 0.5},
            description=f": {' → '.join(cycle)}"
        )

    def _generate_remove_duplication_plan(self, plan_id: str, smell: SmellInstance) -> RefactorPlan:
        actions = []
        
        if smell.locations:
            primary_location = smell.locations[0]
            target_dir = os.path.dirname(primary_location.file_path)
            common_utils = os.path.join(target_dir, 'utils.py')
            
            if not os.path.exists(common_utils):
                actions.append(RefactorAction(
                    operation=RefactorOperation.EXTRACT_FUNCTION,
                    description=f" {common_utils}",
                    source_location=primary_location,
                    new_name="_extract_common_function",
                    affected_files=[loc.file_path for loc in smell.locations]
                ))
            
            for location in smell.locations[1:]:
                actions.append(RefactorAction(
                    operation=RefactorOperation.REMOVE_DUPLICATION,
                    description=f" {common_utils}  {location.file_path} ",
                    source_location=location,
                    affected_files=[location.file_path, common_utils]
                ))
        
        occurrences = smell.metadata.get('occurrences', 1)
        return RefactorPlan(
            plan_id=plan_id,
            smell=smell,
            actions=actions,
            estimated_effort=3.0 * occurrences,
            risk_level="low",
            expected_improvement={'duplication_reduction': 0.8},
            description=f""
        )

    def _generate_split_module_plan(self, plan_id: str, smell: SmellInstance) -> RefactorPlan:
        actions = []
        total_functions = smell.metadata.get('total_functions', 0)
        total_classes = smell.metadata.get('total_classes', 0)
        
        if smell.locations:
            main_file = smell.locations[0].file_path
            module_dir = os.path.dirname(main_file)
            
            if total_functions > 20:
                actions.append(RefactorAction(
                    operation=RefactorOperation.SPLIT_MODULE,
                    description=f"",
                    source_location=smell.locations[0],
                    affected_files=[loc.file_path for loc in smell.locations]
                ))
                
                for i in range((total_functions // 10) + 1):
                    if i == 0:
                        continue
                    new_module = os.path.join(module_dir, f'submodule_{i}.py')
                    actions.append(RefactorAction(
                        operation=RefactorOperation.MOVE_METHOD,
                        description=f" {new_module}",
                        new_name=f'submodule_{i}',
                        affected_files=[main_file, new_module]
                    ))
        
        return RefactorPlan(
            plan_id=plan_id,
            smell=smell,
            actions=actions,
            estimated_effort=5.0 + total_functions * 0.5,
            risk_level="medium",
            expected_improvement={'cohesion_improvement': 0.6},
            description=f""
        )

    def _generate_extract_function_plan(self, plan_id: str, smell: SmellInstance) -> RefactorPlan:
        actions = []
        
        if smell.locations:
            location = smell.locations[0]
            func_name = smell.metadata.get('function_name', 'unknown')
            complexity = smell.metadata.get('complexity', 0)
            
            num_extracts = max(1, complexity // 5)
            
            for i in range(num_extracts):
                actions.append(RefactorAction(
                    operation=RefactorOperation.EXTRACT_FUNCTION,
                    description=f" '{func_name}'  {i+1}",
                    source_location=location,
                    new_name=f"_{func_name}_part_{i+1}",
                    affected_files=[location.file_path]
                ))
            
            actions.append(RefactorAction(
                operation=RefactorOperation.EXTRACT_FUNCTION,
                description=f" '{func_name}' ",
                source_location=location,
                affected_files=[location.file_path]
            ))
        
        return RefactorPlan(
            plan_id=plan_id,
            smell=smell,
            actions=actions,
            estimated_effort=2.0 * len(actions),
            risk_level="low",
            expected_improvement={'complexity_reduction': 0.5},
            description=f" '{func_name}' "
        )

    def _generate_introduce_class_plan(self, plan_id: str, smell: SmellInstance) -> RefactorPlan:
        actions = []
        params = smell.metadata.get('params', [])
        
        if params:
            class_name = ''.join(p.capitalize() for p in params[:2]) + 'Context'
            
            if smell.locations:
                primary_file = smell.locations[0].file_path
                module_dir = os.path.dirname(primary_file)
                new_class_file = os.path.join(module_dir, f'{class_name.lower()}.py')
                
                actions.append(RefactorAction(
                    operation=RefactorOperation.EXTRACT_CLASS,
                    description=f" {class_name} : {', '.join(params)}",
                    new_name=class_name,
                    affected_files=[new_class_file]
                ))
                
                for location in smell.locations:
                    actions.append(RefactorAction(
                        operation=RefactorOperation.MOVE_FIELD,
                        description=f" {location.file_path}  {class_name} ",
                        source_location=location,
                        affected_files=[location.file_path, new_class_file]
                    ))
        
        return RefactorPlan(
            plan_id=plan_id,
            smell=smell,
            actions=actions,
            estimated_effort=4.0 + len(smell.locations),
            risk_level="medium",
            expected_improvement={'clump_reduction': 0.7},
            description=f": {', '.join(params)}"
        )

    def _generate_move_constant_plan(self, plan_id: str, smell: SmellInstance) -> RefactorPlan:
        actions = []
        files = smell.metadata.get('files', [])
        
        if files:
            common_dir = os.path.commonpath(files)
            constants_file = os.path.join(common_dir, 'constants.py')
            
            if not os.path.exists(constants_file):
                actions.append(RefactorAction(
                    operation=RefactorOperation.EXTRACT_CLASS,
                    description=f" {constants_file} ",
                    new_name='constants',
                    affected_files=[constants_file]
                ))
            
            for fp in files:
                actions.append(RefactorAction(
                    operation=RefactorOperation.MOVE_FIELD,
                    description=f" {fp}  {constants_file}",
                    affected_files=[fp, constants_file]
                ))
        
        return RefactorPlan(
            plan_id=plan_id,
            smell=smell,
            actions=actions,
            estimated_effort=2.0 * len(files),
            risk_level="low",
            expected_improvement={'shotgun_reduction': 0.9},
            description=f""
        )

    def _generate_move_method_plan(self, plan_id: str, smell: SmellInstance) -> RefactorPlan:
        actions = []
        func_name = smell.metadata.get('function_name', 'unknown')
        external_classes = smell.metadata.get('external_classes', [])
        
        if external_classes and smell.locations:
            primary_class = external_classes[0]
            
            actions.append(RefactorAction(
                operation=RefactorOperation.MOVE_METHOD,
                description=f" '{func_name}'  {primary_class} ",
                source_location=smell.locations[0],
                affected_files=[loc.file_path for loc in smell.locations]
            ))
            
            for location in smell.locations:
                actions.append(RefactorAction(
                    operation=RefactorOperation.REMOVE_DUPLICATION,
                    description=f" {location.file_path} ",
                    source_location=location,
                    affected_files=[location.file_path]
                ))
        
        return RefactorPlan(
            plan_id=plan_id,
            smell=smell,
            actions=actions,
            estimated_effort=3.0,
            risk_level="medium",
            expected_improvement={'feature_envy_reduction': 0.6},
            description=f" {primary_class}"
        )

    def _module_to_path(self, module_name: str) -> str:
        return os.path.join(self.analyzer.source_dir, module_name.replace('.', os.sep)) + '.py'

    def evaluate_plan_risk(self, plan: RefactorPlan) -> str:
        affected_files = set()
        for action in plan.actions:
            affected_files.update(action.affected_files)
        
        file_count = len(affected_files)
        action_count = len(plan.actions)
        
        if file_count > 10 or action_count > 5:
            return "high"
        elif file_count > 5 or action_count > 3:
            return "medium"
        else:
            return "low"

    def get_plan_summary(self, plan: RefactorPlan) -> str:
        summary = [f" ID: {plan.plan_id}"]
        summary.append(f": {plan.smell.smell_type.value}")
        summary.append(f": {plan.smell.severity.value}")
        summary.append(f": {plan.risk_level}")
        summary.append(f": {plan.estimated_effort} ")
        summary.append(f": {plan.expected_improvement}")
        summary.append(":")
        for i, action in enumerate(plan.actions, 1):
            summary.append(f"  {i}. {action.description}")
        return '\n'.join(summary)