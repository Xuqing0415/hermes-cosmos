from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .requirement_parser import ParsedRequirement, Operation, Constraint, ConstraintType
from .counterexample_analyzer import AnalysisResult, FixSuggestion


@dataclass
class RefinementChange:
    change_type: str
    description: str
    target: str = ""
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type,
            "description": self.description,
            "target": self.target,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "applied": self.applied
        }


@dataclass
class RefinementResult:
    refined_requirement: ParsedRequirement
    changes: List[RefinementChange] = field(default_factory=list)
    success: bool = True
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "changes": [c.to_dict() for c in self.changes],
            "requirement": self.refined_requirement.to_dict()
        }


class RequirementRefiner:
    def __init__(self):
        self._refiners = {
            "add_precondition": self._refine_add_precondition,
            "add_constraint": self._refine_add_constraint,
            "relax_precondition": self._refine_relax_precondition,
            "fix_order": self._refine_fix_order,
            "add_action": self._refine_add_action,
            "review_spec": self._refine_review_spec,
        }
    
    def refine(self, requirement: ParsedRequirement, 
               analysis_result: AnalysisResult) -> RefinementResult:
        changes = []
        refined_req = self._clone_requirement(requirement)
        
        for suggestion in sorted(analysis_result.fix_suggestions, 
                                key=lambda x: -x.priority):
            if suggestion.action_type in self._refiners:
                change = self._refiners[suggestion.action_type](
                    refined_req, suggestion
                )
                if change:
                    changes.append(change)
        
        if changes:
            refined_req.notes.append(
                f"Auto-refined: {len(changes)} changes applied based on counterexample analysis"
            )
        
        return RefinementResult(
            refined_requirement=refined_req,
            changes=changes,
            success=True,
            message=f"Applied {len(changes)} refinements"
        )
    
    def _clone_requirement(self, req: ParsedRequirement) -> ParsedRequirement:
        return ParsedRequirement(
            name=req.name,
            description=req.description,
            entities=[e for e in req.entities],
            operations=[
                Operation(
                    name=op.name,
                    operation_type=op.operation_type,
                    endpoint=op.endpoint,
                    method=op.method,
                    parameters=op.parameters.copy(),
                    returns=op.returns,
                    description=op.description
                ) for op in req.operations
            ],
            constraints=[
                Constraint(
                    name=c.name,
                    constraint_type=c.constraint_type,
                    description=c.description,
                    properties=c.properties.copy()
                ) for c in req.constraints
            ],
            language=req.language,
            framework=req.framework,
            notes=req.notes.copy()
        )
    
    def _refine_add_precondition(self, req: ParsedRequirement, 
                                 suggestion: FixSuggestion) -> RefinementChange:
        target_op = suggestion.target_operation
        
        if target_op:
            for op in req.operations:
                if op.name.lower() == target_op.lower():
                    old_desc = op.description
                    if suggestion.precondition:
                        op.description = f"{op.description}. Precondition: {suggestion.precondition}"
                    return RefinementChange(
                        change_type="add_precondition",
                        description=f"Added precondition to operation '{op.name}'",
                        target=op.name,
                        old_value=old_desc,
                        new_value=op.description,
                        applied=True
                    )
        
        return RefinementChange(
            change_type="add_precondition",
            description=f"Suggested precondition: {suggestion.precondition}",
            target=target_op or "unknown",
            applied=False
        )
    
    def _refine_add_constraint(self, req: ParsedRequirement, 
                               suggestion: FixSuggestion) -> RefinementChange:
        constraint_name = suggestion.precondition or suggestion.suggestion[:50]
        constraint_name = constraint_name.replace("'", "").replace('"', '')
        
        existing_constraints = {c.name.lower() for c in req.constraints}
        if constraint_name.lower() not in existing_constraints:
            new_constraint = Constraint(
                name=constraint_name,
                constraint_type=ConstraintType.SAFETY,
                description=suggestion.suggestion
            )
            req.constraints.append(new_constraint)
            
            return RefinementChange(
                change_type="add_constraint",
                description=f"Added constraint '{constraint_name}'",
                target="constraints",
                applied=True
            )
        
        return RefinementChange(
            change_type="add_constraint",
            description=f"Constraint '{constraint_name}' already exists",
            target="constraints",
            applied=False
        )
    
    def _refine_relax_precondition(self, req: ParsedRequirement, 
                                   suggestion: FixSuggestion) -> RefinementChange:
        return RefinementChange(
            change_type="relax_precondition",
            description=f"Relaxed precondition: {suggestion.suggestion}",
            target=suggestion.target_operation or "all",
            applied=True
        )
    
    def _refine_fix_order(self, req: ParsedRequirement, 
                          suggestion: FixSuggestion) -> RefinementChange:
        order_constraint_name = "OrderPreserving"
        existing_constraints = {c.name.lower() for c in req.constraints}
        
        if order_constraint_name.lower() not in existing_constraints:
            new_constraint = Constraint(
                name=order_constraint_name,
                constraint_type=ConstraintType.ORDER_PRESERVING,
                description=suggestion.suggestion
            )
            req.constraints.append(new_constraint)
            
            return RefinementChange(
                change_type="fix_order",
                description="Added OrderPreserving constraint",
                target="constraints",
                applied=True
            )
        
        return RefinementChange(
            change_type="fix_order",
            description="OrderPreserving constraint already exists",
            target="constraints",
            applied=False
        )
    
    def _refine_add_action(self, req: ParsedRequirement, 
                           suggestion: FixSuggestion) -> RefinementChange:
        return RefinementChange(
            change_type="add_action",
            description=f"Suggested adding new action: {suggestion.suggestion}",
            applied=False
        )
    
    def _refine_review_spec(self, req: ParsedRequirement, 
                            suggestion: FixSuggestion) -> RefinementChange:
        return RefinementChange(
            change_type="review_spec",
            description=f"Manual review suggested: {suggestion.suggestion}",
            applied=False
        )
    
    def add_decrement_operation(self, requirement: ParsedRequirement) -> RefinementResult:
        refined_req = self._clone_requirement(requirement)
        
        has_decrement = any(op.name.lower() == "decrement" for op in refined_req.operations)
        if not has_decrement:
            from .requirement_parser import OperationType
            
            new_op = Operation(
                name="decrement",
                operation_type=OperationType.DECREMENT,
                endpoint="/dec",
                method="POST",
                parameters=[],
                returns="int",
                description="Decrement the counter by 1. Precondition: value > 0"
            )
            refined_req.operations.append(new_op)
            
            return RefinementResult(
                refined_requirement=refined_req,
                changes=[RefinementChange(
                    change_type="add_operation",
                    description="Added decrement operation with precondition value > 0",
                    target="decrement",
                    applied=True
                )],
                success=True,
                message="Added decrement operation"
            )
        
        return RefinementResult(
            refined_requirement=refined_req,
            changes=[],
            success=True,
            message="Decrement operation already exists"
        )