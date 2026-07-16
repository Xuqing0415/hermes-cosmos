from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from .runtime_monitor import ViolationReport, ViolationType, StateTransition
from .counterexample_analyzer import AnalysisResult, FixSuggestion


class RootCauseType(Enum):
    CODE_BUG = "code_bug"
    SPEC_OUTDATED = "spec_outdated"
    MISSING_PRECONDITION = "missing_precondition"
    CONCURRENCY_ISSUE = "concurrency_issue"
    IMPLEMENTATION_GAP = "implementation_gap"
    ENVIRONMENT_ISSUE = "environment_issue"
    UNKNOWN = "unknown"


class FixActionType(Enum):
    CODE_FIX = "code_fix"
    SPEC_UPDATE = "spec_update"
    REQUIREMENT_REFINEMENT = "requirement_refinement"
    HOTFIX_DEPLOY = "hotfix_deploy"
    ROLLBACK = "rollback"
    MANUAL_REVIEW = "manual_review"


@dataclass
class RootCause:
    type: RootCauseType
    description: str
    confidence: float = 0.0
    affected_operation: Optional[str] = None
    suggested_fix: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "confidence": self.confidence,
            "affected_operation": self.affected_operation,
            "suggested_fix": self.suggested_fix
        }


@dataclass
class FixAction:
    action_type: FixActionType
    target: str
    description: str
    priority: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "description": self.description,
            "priority": self.priority,
            "parameters": self.parameters
        }


@dataclass
class AnomalyAnalysisResult:
    success: bool
    violations: List[ViolationReport] = field(default_factory=list)
    root_causes: List[RootCause] = field(default_factory=list)
    fix_actions: List[FixAction] = field(default_factory=list)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "violations": [v.to_dict() for v in self.violations],
            "root_causes": [r.to_dict() for r in self.root_causes],
            "fix_actions": [f.to_dict() for f in self.fix_actions],
            "message": self.message
        }


class AnomalyAnalyzer:
    def __init__(self):
        self._analyzers = {
            ViolationType.STATE_INVARIANT: self._analyze_state_invariant_violation,
            ViolationType.TRANSITION_NOT_ALLOWED: self._analyze_transition_violation,
            ViolationType.VALUE_OUT_OF_BOUNDS: self._analyze_value_out_of_bounds,
            ViolationType.INVALID_OPERATION: self._analyze_invalid_operation,
            ViolationType.CONCURRENCY_VIOLATION: self._analyze_concurrency_violation,
        }
    
    def analyze(self, violations: List[ViolationReport]) -> AnomalyAnalysisResult:
        if not violations:
            return AnomalyAnalysisResult(
                success=True,
                message="No violations to analyze"
            )
        
        root_causes = []
        fix_actions = []
        
        for violation in violations:
            analyzer = self._analyzers.get(violation.violation_type)
            if analyzer:
                cause, actions = analyzer(violation)
                root_causes.append(cause)
                fix_actions.extend(actions)
        
        fix_actions.sort(key=lambda x: -x.priority)
        
        return AnomalyAnalysisResult(
            success=True,
            violations=violations,
            root_causes=root_causes,
            fix_actions=fix_actions,
            message=f"Analyzed {len(violations)} violations, found {len(root_causes)} root causes"
        )
    
    def _analyze_state_invariant_violation(self, violation: ViolationReport) -> tuple:
        transition = violation.transition
        post_state = transition.post_state
        
        val = post_state.get("value", 0)
        
        if val < 0:
            return RootCause(
                type=RootCauseType.MISSING_PRECONDITION,
                description=f"Counter value became negative: {val}. "
                           f"This indicates the decrement operation was allowed when value <= 0.",
                confidence=0.95,
                affected_operation="decrement",
                suggested_fix="Add precondition 'value > 0' for decrement operation"
            ), [
                FixAction(
                    action_type=FixActionType.REQUIREMENT_REFINEMENT,
                    target="decrement",
                    description="Add precondition 'value > 0' for decrement operation",
                    priority=10,
                    parameters={"precondition": "value > 0"}
                ),
                FixAction(
                    action_type=FixActionType.CODE_FIX,
                    target="app.py",
                    description="Update decrement method to check value > 0 before decrementing",
                    priority=9
                ),
                FixAction(
                    action_type=FixActionType.HOTFIX_DEPLOY,
                    target="deployment",
                    description="Deploy updated code with precondition check",
                    priority=8
                )
            ]
        
        return RootCause(
            type=RootCauseType.CODE_BUG,
            description=f"State invariant violated: {post_state}",
            confidence=0.8,
            suggested_fix="Review implementation for state corruption"
        ), [
            FixAction(
                action_type=FixActionType.MANUAL_REVIEW,
                target="code",
                description="Manual review required for state invariant violation",
                priority=7
            )
        ]
    
    def _analyze_transition_violation(self, violation: ViolationReport) -> tuple:
        transition = violation.transition
        operation = transition.operation
        pre_state = transition.pre_state
        post_state = transition.post_state
        
        pre_val = pre_state.get("value", 0)
        post_val = post_state.get("value", 0)
        
        if operation.lower() == "decrement":
            if pre_val == 0 and post_val < 0:
                return RootCause(
                    type=RootCauseType.MISSING_PRECONDITION,
                    description=f"Decrement allowed when value was {pre_val}, resulted in {post_val}",
                    confidence=0.95,
                    affected_operation="decrement",
                    suggested_fix="Add precondition 'value > 0' for decrement"
                ), [
                    FixAction(
                        action_type=FixActionType.REQUIREMENT_REFINEMENT,
                        target="decrement",
                        description="Add precondition 'value > 0'",
                        priority=10
                    ),
                    FixAction(
                        action_type=FixActionType.CODE_FIX,
                        target="app.py",
                        description="Add guard clause in decrement method",
                        priority=9
                    )
                ]
        
        if operation.lower() == "increment":
            if post_val != pre_val + 1:
                return RootCause(
                    type=RootCauseType.CONCURRENCY_ISSUE,
                    description=f"Increment resulted in jump from {pre_val} to {post_val}, "
                               f"skipping intermediate values. Possible race condition.",
                    confidence=0.9,
                    affected_operation="increment",
                    suggested_fix="Add mutex lock or atomic operation"
                ), [
                    FixAction(
                        action_type=FixActionType.CODE_FIX,
                        target="app.py",
                        description="Add mutex lock around increment operation",
                        priority=10
                    ),
                    FixAction(
                        action_type=FixActionType.REQUIREMENT_REFINEMENT,
                        target="increment",
                        description="Add concurrency constraint",
                        priority=8
                    )
                ]
        
        return RootCause(
            type=RootCauseType.CODE_BUG,
            description=f"Transition '{operation}' not allowed: {pre_state} -> {post_state}",
            confidence=0.75,
            affected_operation=operation,
            suggested_fix="Review operation implementation"
        ), [
            FixAction(
                action_type=FixActionType.CODE_FIX,
                target="app.py",
                description=f"Fix {operation} implementation",
                priority=8
            )
        ]
    
    def _analyze_value_out_of_bounds(self, violation: ViolationReport) -> tuple:
        return RootCause(
            type=RootCauseType.MISSING_PRECONDITION,
            description=f"Value out of allowed bounds",
            confidence=0.85,
            suggested_fix="Add bounds checking in operations"
        ), [
            FixAction(
                action_type=FixActionType.CODE_FIX,
                target="app.py",
                description="Add bounds validation",
                priority=9
            )
        ]
    
    def _analyze_invalid_operation(self, violation: ViolationReport) -> tuple:
        return RootCause(
            type=RootCauseType.IMPLEMENTATION_GAP,
            description=f"Invalid or unknown operation: {violation.transition.operation}",
            confidence=0.8,
            suggested_fix="Implement missing operation or validate input"
        ), [
            FixAction(
                action_type=FixActionType.CODE_FIX,
                target="app.py",
                description="Implement missing operation",
                priority=7
            )
        ]
    
    def _analyze_concurrency_violation(self, violation: ViolationReport) -> tuple:
        return RootCause(
            type=RootCauseType.CONCURRENCY_ISSUE,
            description=f"Concurrency violation detected during {violation.transition.operation}",
            confidence=0.9,
            suggested_fix="Add synchronization mechanism"
        ), [
            FixAction(
                action_type=FixActionType.CODE_FIX,
                target="app.py",
                description="Add thread synchronization",
                priority=10
            ),
            FixAction(
                action_type=FixActionType.SPEC_UPDATE,
                target="spec",
                description="Update spec with concurrency constraints",
                priority=8
            )
        ]
    
    def analyze_sequence(self, transitions: List[StateTransition]) -> AnomalyAnalysisResult:
        violations = []
        
        for i, transition in enumerate(transitions):
            pre_val = transition.pre_state.get("value", 0)
            post_val = transition.post_state.get("value", 0)
            operation = transition.operation.lower()
            
            if operation == "decrement" and pre_val <= 0 and post_val < 0:
                from .runtime_monitor import ViolationReport as VR, ViolationType as VT
                violation = VR(
                    id=f"seq_violation_{i}",
                    timestamp=transition.timestamp,
                    violation_type=VT.STATE_INVARIANT,
                    transition=transition,
                    message=f"Decrement on zero resulted in negative value"
                )
                violations.append(violation)
            
            elif operation == "increment" and post_val != pre_val + 1:
                from .runtime_monitor import ViolationReport as VR, ViolationType as VT
                violation = VR(
                    id=f"seq_violation_{i}",
                    timestamp=transition.timestamp,
                    violation_type=VT.TRANSITION_NOT_ALLOWED,
                    transition=transition,
                    message=f"Increment resulted in unexpected state change"
                )
                violations.append(violation)
        
        return self.analyze(violations)