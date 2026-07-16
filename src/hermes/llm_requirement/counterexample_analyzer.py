from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .tlc_validator import TLCCounterexample, TLCState


class ViolationType(Enum):
    INVARIANT_VIOLATION = "invariant_violation"
    PROPERTY_VIOLATION = "property_violation"
    DEADLOCK = "deadlock"
    LIVELOCK = "livelock"
    BOUNDS_VIOLATION = "bounds_violation"
    TRANSITION_ERROR = "transition_error"
    UNKNOWN = "unknown"


@dataclass
class FixSuggestion:
    action_type: str
    target_operation: Optional[str] = None
    suggestion: str = ""
    precondition: Optional[str] = None
    effect_change: Optional[str] = None
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_operation": self.target_operation,
            "suggestion": self.suggestion,
            "precondition": self.precondition,
            "effect_change": self.effect_change,
            "priority": self.priority
        }


@dataclass
class AnalysisResult:
    violation_type: ViolationType
    violating_property: str
    description: str = ""
    problematic_action: Optional[str] = None
    problematic_state: Optional[Dict[str, str]] = None
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "violating_property": self.violating_property,
            "description": self.description,
            "problematic_action": self.problematic_action,
            "problematic_state": self.problematic_state,
            "fix_suggestions": [fs.to_dict() for fs in self.fix_suggestions],
            "confidence": self.confidence
        }


class CounterexampleAnalyzer:
    def __init__(self):
        self._analyzers = {
            "NoLostUpdates": self._analyze_no_lost_updates,
            "TypeOK": self._analyze_type_ok,
            "Deadlock": self._analyze_deadlock,
            "LIFO": self._analyze_lifo,
            "FIFO": self._analyze_fifo,
        }
    
    def analyze(self, counterexample: TLCCounterexample) -> AnalysisResult:
        property_name = counterexample.violating_property
        
        if property_name in self._analyzers:
            return self._analyzers[property_name](counterexample)
        
        return self._analyze_generic(counterexample)
    
    def _analyze_no_lost_updates(self, ce: TLCCounterexample) -> AnalysisResult:
        if not ce.states:
            return AnalysisResult(
                violation_type=ViolationType.INVARIANT_VIOLATION,
                violating_property=ce.violating_property,
                description="NoLostUpdates invariant violated",
                confidence=0.8
            )
        
        suggestions = []
        problematic_action = None
        problematic_state = None
        
        for i, state in enumerate(ce.states[:-1]):
            next_state = ce.states[i + 1]
            
            if "value" in state.variables and "value" in next_state.variables:
                try:
                    current_val = int(state.variables["value"])
                    next_val = int(next_state.variables["value"])
                    
                    if next_val > current_val + 1:
                        problematic_action = state.action
                        problematic_state = state.variables
                        suggestions.append(FixSuggestion(
                            action_type="add_precondition",
                            target_operation="increment",
                            suggestion=f"Value jumped from {current_val} to {next_val}, skipping intermediate values. This indicates lost updates.",
                            precondition="Add mutex lock or atomic operation to prevent race conditions",
                            priority=10
                        ))
                        break
                    
                    if next_val < current_val:
                        problematic_action = state.action
                        problematic_state = state.variables
                        suggestions.append(FixSuggestion(
                            action_type="add_precondition",
                            target_operation="decrement",
                            suggestion=f"Value decreased from {current_val} to {next_val} without explicit decrement",
                            precondition="Add precondition 'value > 0' for decrement operation",
                            priority=9
                        ))
                        break
                except ValueError:
                    pass
        
        if not suggestions:
            suggestions.append(FixSuggestion(
                action_type="add_constraint",
                suggestion="Concurrent operations may cause lost updates",
                priority=8
            ))
        
        return AnalysisResult(
            violation_type=ViolationType.INVARIANT_VIOLATION,
            violating_property=ce.violating_property,
            description="NoLostUpdates invariant violated: concurrent updates may be lost",
            problematic_action=problematic_action,
            problematic_state=problematic_state,
            fix_suggestions=suggestions,
            confidence=0.85
        )
    
    def _analyze_type_ok(self, ce: TLCCounterexample) -> AnalysisResult:
        suggestions = []
        problematic_action = None
        problematic_state = None
        
        for state in ce.states:
            if "value" in state.variables:
                try:
                    val = int(state.variables["value"])
                    if val < 0:
                        problematic_action = state.action
                        problematic_state = state.variables
                        suggestions.append(FixSuggestion(
                            action_type="add_precondition",
                            target_operation="decrement",
                            suggestion=f"Value became negative: {val}",
                            precondition="Add precondition 'value > 0' for decrement",
                            priority=10
                        ))
                        break
                except ValueError:
                    pass
        
        if not suggestions:
            suggestions.append(FixSuggestion(
                action_type="add_constraint",
                suggestion="TypeOK invariant violated: variable out of allowed range",
                priority=7
            ))
        
        return AnalysisResult(
            violation_type=ViolationType.INVARIANT_VIOLATION,
            violating_property=ce.violating_property,
            description="TypeOK invariant violated: variable value out of bounds",
            problematic_action=problematic_action,
            problematic_state=problematic_state,
            fix_suggestions=suggestions,
            confidence=0.9
        )
    
    def _analyze_deadlock(self, ce: TLCCounterexample) -> AnalysisResult:
        suggestions = [
            FixSuggestion(
                action_type="add_action",
                suggestion="Deadlock detected: no enabled actions in current state",
                priority=10
            ),
            FixSuggestion(
                action_type="relax_precondition",
                suggestion="Relax preconditions of operations to prevent deadlock",
                priority=9
            )
        ]
        
        return AnalysisResult(
            violation_type=ViolationType.DEADLOCK,
            violating_property=ce.violating_property,
            description="Deadlock: no transitions enabled from current state",
            fix_suggestions=suggestions,
            confidence=0.95
        )
    
    def _analyze_lifo(self, ce: TLCCounterexample) -> AnalysisResult:
        suggestions = [
            FixSuggestion(
                action_type="fix_order",
                suggestion="LIFO order violated: stack operations do not preserve last-in-first-out order",
                priority=8
            )
        ]
        
        return AnalysisResult(
            violation_type=ViolationType.INVARIANT_VIOLATION,
            violating_property=ce.violating_property,
            description="LIFO invariant violated",
            fix_suggestions=suggestions,
            confidence=0.75
        )
    
    def _analyze_fifo(self, ce: TLCCounterexample) -> AnalysisResult:
        suggestions = [
            FixSuggestion(
                action_type="fix_order",
                suggestion="FIFO order violated: queue operations do not preserve first-in-first-out order",
                priority=8
            )
        ]
        
        return AnalysisResult(
            violation_type=ViolationType.INVARIANT_VIOLATION,
            violating_property=ce.violating_property,
            description="FIFO invariant violated",
            fix_suggestions=suggestions,
            confidence=0.75
        )
    
    def _analyze_generic(self, ce: TLCCounterexample) -> AnalysisResult:
        suggestions = []
        
        if ce.states:
            last_state = ce.states[-1]
            problematic_state = last_state.variables
            problematic_action = last_state.action
            
            for var_name, var_value in problematic_state.items():
                try:
                    val = int(var_value)
                    if val < 0:
                        suggestions.append(FixSuggestion(
                            action_type="add_precondition",
                            suggestion=f"Variable '{var_name}' has negative value {val}",
                            priority=8
                        ))
                except ValueError:
                    pass
        
        if not suggestions:
            suggestions.append(FixSuggestion(
                action_type="review_spec",
                suggestion=f"Property '{ce.violating_property}' violated. Review the specification.",
                priority=5
            ))
        
        return AnalysisResult(
            violation_type=ViolationType.UNKNOWN,
            violating_property=ce.violating_property,
            description=f"Property '{ce.violating_property}' violated",
            problematic_action=problematic_action if ce.states else None,
            problematic_state=ce.states[-1].variables if ce.states else None,
            fix_suggestions=suggestions,
            confidence=0.6
        )