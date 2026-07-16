from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import uuid


class MonitorStatus(Enum):
    OK = "ok"
    VIOLATION = "violation"
    WARNING = "warning"
    ERROR = "error"


class ViolationType(Enum):
    STATE_INVARIANT = "state_invariant"
    TRANSITION_NOT_ALLOWED = "transition_not_allowed"
    VALUE_OUT_OF_BOUNDS = "value_out_of_bounds"
    INVALID_OPERATION = "invalid_operation"
    CONCURRENCY_VIOLATION = "concurrency_violation"


@dataclass
class StateTransition:
    id: str
    timestamp: float
    operation: str
    pre_state: Dict[str, Any]
    post_state: Dict[str, Any]
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "pre_state": self.pre_state,
            "post_state": self.post_state,
            "duration": self.duration
        }


@dataclass
class ViolationReport:
    id: str
    timestamp: float
    violation_type: ViolationType
    transition: StateTransition
    expected_state: Optional[Dict[str, Any]] = None
    message: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "violation_type": self.violation_type.value,
            "transition": self.transition.to_dict(),
            "expected_state": self.expected_state,
            "message": self.message,
            "confidence": self.confidence
        }


@dataclass
class MonitorResult:
    status: MonitorStatus
    transitions: List[StateTransition] = field(default_factory=list)
    violations: List[ViolationReport] = field(default_factory=list)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "transitions": [t.to_dict() for t in self.transitions],
            "violations": [v.to_dict() for v in self.violations],
            "message": self.message
        }


class TLASpecStateMachine:
    def __init__(self, spec_name: str = "Counter"):
        self.spec_name = spec_name
        self._allowed_transitions = {}
        self._invariants = []
        self._initialize_counter_spec()
    
    def _initialize_counter_spec(self):
        self._allowed_transitions = {
            "increment": self._is_valid_increment,
            "decrement": self._is_valid_decrement,
            "get": self._is_valid_get,
            "reset": self._is_valid_reset,
        }
        
        self._invariants = [
            self._check_non_negative,
            self._check_value_integer,
        ]
    
    def _is_valid_increment(self, pre_state: Dict[str, Any], 
                            post_state: Dict[str, Any]) -> bool:
        pre_val = pre_state.get("value", 0)
        post_val = post_state.get("value", 0)
        return post_val == pre_val + 1
    
    def _is_valid_decrement(self, pre_state: Dict[str, Any], 
                            post_state: Dict[str, Any]) -> bool:
        pre_val = pre_state.get("value", 0)
        post_val = post_state.get("value", 0)
        
        if pre_val <= 0:
            return post_val == pre_val
        
        return post_val == pre_val - 1
    
    def _is_valid_get(self, pre_state: Dict[str, Any], 
                      post_state: Dict[str, Any]) -> bool:
        return pre_state.get("value") == post_state.get("value")
    
    def _is_valid_reset(self, pre_state: Dict[str, Any], 
                        post_state: Dict[str, Any]) -> bool:
        return post_state.get("value") == 0
    
    def _check_non_negative(self, state: Dict[str, Any]) -> bool:
        val = state.get("value", 0)
        return val >= 0
    
    def _check_value_integer(self, state: Dict[str, Any]) -> bool:
        val = state.get("value")
        return isinstance(val, int)
    
    def validate_transition(self, operation: str, 
                            pre_state: Dict[str, Any],
                            post_state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        operation_lower = operation.lower()
        
        if operation_lower in self._allowed_transitions:
            if not self._allowed_transitions[operation_lower](pre_state, post_state):
                return False, f"Transition '{operation}' not allowed: {pre_state} -> {post_state}"
        else:
            return False, f"Unknown operation: {operation}"
        
        for invariant in self._invariants:
            if not invariant(post_state):
                return False, f"Invariant violated in post-state: {post_state}"
        
        return True, None
    
    def validate_state(self, state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        for invariant in self._invariants:
            if not invariant(state):
                return False, f"Invariant violated: {state}"
        return True, None


class RuntimeMonitor:
    def __init__(self, state_machine: Optional[TLASpecStateMachine] = None):
        self.state_machine = state_machine or TLASpecStateMachine()
        self._transitions: List[StateTransition] = []
        self._violations: List[ViolationReport] = []
        self._current_state: Dict[str, Any] = {"value": 0}
        self._monitors: List[Callable[[StateTransition], Optional[ViolationReport]]] = []
        
        self._register_default_monitors()
    
    def _register_default_monitors(self):
        self._monitors.append(self._monitor_state_invariants)
        self._monitors.append(self._monitor_transitions)
    
    def _monitor_state_invariants(self, transition: StateTransition) -> Optional[ViolationReport]:
        valid, reason = self.state_machine.validate_state(transition.post_state)
        if not valid:
            return ViolationReport(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                violation_type=ViolationType.STATE_INVARIANT,
                transition=transition,
                message=reason or "State invariant violation",
                confidence=0.95
            )
        return None
    
    def _monitor_transitions(self, transition: StateTransition) -> Optional[ViolationReport]:
        valid, reason = self.state_machine.validate_transition(
            transition.operation,
            transition.pre_state,
            transition.post_state
        )
        if not valid:
            return ViolationReport(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                violation_type=ViolationType.TRANSITION_NOT_ALLOWED,
                transition=transition,
                message=reason or "Transition not allowed by spec",
                confidence=0.90
            )
        return None
    
    def record_transition(self, operation: str, pre_state: Dict[str, Any],
                          post_state: Dict[str, Any], duration: float = 0.0) -> MonitorResult:
        transition = StateTransition(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            operation=operation,
            pre_state=pre_state.copy(),
            post_state=post_state.copy(),
            duration=duration
        )
        
        self._transitions.append(transition)
        self._current_state = post_state.copy()
        
        violations = []
        for monitor in self._monitors:
            violation = monitor(transition)
            if violation:
                violations.append(violation)
                self._violations.append(violation)
        
        if violations:
            return MonitorResult(
                status=MonitorStatus.VIOLATION,
                transitions=[transition],
                violations=violations,
                message=f"Detected {len(violations)} violations"
            )
        
        return MonitorResult(
            status=MonitorStatus.OK,
            transitions=[transition],
            violations=[],
            message="Transition validated successfully"
        )
    
    def get_current_state(self) -> Dict[str, Any]:
        return self._current_state.copy()
    
    def get_transitions(self, limit: Optional[int] = None) -> List[StateTransition]:
        if limit:
            return self._transitions[-limit:]
        return self._transitions.copy()
    
    def get_violations(self, limit: Optional[int] = None) -> List[ViolationReport]:
        if limit:
            return self._violations[-limit:]
        return self._violations.copy()
    
    def has_violations(self) -> bool:
        return len(self._violations) > 0
    
    def reset(self):
        self._transitions = []
        self._violations = []
        self._current_state = {"value": 0}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": self._current_state,
            "transition_count": len(self._transitions),
            "violation_count": len(self._violations),
            "has_violations": self.has_violations()
        }


class HTTPClientMonitor:
    def __init__(self, base_url: str, timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.monitor = RuntimeMonitor()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        import requests
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            duration = time.time() - start_time
            
            try:
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "json": response.json(),
                    "duration": duration
                }
            except ValueError:
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "text": response.text,
                    "duration": duration
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def get_state(self) -> Dict[str, Any]:
        result = self._make_request("GET", "/state")
        if result.get("success"):
            return result.get("json", {})
        return {"value": 0}
    
    def increment(self) -> MonitorResult:
        pre_state = self.get_state()
        result = self._make_request("POST", "/inc")
        post_state = self.get_state()
        
        return self.monitor.record_transition("increment", pre_state, post_state, result.get("duration", 0))
    
    def decrement(self) -> MonitorResult:
        pre_state = self.get_state()
        result = self._make_request("POST", "/dec")
        post_state = self.get_state()
        
        return self.monitor.record_transition("decrement", pre_state, post_state, result.get("duration", 0))
    
    def get(self) -> MonitorResult:
        pre_state = self.get_state()
        result = self._make_request("GET", "/get")
        post_state = self.get_state()
        
        return self.monitor.record_transition("get", pre_state, post_state, result.get("duration", 0))
    
    def reset(self) -> MonitorResult:
        pre_state = self.get_state()
        result = self._make_request("POST", "/reset")
        post_state = self.get_state()
        
        return self.monitor.record_transition("reset", pre_state, post_state, result.get("duration", 0))
    
    def run_sequence(self, operations: List[str]) -> List[MonitorResult]:
        results = []
        for op in operations:
            if op == "inc":
                results.append(self.increment())
            elif op == "dec":
                results.append(self.decrement())
            elif op == "get":
                results.append(self.get())
            elif op == "reset":
                results.append(self.reset())
        return results