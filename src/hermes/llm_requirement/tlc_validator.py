import subprocess
import os
import re
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class TLCStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class TLCState:
    step: int
    action: str
    variables: Dict[str, str]


@dataclass
class TLCCounterexample:
    violating_property: str
    states: List[TLCState] = field(default_factory=list)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violating_property": self.violating_property,
            "message": self.message,
            "states": [{
                "step": s.step,
                "action": s.action,
                "variables": s.variables
            } for s in self.states]
        }


@dataclass
class TLCResult:
    status: TLCStatus
    spec_name: str
    counterexample: Optional[TLCCounterexample] = None
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "spec_name": self.spec_name,
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
            "error_message": self.error_message,
            "duration": self.duration
        }


class TLCValidator:
    def __init__(self, tlc_path: str = "tlc", java_path: str = "java", 
                 tla2tools_jar: Optional[str] = None):
        self.tlc_path = tlc_path
        self.java_path = java_path
        self.tla2tools_jar = tla2tools_jar
        self._tlc_available = None
    
    def is_tlc_available(self) -> bool:
        if self._tlc_available is not None:
            return self._tlc_available
        
        try:
            result = subprocess.run(
                [self.tlc_path, "-version"],
                capture_output=True, text=True, timeout=5
            )
            self._tlc_available = result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
            try:
                if self.tla2tools_jar and os.path.exists(self.tla2tools_jar):
                    result = subprocess.run(
                        [self.java_path, "-jar", self.tla2tools_jar, "-version"],
                        capture_output=True, text=True, timeout=5
                    )
                    self._tlc_available = result.returncode == 0
                else:
                    self._tlc_available = False
            except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
                self._tlc_available = False
        
        return self._tlc_available
    
    def validate(self, spec_file: str, 
                 timeout: int = 120,
                 deadlock: bool = True,
                 coverage: bool = False) -> TLCResult:
        spec_name = os.path.basename(spec_file).replace(".tla", "")
        spec_dir = os.path.dirname(spec_file) or "."
        
        if not self.is_tlc_available():
            return TLCResult(
                status=TLCStatus.ERROR,
                spec_name=spec_name,
                error_message="TLC model checker not available. Please install TLA+ Toolbox or configure tla2tools.jar path."
            )
        
        try:
            args = self._build_tlc_command(spec_file, deadlock, coverage)
            
            result = subprocess.run(
                args,
                capture_output=True, text=True,
                cwd=spec_dir,
                timeout=timeout
            )
            
            return self._parse_tlc_output(spec_name, result)
            
        except subprocess.TimeoutExpired:
            return TLCResult(
                status=TLCStatus.TIMEOUT,
                spec_name=spec_name,
                error_message="TLC validation timed out"
            )
        except Exception as e:
            return TLCResult(
                status=TLCStatus.ERROR,
                spec_name=spec_name,
                error_message=str(e)
            )
    
    def _build_tlc_command(self, spec_file: str, deadlock: bool, coverage: bool) -> List[str]:
        if self.tla2tools_jar and os.path.exists(self.tla2tools_jar):
            args = [self.java_path, "-jar", self.tla2tools_jar]
        else:
            args = [self.tlc_path]
        
        if deadlock:
            args.append("-deadlock")
        if coverage:
            args.append("-coverage")
        
        args.append(spec_file)
        return args
    
    def _parse_tlc_output(self, spec_name: str, result: subprocess.CompletedProcess) -> TLCResult:
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode == 0:
            if "No errors found" in stdout or "successfully" in stdout.lower():
                return TLCResult(
                    status=TLCStatus.SUCCESS,
                    spec_name=spec_name,
                    stdout=stdout
                )
        
        counterexample = self._extract_counterexample(stdout, stderr)
        
        if counterexample:
            return TLCResult(
                status=TLCStatus.FAILURE,
                spec_name=spec_name,
                counterexample=counterexample,
                stdout=stdout,
                stderr=stderr
            )
        
        if "deadlock" in stdout.lower() or "deadlock" in stderr.lower():
            return TLCResult(
                status=TLCStatus.FAILURE,
                spec_name=spec_name,
                counterexample=TLCCounterexample(
                    violating_property="Deadlock",
                    message="Deadlock detected in the specification"
                ),
                stdout=stdout,
                stderr=stderr
            )
        
        return TLCResult(
            status=TLCStatus.FAILURE,
            spec_name=spec_name,
            error_message=f"TLC validation failed: {stdout[:200]} {stderr[:200]}",
            stdout=stdout,
            stderr=stderr
        )
    
    def _extract_counterexample(self, stdout: str, stderr: str) -> Optional[TLCCounterexample]:
        full_output = stdout + "\n" + stderr
        
        violation_match = re.search(
            r"(Invariant|Property|Theorem)\s+(\w+)\s+is\s+violated",
            full_output, re.IGNORECASE
        )
        violating_property = violation_match.group(2) if violation_match else "Unknown"
        
        state_pattern = r"State\s+(\d+):\s*((?:.|\n)*?)(?=\nState\s+\d+:|\Z)"
        state_matches = re.findall(state_pattern, full_output)
        
        states = []
        for step_str, state_content in state_matches:
            step = int(step_str)
            action_match = re.search(r"Action:\s*(.+)", state_content)
            action = action_match.group(1).strip() if action_match else ""
            
            var_pattern = r"(\w+)\s*=\s*(.+?)\s*(?=\n\w+\s*=|\Z)"
            variables = {}
            for var_name, var_value in re.findall(var_pattern, state_content):
                variables[var_name.strip()] = var_value.strip()
            
            if variables:
                states.append(TLCState(step=step, action=action, variables=variables))
        
        if states:
            return TLCCounterexample(
                violating_property=violating_property,
                states=states,
                message=f"Counterexample with {len(states)} states"
            )
        
        return None
    
    def validate_counter_spec(self) -> TLCResult:
        from .spec_generator import SpecGenerator
        import tempfile
        
        generator = SpecGenerator()
        spec = generator.generate_counter_tlaplus()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, f"{spec.name}.tla")
            spec.save(spec_path)
            return self.validate(spec_path)