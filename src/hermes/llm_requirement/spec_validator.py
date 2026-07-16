from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from .spec_generator import TLAPlusSpec


class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


class ValidationResult:
    def __init__(self, spec_name: str, status: ValidationStatus):
        self.spec_name = spec_name
        self.status = status
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.details: Dict[str, Any] = {}
    
    def add_error(self, error: str):
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def set_detail(self, key: str, value: Any):
        self.details[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "status": self.status.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details
        }


class SpecValidator:
    def __init__(self, use_tlc: bool = False):
        self.use_tlc = use_tlc
        self._validators = [
            self._validate_syntax,
            self._validate_constants,
            self._validate_variables,
            self._validate_invariants,
            self._validate_properties,
        ]
    
    def validate(self, spec: TLAPlusSpec) -> ValidationResult:
        result = ValidationResult(spec.name, ValidationStatus.PASSED)
        
        if self.use_tlc:
            return self._validate_with_tlc(spec)
        
        for validator in self._validators:
            validator(spec, result)
        
        if result.errors:
            result.status = ValidationStatus.FAILED
        
        return result
    
    def _validate_syntax(self, spec: TLAPlusSpec, result: ValidationResult):
        content = spec.module_content
        
        if not content.startswith("---- MODULE"):
            result.add_error("Module must start with '---- MODULE'")
        
        if not content.strip().endswith("===="):
            result.add_error("Module must end with '===='")
        
        if "Init ==" not in content and "Init ==" not in content:
            result.add_warning("No Init definition found")
        
        if "Next ==" not in content:
            result.add_warning("No Next definition found")
        
        if "Spec ==" not in content:
            result.add_warning("No Spec definition found")
    
    def _validate_constants(self, spec: TLAPlusSpec, result: ValidationResult):
        content = spec.module_content
        
        constants_match = re.search(r"CONSTANTS\s*\n((?:\s*\w+\s*)+)", content)
        if constants_match:
            constants = [c.strip() for c in constants_match.group(1).split() if c.strip()]
            result.set_detail("constants", constants)
    
    def _validate_variables(self, spec: TLAPlusSpec, result: ValidationResult):
        content = spec.module_content
        
        variables_match = re.search(r"VARIABLES\s*\n((?:\s*\w+\s*)+)", content)
        if variables_match:
            variables = [v.strip() for v in variables_match.group(1).split() if v.strip()]
            result.set_detail("variables", variables)
            result.set_detail("variable_count", len(variables))
        else:
            result.add_warning("No VARIABLES section found")
    
    def _validate_invariants(self, spec: TLAPlusSpec, result: ValidationResult):
        content = spec.module_content
        
        invariant_pattern = r"(\w+)\s*==\s*[^=]"
        invariants = re.findall(invariant_pattern, content)
        valid_invariants = [i for i in invariants if i not in ["Init", "Next", "Spec", "CONSTANTS", "VARIABLES"]]
        
        result.set_detail("invariants_found", valid_invariants)
        
        if not spec.invariants and not valid_invariants:
            result.add_warning("No invariants defined")
    
    def _validate_properties(self, spec: TLAPlusSpec, result: ValidationResult):
        if spec.properties:
            result.set_detail("properties", spec.properties)
        
        if spec.temporal_properties:
            result.set_detail("temporal_properties", spec.temporal_properties)
            for prop in spec.temporal_properties:
                if "=>" not in prop:
                    result.add_warning(f"Temporal property '{prop}' may not be properly formatted")
    
    def _validate_with_tlc(self, spec: TLAPlusSpec) -> ValidationResult:
        result = ValidationResult(spec.name, ValidationStatus.INCONCLUSIVE)
        result.add_warning("TLC validator not available in this environment")
        result.add_warning("Using mock validation instead")
        
        mock_result = self.validate(spec)
        result.status = mock_result.status
        result.errors = mock_result.errors
        result.warnings = mock_result.warnings
        result.details = mock_result.details
        
        return result
    
    def validate_counter_spec(self) -> ValidationResult:
        from .spec_generator import SpecGenerator
        generator = SpecGenerator()
        spec = generator.generate_counter_tlaplus()
        return self.validate(spec)