from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import re


class AbstractPatternType(str, Enum):
    BOUNDARY_CHECK_MISSING = "boundary_check_missing"
    RESOURCE_LIMIT_MISSING = "resource_limit_missing"
    INPUT_VALIDATION_MISSING = "input_validation_missing"
    API_VERSION_DEPRECATED = "api_version_deprecated"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    DIVISION_BY_ZERO = "division_by_zero"
    UNINITIALIZED_VARIABLE = "uninitialized_variable"


@dataclass
class AbstractPattern:
    id: str
    pattern_type: AbstractPatternType
    domain: str
    original_type: str
    severity: str
    message: str
    location: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    context: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "domain": self.domain,
            "original_type": self.original_type,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "keywords": self.keywords,
            "context": self.context,
            "confidence": self.confidence
        }


DOMAIN_PATTERN_MAP: Dict[str, Dict[str, AbstractPatternType]] = {
    "mlir": {
        "index_out_of_bounds": AbstractPatternType.BOUNDARY_CHECK_MISSING,
        "missing_bound_check": AbstractPatternType.BOUNDARY_CHECK_MISSING,
        "potential_divide_by_zero": AbstractPatternType.DIVISION_BY_ZERO,
        "uninitialized_variable": AbstractPatternType.UNINITIALIZED_VARIABLE,
    },
    "k8s": {
        "missing_resources": AbstractPatternType.RESOURCE_LIMIT_MISSING,
        "hpa_missing": AbstractPatternType.RESOURCE_LIMIT_MISSING,
        "deprecated_api": AbstractPatternType.API_VERSION_DEPRECATED,
        "old_image": AbstractPatternType.API_VERSION_DEPRECATED,
    },
    "default": {
        "null_pointer": AbstractPatternType.BOUNDARY_CHECK_MISSING,
        "missing_validation": AbstractPatternType.INPUT_VALIDATION_MISSING,
        "performance": AbstractPatternType.PERFORMANCE_BOTTLENECK,
    },
}


SEVERITY_MAPPING: Dict[str, str] = {
    "high": "critical",
    "medium": "important",
    "low": "warning",
}


KEYWORD_EXTRACTORS: Dict[str, List[str]] = {
    AbstractPatternType.BOUNDARY_CHECK_MISSING.value: [
        "bound", "limit", "index", "size", "range", "check", "overflow", "access"
    ],
    AbstractPatternType.RESOURCE_LIMIT_MISSING.value: [
        "resource", "limit", "memory", "cpu", "hpa", "request", "capacity"
    ],
    AbstractPatternType.INPUT_VALIDATION_MISSING.value: [
        "validate", "input", "request", "body", "parameter", "check", "sanitize"
    ],
    AbstractPatternType.API_VERSION_DEPRECATED.value: [
        "version", "deprecated", "old", "upgrade", "api", "image", "tag"
    ],
    AbstractPatternType.PERFORMANCE_BOTTLENECK.value: [
        "performance", "optimize", "speed", "latency", "throughput", "bottleneck"
    ],
    AbstractPatternType.DIVISION_BY_ZERO.value: [
        "divide", "zero", "denominator", "divisor", "arithmetic"
    ],
    AbstractPatternType.UNINITIALIZED_VARIABLE.value: [
        "variable", "init", "assign", "declare", "undefined", "result"
    ],
}


class AbstractPatternExtractor:
    def __init__(self):
        self._pattern_id_counter = 0

    def extract_pattern(self, domain: str, pain_point_data: Dict[str, Any]) -> Optional[AbstractPattern]:
        original_type = pain_point_data.get("type", "")
        severity = pain_point_data.get("severity", "medium")
        message = pain_point_data.get("message", "")

        domain_map = DOMAIN_PATTERN_MAP.get(domain, {})
        pattern_type = domain_map.get(original_type)

        if not pattern_type:
            pattern_type = self._infer_pattern_type(domain, original_type, message)

        if not pattern_type:
            return None

        keywords = self._extract_keywords(pattern_type, message)
        confidence = self._calculate_confidence(pattern_type, keywords)

        self._pattern_id_counter += 1

        return AbstractPattern(
            id=f"pattern-{self._pattern_id_counter:04d}",
            pattern_type=pattern_type,
            domain=domain,
            original_type=original_type,
            severity=SEVERITY_MAPPING.get(severity, severity),
            message=message,
            location=pain_point_data.get("location"),
            keywords=keywords,
            context=pain_point_data.get("context"),
            confidence=confidence
        )

    def extract_patterns(self, domain: str, pain_points_data: List[Dict[str, Any]]) -> List[AbstractPattern]:
        patterns = []
        for pain_point in pain_points_data:
            pattern = self.extract_pattern(domain, pain_point)
            if pattern:
                patterns.append(pattern)
        return patterns

    def _infer_pattern_type(self, domain: str, original_type: str, message: str) -> Optional[AbstractPatternType]:
        message_lower = message.lower()

        if any(kw in message_lower for kw in ["bound", "limit", "index", "out of"]):
            return AbstractPatternType.BOUNDARY_CHECK_MISSING
        elif any(kw in message_lower for kw in ["resource", "cpu", "memory", "hpa"]):
            return AbstractPatternType.RESOURCE_LIMIT_MISSING
        elif any(kw in message_lower for kw in ["validate", "input", "null", "none"]):
            return AbstractPatternType.INPUT_VALIDATION_MISSING
        elif any(kw in message_lower for kw in ["version", "deprecated", "old"]):
            return AbstractPatternType.API_VERSION_DEPRECATED
        elif any(kw in message_lower for kw in ["performance", "optimize", "slow"]):
            return AbstractPatternType.PERFORMANCE_BOTTLENECK
        elif any(kw in message_lower for kw in ["divide", "zero"]):
            return AbstractPatternType.DIVISION_BY_ZERO
        elif any(kw in message_lower for kw in ["uninitialized", "undefined"]):
            return AbstractPatternType.UNINITIALIZED_VARIABLE

        return None

    def _extract_keywords(self, pattern_type: AbstractPatternType, message: str) -> List[str]:
        keywords = []
        message_lower = message.lower()
        extractors = KEYWORD_EXTRACTORS.get(pattern_type.value, [])

        for kw in extractors:
            if kw in message_lower:
                keywords.append(kw)

        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', message)
        for token in tokens[:5]:
            if token.lower() not in keywords:
                keywords.append(token.lower())

        return list(set(keywords))[:10]

    def _calculate_confidence(self, pattern_type: AbstractPatternType, keywords: List[str]) -> float:
        extractors = KEYWORD_EXTRACTORS.get(pattern_type.value, [])
        matched = sum(1 for kw in keywords if kw in extractors)

        if matched == 0:
            return 0.3
        elif matched <= 2:
            return 0.5 + matched * 0.1
        elif matched <= 4:
            return 0.7 + (matched - 2) * 0.075
        else:
            return 0.95