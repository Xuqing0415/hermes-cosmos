from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .abstract_pattern_extractor import AbstractPattern, AbstractPatternType


@dataclass
class DomainFix:
    domain: str
    fix_description: str
    code_template: str
    implementation_steps: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "fix_description": self.fix_description,
            "code_template": self.code_template,
            "implementation_steps": self.implementation_steps,
            "confidence": round(self.confidence, 2)
        }


@dataclass
class UniversalFixTemplate:
    abstract_pattern_type: AbstractPatternType
    problem_summary: str
    domain_fixes: List[DomainFix] = field(default_factory=list)
    cross_domain_insight: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstract_pattern_type": self.abstract_pattern_type.value,
            "problem_summary": self.problem_summary,
            "domain_fixes": [df.to_dict() for df in self.domain_fixes],
            "cross_domain_insight": self.cross_domain_insight
        }


FIX_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    AbstractPatternType.BOUNDARY_CHECK_MISSING.value: {
        "default": {
            "description": "Add bounds checking before array access",
            "template": "if index < 0 or index >= len(array):\n    raise ValueError(f\"Index {index} out of bounds\")",
            "steps": [
                "Identify array access locations",
                "Add bounds validation before access",
                "Use try-except or explicit condition",
                "Add error handling for invalid indices"
            ],
            "confidence": 0.9
        },
        "mlir": {
            "description": "Add cf.assert for index range",
            "template": """cf.assert (arith.cmpi slt, %idx, %size) {
  message = "Index out of bounds"
}""",
            "steps": [
                "Insert cf.assert operation before memref access",
                "Use arith.cmpi to compare index with bound",
                "Add diagnostic message",
                "Run mlir-opt to verify"
            ],
            "confidence": 0.95
        },
        "k8s": {
            "description": "Add resource limits to prevent unbounded usage",
            "template": """resources:
  limits:
    cpu: "100m"
    memory: "256Mi"
  requests:
    cpu: "50m"
    memory: "128Mi"
""",
            "steps": [
                "Add resources section to container spec",
                "Define CPU and memory limits",
                "Set appropriate requests",
                "Apply with kubectl patch"
            ],
            "confidence": 0.85
        },
    },
    AbstractPatternType.RESOURCE_LIMIT_MISSING.value: {
        "default": {
            "description": "Add memory protection and resource limits",
            "template": "import resource\nresource.setrlimit(resource.RLIMIT_MEMLOCK, (max_mem, max_mem))",
            "steps": [
                "Import resource module",
                "Set memory limits using setrlimit",
                "Define maximum allowed memory",
                "Add fallback for platforms without resource support"
            ],
            "confidence": 0.7
        },
        "mlir": {
            "description": "Add buffer size assertions",
            "template": """%buffer_size = memref.dim %buf, 0 : memref<?xf32>
cf.assert (arith.cmpi sge, %buffer_size, %required_size) {
  message = "Insufficient buffer capacity"
}""",
            "steps": [
                "Get buffer dimensions with memref.dim",
                "Compare with required size",
                "Add assertion for minimum capacity",
                "Validate with symbolic execution"
            ],
            "confidence": 0.8
        },
        "k8s": {
            "description": "Add CPU/memory limits to deployment",
            "template": """spec:
  template:
    spec:
      containers:
      - name: {{name}}
        resources:
          limits:
            cpu: "{{cpu_limit}}"
            memory: "{{memory_limit}}"
          requests:
            cpu: "{{cpu_request}}"
            memory: "{{memory_request}}" """,
            "steps": [
                "Update deployment spec",
                "Add resources.limits for CPU and memory",
                "Add resources.requests for guaranteed QoS",
                "Apply changes and verify with kubectl describe"
            ],
            "confidence": 0.95
        },
    },
    AbstractPatternType.INPUT_VALIDATION_MISSING.value: {
        "default": {
            "description": "Add input validation and sanitization",
            "template": "def validate_input(data):\n    if not data:\n        raise ValueError(\"Input cannot be empty\")\n    return sanitize(data)",
            "steps": [
                "Create validation function",
                "Check for None/empty values",
                "Validate data types",
                "Sanitize user input"
            ],
            "confidence": 0.9
        },
        "mlir": {
            "description": "Add type and shape verification",
            "template": """%result = call @validate_input(%input) : (memref<?x?xf32>) -> i1
cf.assert %result {
  message = "Invalid input shape or type"
}""",
            "steps": [
                "Create validation function",
                "Check tensor dimensions",
                "Verify element types",
                "Add assertion for validation result"
            ],
            "confidence": 0.85
        },
        "k8s": {
            "description": "Add admission controller validation",
            "template": """apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: input-validator
webhooks:
- name: validate.example.com
  rules:
  - apiGroups: ["apps"]
    apiVersions: ["v1"]
    operations: ["CREATE", "UPDATE"]
    resources: ["deployments"]""",
            "steps": [
                "Deploy validation webhook",
                "Configure admission rules",
                "Define validation criteria",
                "Test with kubectl apply"
            ],
            "confidence": 0.75
        },
    },
    AbstractPatternType.API_VERSION_DEPRECATED.value: {
        "default": {
            "description": "Upgrade deprecated API calls",
            "template": "# Replace deprecated API call\n# OLD: deprecated_api.call()\n# NEW: new_api.call_v2()",
            "steps": [
                "Identify deprecated API usages",
                "Check compatibility matrix",
                "Update to latest version",
                "Run regression tests"
            ],
            "confidence": 0.8
        },
        "mlir": {
            "description": "Update to latest dialect version",
            "template": "# Convert from old dialect to new\n# dialect.use_old_version\n# ->\n# dialect.use_new_version",
            "steps": [
                "Identify deprecated operations",
                "Check dialect migration guide",
                "Apply mlir-translate conversion",
                "Verify with mlir-opt"
            ],
            "confidence": 0.85
        },
        "k8s": {
            "description": "Update to stable API version",
            "template": "# apiVersion: networking.k8s.io/v1beta1\napiVersion: networking.k8s.io/v1",
            "steps": [
                "Identify deprecated API versions",
                "Update apiVersion field",
                "Adjust CRD schema if needed",
                "Apply with kubectl apply"
            ],
            "confidence": 0.95
        },
    },
}


CROSS_DOMAIN_INSIGHTS: Dict[str, str] = {
    AbstractPatternType.BOUNDARY_CHECK_MISSING.value:
        "All domains share the fundamental concept of boundary validation. "
        "What appears as array bounds checking in Python becomes memory protection in MLIR "
        "and resource limits in Kubernetes.",
    AbstractPatternType.RESOURCE_LIMIT_MISSING.value:
        "Resource exhaustion is a universal concern across domains. "
        "While Python uses memory limits, MLIR uses buffer capacity checks, "
        "and Kubernetes uses resource requests/limits - all serve the same purpose.",
    AbstractPatternType.INPUT_VALIDATION_MISSING.value:
        "Input validation prevents invalid data from propagating through systems. "
        "Python validates function arguments, MLIR validates tensor shapes/types, "
        "and Kubernetes validates deployment configurations.",
    AbstractPatternType.API_VERSION_DEPRECATED.value:
        "Technology evolves, and APIs become deprecated. "
        "All domains require regular updates to stay compatible with latest standards."
}


class CrossDomainTranslator:
    def __init__(self):
        self._fix_templates = FIX_TEMPLATES
        self._insights = CROSS_DOMAIN_INSIGHTS

    def translate_fix(self, source_domain: str, target_domain: str,
                      pattern_type: AbstractPatternType) -> Optional[DomainFix]:
        templates = self._fix_templates.get(pattern_type.value, {})

        if target_domain not in templates:
            return None

        template = templates[target_domain]

        return DomainFix(
            domain=target_domain,
            fix_description=template["description"],
            code_template=template["template"],
            implementation_steps=template["steps"],
            confidence=template["confidence"]
        )

    def generate_universal_fix(self, pattern_type: AbstractPatternType) -> UniversalFixTemplate:
        templates = self._fix_templates.get(pattern_type.value, {})
        domain_fixes = []

        for domain in ["default", "mlir", "k8s"]:
            if domain in templates:
                template = templates[domain]
                domain_fixes.append(DomainFix(
                    domain=domain,
                    fix_description=template["description"],
                    code_template=template["template"],
                    implementation_steps=template["steps"],
                    confidence=template["confidence"]
                ))

        insight = self._insights.get(pattern_type.value, "")

        return UniversalFixTemplate(
            abstract_pattern_type=pattern_type,
            problem_summary=self._generate_problem_summary(pattern_type),
            domain_fixes=domain_fixes,
            cross_domain_insight=insight
        )

    def translate_from_pattern(self, pattern: AbstractPattern, target_domain: str) -> Optional[DomainFix]:
        return self.translate_fix(pattern.domain, target_domain, pattern.pattern_type)

    def _generate_problem_summary(self, pattern_type: AbstractPatternType) -> str:
        summaries = {
            AbstractPatternType.BOUNDARY_CHECK_MISSING.value:
                "Missing boundary validation allows access beyond allowed limits.",
            AbstractPatternType.RESOURCE_LIMIT_MISSING.value:
                "Missing resource constraints can lead to resource exhaustion.",
            AbstractPatternType.INPUT_VALIDATION_MISSING.value:
                "Missing input validation allows invalid data to enter the system.",
            AbstractPatternType.API_VERSION_DEPRECATED.value:
                "Using deprecated API versions causes compatibility issues.",
            AbstractPatternType.PERFORMANCE_BOTTLENECK.value:
                "Performance bottlenecks reduce system throughput.",
            AbstractPatternType.DIVISION_BY_ZERO.value:
                "Potential division by zero causes runtime crashes.",
            AbstractPatternType.UNINITIALIZED_VARIABLE.value:
                "Using uninitialized variables leads to undefined behavior.",
        }
        return summaries.get(pattern_type.value, "Unknown problem type")