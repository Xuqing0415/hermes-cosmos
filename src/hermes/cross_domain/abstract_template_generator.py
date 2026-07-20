"""
Abstract Template Generator Module
Generates abstract fix templates from concrete cross-domain fix cases.
When the system finds that fix patterns from multiple domains share the
same structure, it generates an abstract template applicable across domains.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re

from .abstract_pattern_extractor import AbstractPattern, AbstractPatternType
from .knowledge_amalgamator import KnowledgeAmalgamator, KnowledgeNode, KnowledgeEdge
from .cross_domain_translator import DomainFix, UniversalFixTemplate


@dataclass
class AbstractFixTemplate:
    """An abstract fix template generated from multiple domain-specific fixes."""
    template_id: str
    name: str
    description: str
    pattern_type: str
    applicable_domains: List[str]
    template_code: str
    implementation_steps: List[str]
    confidence: float
    source_count: int
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "pattern_type": self.pattern_type,
            "applicable_domains": self.applicable_domains,
            "template_code": self.template_code,
            "implementation_steps": self.implementation_steps,
            "confidence": round(self.confidence, 2),
            "source_count": self.source_count,
            "generated_at": self.generated_at.isoformat(),
        }


class AbstractTemplateGenerator:
    """
    Generates abstract fix templates from the knowledge graph.
    Analyzes cross-domain fix patterns and extracts common structures.
    """

    def __init__(self, amalgamator: KnowledgeAmalgamator):
        self._amalgamator = amalgamator
        self._template_counter = 0

    def generate_templates(self, min_domains: int = 2) -> List[AbstractFixTemplate]:
        """
        Generate abstract fix templates from the knowledge graph.
        A template is generated when the same pattern type appears in
        at least `min_domains` different domains.
        """
        graph = self._amalgamator.get_graph()
        templates: List[AbstractFixTemplate] = []

        # Group nodes by pattern type
        type_to_nodes: Dict[str, List[KnowledgeNode]] = {}
        for node in graph.nodes:
            pt = node.pattern_type.value
            if pt not in type_to_nodes:
                type_to_nodes[pt] = []
            type_to_nodes[pt].append(node)

        for pattern_type, nodes in type_to_nodes.items():
            # Collect all domains for this pattern type
            all_domains = set()
            for n in nodes:
                all_domains.update(n.domains)

            if len(all_domains) < min_domains:
                continue

            # Generate a template for this pattern type
            self._template_counter += 1
            template = self._build_template(
                pattern_type, nodes, list(all_domains)
            )
            if template:
                templates.append(template)

        return sorted(templates, key=lambda t: t.confidence, reverse=True)

    def generate_from_discovered_pattern(self, name: str, description: str,
                                          domains: List[str],
                                          keywords: List[str]) -> Optional[AbstractFixTemplate]:
        """Generate an abstract template from a newly discovered pattern."""
        self._template_counter += 1
        template_id = f"at-{self._template_counter:04d}"

        # Map keywords to template code
        code = self._generate_template_code(name, keywords)
        steps = self._generate_implementation_steps(name, keywords)

        return AbstractFixTemplate(
            template_id=template_id,
            name=name.replace("_", " ").title(),
            description=description,
            pattern_type=name,
            applicable_domains=domains,
            template_code=code,
            implementation_steps=steps,
            confidence=0.6,
            source_count=len(domains),
        )

    def _build_template(self, pattern_type: str, nodes: List[KnowledgeNode],
                        domains: List[str]) -> Optional[AbstractFixTemplate]:
        """Build an abstract template from nodes of the same pattern type."""
        self._template_counter += 1
        template_id = f"at-{self._template_counter:04d}"

        # Collect all keywords across nodes
        all_keywords = []
        max_confidence = 0.0
        for n in nodes:
            all_keywords.extend(n.representative_keywords)
            max_confidence = max(max_confidence, n.confidence)

        # Count keyword frequency
        from collections import Counter
        kw_counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in kw_counter.most_common(8)]

        # Generate name from pattern type
        name = pattern_type.replace("_", " ").title()

        # Generate description
        domain_list = ", ".join(sorted(domains))
        kw_list = ", ".join(top_keywords[:5])
        description = f"Abstract fix for '{pattern_type}': applicable to [{domain_list}], keywords: [{kw_list}]"

        # Generate template code
        code = self._generate_template_code(pattern_type, top_keywords)
        steps = self._generate_implementation_steps(pattern_type, top_keywords)

        return AbstractFixTemplate(
            template_id=template_id,
            name=name,
            description=description,
            pattern_type=pattern_type,
            applicable_domains=domains,
            template_code=code,
            implementation_steps=steps,
            confidence=max_confidence,
            source_count=len(nodes),
        )

    def _generate_template_code(self, pattern_type: str,
                                keywords: List[str]) -> str:
        """Generate abstract template code from keywords."""
        kw_set = set(k.lower() for k in keywords)
        lines = []

        if "bound" in kw_set or "index" in kw_set or "range" in kw_set or "size" in kw_set:
            lines.append("if index < 0 or index >= limit:")
            lines.append("    raise ValueError(f\"Index {index} out of bounds [0, {limit})\")")
            lines.append("")

        if "null" in kw_set or "missing" in kw_set or "check" in kw_set:
            lines.append("if value is None:")
            lines.append("    raise ValueError(\"Required value is missing\")")
            lines.append("")

        if "resource" in kw_set or "limit" in kw_set or "capacity" in kw_set:
            lines.append("if requested > available:")
            lines.append("    raise ResourceWarning(f\"Requested {requested} exceeds available {available}\")")
            lines.append("")

        if "type" in kw_set or "mismatch" in kw_set or "validation" in kw_set:
            lines.append("if not isinstance(value, expected_type):")
            lines.append("    raise TypeError(f\"Expected {expected_type}, got {type(value)}\")")
            lines.append("")

        if "timeout" in kw_set or "deadline" in kw_set or "duration" in kw_set:
            lines.append("if elapsed > timeout_limit:")
            lines.append("    raise TimeoutError(f\"Operation timed out after {elapsed}s\")")
            lines.append("")

        if not lines:
            # Fallback: generic validation template
            lines.append("def validate(value, constraints):")
            lines.append("    for constraint in constraints:")
            lines.append("        if not constraint.check(value):")
            lines.append("            raise ValueError(f\"Validation failed: {constraint}\")")
            lines.append("")

        return "\n".join(lines)

    def _generate_implementation_steps(self, pattern_type: str,
                                       keywords: List[str]) -> List[str]:
        """Generate implementation steps from keywords."""
        kw_set = set(k.lower() for k in keywords)
        steps = []

        if "bound" in kw_set or "index" in kw_set or "range" in kw_set or "size" in kw_set:
            steps.append("Identify all array/list index access points")
            steps.append("Add bounds check before each access")
            steps.append("Handle out-of-bounds cases with appropriate error")

        if "null" in kw_set or "missing" in kw_set or "check" in kw_set:
            steps.append("Identify nullable values and missing data paths")
            steps.append("Add null/None checks before usage")
            steps.append("Provide fallback values or error handling")

        if "resource" in kw_set or "limit" in kw_set or "capacity" in kw_set:
            steps.append("Identify resource allocation points")
            steps.append("Add capacity check before allocation")
            steps.append("Implement graceful degradation on limit exceeded")

        if "type" in kw_set or "mismatch" in kw_set or "validation" in kw_set:
            steps.append("Identify type conversion points")
            steps.append("Add type validation before conversion")
            steps.append("Handle type mismatch with clear error messages")

        if "timeout" in kw_set or "deadline" in kw_set or "duration" in kw_set:
            steps.append("Identify blocking operations with timeouts")
            steps.append("Add timeout tracking and deadline checks")
            steps.append("Handle timeout with cleanup and retry logic")

        if not steps:
            steps.append("Identify the root cause of the issue")
            steps.append("Add defensive validation")
            steps.append("Verify the fix with tests")

        return steps