"""
Orchestrator that integrates the Perceiver -> Sage -> Knight pipeline
with cross-domain knowledge distillation for automated fix generation.
"""

from typing import List, Optional, Dict, Any
import time
import logging

from hermes.core.interfaces import (
    DomainContext, PainPoint, PatchPlan, ExecutionResult,
    Perceiver, Sage, Knight, DomainPlugin
)
from hermes.cross_domain.abstract_pattern_extractor import (
    AbstractPatternExtractor, AbstractPattern, AbstractPatternType
)
from hermes.cross_domain.pattern_similarity_engine import PatternSimilarityEngine
from hermes.cross_domain.cross_domain_translator import CrossDomainTranslator, DomainFix
from hermes.cross_domain.knowledge_amalgamator import KnowledgeAmalgamator
from hermes.cross_domain.knowledge_self_evolver import KnowledgeSelfEvolver, EvolutionResult

logger = logging.getLogger(__name__)


class OrchestratorResult:
    """Result of a cross-domain orchestration run for a single pain point."""

    def __init__(self, pain_point: PainPoint):
        self.pain_point = pain_point
        self.direct_fix: Optional[PatchPlan] = None
        self.cross_domain_fix: Optional[PatchPlan] = None
        self.execution_result: Optional[ExecutionResult] = None
        self.cross_domain_used: bool = False
        self.source_domain: Optional[str] = None
        self.similarity: float = 0.0
        self.pattern_type: Optional[str] = None
        self.success: bool = False
        self.duration_ms: float = 0.0
        self.evolution_result: Optional[EvolutionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pain_point_id": self.pain_point.id,
            "pain_point_type": self.pain_point.type,
            "pain_point_message": self.pain_point.message,
            "direct_fix": self.direct_fix is not None,
            "cross_domain_used": self.cross_domain_used,
            "source_domain": self.source_domain,
            "similarity": round(self.similarity, 2),
            "pattern_type": self.pattern_type,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
        }


class CrossDomainOrchestrator:
    """
    Orchestrator that runs the Perceiver -> Sage -> Knight pipeline
    with cross-domain knowledge distillation fallback.

    When a Sage cannot find a direct fix template for a pain point,
    the orchestrator queries the cross-domain knowledge graph for
    similar patterns from other domains, translates the fix, and
    applies it in the current domain.
    """

    def __init__(
        self,
        plugin: DomainPlugin,
        cross_domain_enabled: bool = True,
        similarity_threshold: float = 0.5,
        knowledge_amalgamator: Optional[KnowledgeAmalgamator] = None,
        translator: Optional[CrossDomainTranslator] = None,
        similarity_engine: Optional[PatternSimilarityEngine] = None,
        pattern_extractor: Optional[AbstractPatternExtractor] = None,
    ):
        self.plugin = plugin
        self.cross_domain_enabled = cross_domain_enabled
        self.similarity_threshold = similarity_threshold

        self._perceiver: Perceiver = plugin.get_perceiver()
        self._sage: Sage = plugin.get_sage()
        self._knight: Knight = plugin.get_knight()

        self._knowledge = knowledge_amalgamator or KnowledgeAmalgamator(
            storage_path=".cross_domain_graph.json"
        )
        self._translator = translator or CrossDomainTranslator()
        self._similarity_engine = similarity_engine or PatternSimilarityEngine()
        self._pattern_extractor = pattern_extractor or AbstractPatternExtractor()

        self._evolver = KnowledgeSelfEvolver(
            amalgamator=self._knowledge,
            full_evolution_interval=10,
            min_cluster_size=2,
        )

    def run(self, context: DomainContext) -> List[OrchestratorResult]:
        """Run the full pipeline with cross-domain fallback."""
        domain = context.domain
        results: List[OrchestratorResult] = []

        logger.info("[Orchestrator] Running domain: %s", domain)

        # Step 1: Perceiver - detect issues
        pain_points = self._perceiver.detect(context)
        logger.info("[Perceiver] Found %d issue(s)", len(pain_points))

        for pain_point in pain_points:
            start = time.time()
            result = OrchestratorResult(pain_point)
            logger.info("  [Perceiver] Found: %s", pain_point.message)

            # Step 2: Sage - try direct fix
            direct_plan = self._sage.generate_patch(context, pain_point)
            has_direct_operations = bool(direct_plan and direct_plan.operations)
            result.direct_fix = direct_plan if has_direct_operations else None

            if has_direct_operations:
                logger.info("  [Sage] Direct fix template found")
                patch_plan = direct_plan
            elif self.cross_domain_enabled:
                # Step 3: Cross-domain fallback
                logger.info("  [Sage] No direct fix template found.")
                logger.info("  [CrossDomain] Querying knowledge graph...")

                cross_domain_result = self._try_cross_domain_fix(
                    context, pain_point, result
                )

                if cross_domain_result is not None:
                    patch_plan = cross_domain_result
                    result.cross_domain_fix = patch_plan
                    result.cross_domain_used = True
                else:
                    logger.info("  [CrossDomain] No suitable cross-domain fix found")
                    patch_plan = direct_plan
            else:
                logger.info("  [Sage] No direct fix template found (cross-domain disabled)")
                patch_plan = direct_plan

            # Step 4: Knight - execute and verify
            if patch_plan and patch_plan.operations:
                logger.info("  [Knight] Applying patch...")
                exec_result = self._knight.execute(context, patch_plan)
                verify_result = self._knight.verify(context, patch_plan)
                result.execution_result = verify_result

                if verify_result.success:
                    result.success = True
                    logger.info("  [Knight] Tests passed!")

                    # Step 5: Record cross-domain success
                    if result.cross_domain_used and result.source_domain:
                        self._record_cross_domain_success(pain_point, result, context)
                else:
                    logger.info("  [Knight] Tests failed.")
            else:
                result.execution_result = ExecutionResult(
                    success=False,
                    patch_plan_id="",
                    message="No patch plan available",
                    verification_results=[]
                )

            result.duration_ms = (time.time() - start) * 1000
            results.append(result)

        return results

    def _try_cross_domain_fix(
        self,
        context: DomainContext,
        pain_point: PainPoint,
        result: OrchestratorResult,
    ) -> Optional[PatchPlan]:
        """Attempt to find and apply a cross-domain fix."""
        domain = context.domain

        # Extract pattern from the current pain point
        pain_point_data = {
            "type": pain_point.type,
            "severity": pain_point.severity,
            "message": pain_point.message,
            "location": pain_point.location,
            "context": pain_point.context,
        }
        current_pattern = self._pattern_extractor.extract_pattern(
            domain, pain_point_data
        )
        pattern_type_str = current_pattern.pattern_type.value if current_pattern else pain_point.type

        # Query knowledge graph for similar patterns using the extracted pattern type
        similar_entries = self._knowledge.query_similar(
            pattern_type_str, top_k=5
        )

        if not similar_entries:
            logger.info("  [CrossDomain] No similar patterns found in knowledge graph")
            return None

        best_match = similar_entries[0]
        similarity = best_match.get("similarity", 0.0)
        result.similarity = similarity

        if similarity < self.similarity_threshold:
            logger.info(
                "  [CrossDomain] Best match similarity %.2f below threshold %.2f",
                similarity, self.similarity_threshold
            )
            return None

        source_domain = best_match.get("source_domain", "default")
        pattern_type_str = best_match.get("pattern_type", "")
        result.source_domain = source_domain
        result.pattern_type = pattern_type_str

        logger.info(
            "  [CrossDomain] Found similar pattern '%s' in %s plugin (sim: %.2f)",
            pattern_type_str, source_domain, similarity
        )

        # Map pattern type string to enum
        pattern_type = self._map_pattern_type(pattern_type_str)

        # Translate fix from source domain to target domain
        domain_fix = self._translator.translate_fix(
            source_domain, domain, pattern_type
        )

        if domain_fix is None:
            logger.info("  [CrossDomain] No translation available for %s -> %s",
                         source_domain, domain)
            return None

        logger.info("  [CrossDomain] Translating fix from %s to %s...",
                     source_domain, domain)

        # Generate patch plan using Sage's cross_domain_fallback
        patch_plan = self._sage.cross_domain_fallback(
            context, pain_point, domain_fix
        )

        return patch_plan

    def _record_cross_domain_success(
        self,
        pain_point: PainPoint,
        result: OrchestratorResult,
        context: DomainContext,
    ):
        """Record a successful cross-domain fix in the knowledge graph."""
        logger.info("  [CrossDomain] New cross-domain link added!")

        self._knowledge.add_cross_domain_link(
            source_domain=result.source_domain or "unknown",
            target_domain=context.domain,
            pattern_type=result.pattern_type or pain_point.type,
            similarity=result.similarity,
            success=True
        )

        # Boost similarity weight for future queries
        self._knowledge.boost_similarity(
            pattern_type=result.pattern_type or pain_point.type,
            increment=0.05
        )

        # Run knowledge self-evolution
        evolution = self._evolver.on_migration_result(
            source_type=result.pattern_type or pain_point.type,
            target_type=pain_point.type,
            source_domain=result.source_domain or "unknown",
            target_domain=context.domain,
            similarity=result.similarity,
            success=True,
        )
        result.evolution_result = evolution

        if evolution.has_changes:
            if evolution.new_patterns:
                for p in evolution.new_patterns:
                    logger.info("  [KnowledgeEvolver] New pattern discovered: \"%s\" (%.2f)",
                                p.name, p.confidence)
            if evolution.matrix_updates:
                for u in evolution.matrix_updates:
                    logger.info("  [KnowledgeEvolver] Similarity matrix: %s ↔ %s: %.2f → %.2f (%+.2f)",
                                u["type_a"], u["type_b"],
                                u["old_value"], u["new_value"], u["delta"])
            if evolution.new_templates:
                for t in evolution.new_templates:
                    logger.info("  [KnowledgeEvolver] New template: \"%s\" (applicable to %s)",
                                t.name, ", ".join(t.applicable_domains))

    def _map_pattern_type(self, pattern_type_str: str) -> AbstractPatternType:
        """Map a pattern type string to an AbstractPatternType enum."""
        for pt in AbstractPatternType:
            if pt.value == pattern_type_str:
                return pt
        return AbstractPatternType.BOUNDARY_CHECK_MISSING