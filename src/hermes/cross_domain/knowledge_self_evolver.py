"""
Knowledge Self-Evolver Module
Orchestrates the entire self-evolution process of the knowledge graph.
Runs discovery and optimization after each successful migration,
and periodically performs full clustering.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

from .knowledge_amalgamator import KnowledgeAmalgamator
from .pattern_discovery import PatternDiscoveryEngine, DiscoveredPattern
from .similarity_learner import SimilarityLearner
from .abstract_template_generator import AbstractTemplateGenerator, AbstractFixTemplate
from .pattern_similarity_engine import PatternSimilarityEngine, PATTERN_TYPE_SIMILARITY

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """Result of a single evolution cycle."""
    new_patterns: List[DiscoveredPattern] = field(default_factory=list)
    matrix_updates: List[Dict[str, Any]] = field(default_factory=list)
    new_templates: List[AbstractFixTemplate] = field(default_factory=list)
    evolution_type: str = "lightweight"
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_patterns": [p.to_dict() for p in self.new_patterns],
            "matrix_updates": self.matrix_updates,
            "new_templates": [t.to_dict() for t in self.new_templates],
            "evolution_type": self.evolution_type,
            "duration_ms": round(self.duration_ms, 2),
        }

    @property
    def has_changes(self) -> bool:
        return bool(self.new_patterns or self.matrix_updates or self.new_templates)


class KnowledgeSelfEvolver:
    """
    Orchestrates the self-evolution of the knowledge graph.
    After each successful cross-domain migration, runs a lightweight evolution.
    Every N migrations, runs a full clustering cycle.

    The evolver:
    1. Runs pattern discovery to find new pattern types
    2. Updates similarity weights based on migration success
    3. Generates abstract fix templates for multi-domain patterns
    """

    def __init__(
        self,
        amalgamator: KnowledgeAmalgamator,
        similarity_learner: Optional[SimilarityLearner] = None,
        full_evolution_interval: int = 10,
        min_cluster_size: int = 2,
    ):
        self._amalgamator = amalgamator
        self._similarity_learner = similarity_learner or SimilarityLearner()
        self._discovery_engine = PatternDiscoveryEngine(amalgamator)
        self._template_generator = AbstractTemplateGenerator(amalgamator)
        self._full_evolution_interval = full_evolution_interval
        self._min_cluster_size = min_cluster_size
        self._total_migrations = 0

    @property
    def total_migrations(self) -> int:
        return self._total_migrations

    @property
    def similarity_learner(self) -> SimilarityLearner:
        return self._similarity_learner

    def on_migration_result(self, source_type: str, target_type: str,
                            source_domain: str, target_domain: str,
                            similarity: float, success: bool) -> EvolutionResult:
        """
        Called after each cross-domain migration attempt.
        Records the result and runs evolution.
        """
        self._total_migrations += 1
        self._similarity_learner.record_migration(
            source_type, target_type, source_domain, target_domain,
            similarity, success
        )

        if not success:
            return EvolutionResult(evolution_type="none")

        # Lightweight evolution: update similarity matrix
        result = EvolutionResult(evolution_type="lightweight")
        matrix_updates = self._get_matrix_updates(source_type, target_type, similarity)
        result.matrix_updates = matrix_updates

        # Full evolution every N successful migrations
        if self._total_migrations % self._full_evolution_interval == 0:
            result = self._run_full_evolution()
            result.evolution_type = "full"

        return result

    def run_full_evolution(self) -> EvolutionResult:
        """Run a full evolution cycle (pattern discovery + template gen)."""
        return self._run_full_evolution()

    def get_learned_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the current learned similarity matrix merged with defaults."""
        return self._similarity_learner.apply_to_initial_matrix(PATTERN_TYPE_SIMILARITY)

    def save_state(self):
        """Save the evolver's state (similarity learner) to disk."""
        self._similarity_learner.save()

    def load_state(self):
        """Load the evolver's state from disk."""
        self._similarity_learner.load()

    def _run_full_evolution(self) -> EvolutionResult:
        """Run the full evolution cycle."""
        import time
        start = time.time()

        result = EvolutionResult(evolution_type="full")

        # Step 1: Pattern discovery
        logger.info("[KnowledgeEvolver] Running pattern discovery...")
        new_patterns = self._discovery_engine.discover_new_patterns(
            min_cluster_size=self._min_cluster_size
        )
        result.new_patterns = new_patterns

        if new_patterns:
            logger.info("[KnowledgeEvolver] Found %d new pattern cluster(s):",
                        len(new_patterns))
            for p in new_patterns:
                logger.info("  - \"%s\" (linked to %s, confidence: %.2f)",
                           p.name, ", ".join(p.domains), p.confidence)

        # Step 2: Generate abstract templates
        logger.info("[KnowledgeEvolver] Generating abstract fix templates...")
        templates = self._template_generator.generate_templates(min_domains=2)

        # Also generate templates from discovered patterns
        for dp in new_patterns:
            dt = self._template_generator.generate_from_discovered_pattern(
                dp.name, dp.description, dp.domains, dp.representative_keywords
            )
            if dt:
                templates.append(dt)

        result.new_templates = templates

        if templates:
            logger.info("[KnowledgeEvolver] Generated %d abstract template(s):",
                        len(templates))
            for t in templates:
                logger.info("  - \"%s\" (applicable to %s)",
                           t.name, ", ".join(t.applicable_domains))

        # Step 3: Matrix updates summary
        matrix_updates = []
        for type_a, neighbors in self._similarity_learner.matrix.weights.items():
            for type_b, weight in neighbors.items():
                default_sim = PATTERN_TYPE_SIMILARITY.get(type_a, {}).get(type_b, None)
                if default_sim is not None and abs(weight - default_sim) > 0.01:
                    delta = weight - default_sim
                    matrix_updates.append({
                        "type_a": type_a,
                        "type_b": type_b,
                        "old_value": round(default_sim, 2),
                        "new_value": round(weight, 2),
                        "delta": round(delta, 2),
                    })
        result.matrix_updates = matrix_updates

        if matrix_updates:
            logger.info("[KnowledgeEvolver] Similarity matrix updated:")
            for u in matrix_updates[:5]:
                logger.info("  - %s ↔ %s: %.2f → %.2f (%+.2f)",
                           u["type_a"], u["type_b"],
                           u["old_value"], u["new_value"], u["delta"])

        result.duration_ms = (time.time() - start) * 1000
        logger.info("[KnowledgeEvolver] Full evolution completed in %.0fms",
                    result.duration_ms)
        return result

    def _get_matrix_updates(self, source_type: str, target_type: str,
                            old_similarity: float) -> List[Dict[str, Any]]:
        """Get the similarity matrix updates after a migration."""
        current = self._similarity_learner.matrix.get(source_type, target_type)
        if abs(current - old_similarity) < 0.01:
            return []
        return [{
            "type_a": source_type,
            "type_b": target_type,
            "old_value": round(old_similarity, 2),
            "new_value": round(current, 2),
            "delta": round(current - old_similarity, 2),
        }]