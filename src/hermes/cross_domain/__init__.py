from .abstract_pattern_extractor import (
    AbstractPatternType,
    AbstractPattern,
    AbstractPatternExtractor,
)
from .pattern_similarity_engine import (
    SimilarityMatch,
    PatternCluster,
    PatternSimilarityEngine,
)
from .cross_domain_translator import (
    DomainFix,
    UniversalFixTemplate,
    CrossDomainTranslator,
)
from .knowledge_amalgamator import (
    KnowledgeNode,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeAmalgamator,
)
from .pattern_discovery import (
    PatternDiscoveryEngine,
    DiscoveredPattern,
)
from .similarity_learner import (
    SimilarityLearner,
    MigrationRecord,
    SimilarityMatrix,
)
from .abstract_template_generator import (
    AbstractTemplateGenerator,
    AbstractFixTemplate,
)
from .knowledge_self_evolver import (
    KnowledgeSelfEvolver,
    EvolutionResult,
)

__all__ = [
    "AbstractPatternType",
    "AbstractPattern",
    "AbstractPatternExtractor",
    "SimilarityMatch",
    "PatternCluster",
    "PatternSimilarityEngine",
    "DomainFix",
    "UniversalFixTemplate",
    "CrossDomainTranslator",
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeGraph",
    "KnowledgeAmalgamator",
    "PatternDiscoveryEngine",
    "DiscoveredPattern",
    "SimilarityLearner",
    "MigrationRecord",
    "SimilarityMatrix",
    "AbstractTemplateGenerator",
    "AbstractFixTemplate",
    "KnowledgeSelfEvolver",
    "EvolutionResult",
]