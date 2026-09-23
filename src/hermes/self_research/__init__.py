"""Self-Researcher：让系统自动撰写关于自身的学术论文。"""

from hermes.cross_domain.git_phase_detector import (
    CommitRecord,
    EvolutionPhase,
    GitPhaseDetector,
    PhaseDetectionResult,
    classify_commit,
)

from .adversarial_reviewer import (
    ATTACK_LABELS,
    FATAL,
    MAJOR,
    MINOR,
    SEVERITY_LABELS,
    AdversarialReviewer,
    Attack,
    AuditReport,
    ReviewContext,
    ReviewContextLoader,
)
from .confidence_scorer import ConfidenceBreakdown, ConfidenceScorer
from .data_provenance import (
    ALL_PROVENANCE,
    FALLBACK,
    PROVENANCE_LABELS,
    REAL,
    SYNTHETIC,
    UNVERIFIED,
    DataProvenanceTagger,
    ProvenanceEntry,
    ProvenanceReport,
)
from .evidence_grader import EvidenceGrade, EvidenceGrader
from .figure_generator import Figure, FigureGenerator
from .finding_extractor import Finding, FindingExtractor
from .integrity_checker import Disclaimer, IntegrityChecker, IntegrityReport
from .latex_compiler import CompileResult, LatexCompiler
from .paper_revision_engine import (
    PaperRevisionEngine,
    Revision,
    RevisionResult,
)
from .paper_writer import Paper, PaperSection, PaperWriter
from .rebuttal_generator import Rebuttal, RebuttalGenerator
from .research_data_collector import (
    ResearchDataCollector,
    ResearchDataset,
    SimilarityObservation,
    StrategyObservation,
)
from .self_researcher import SelfResearchConfig, SelfResearcher, SelfResearchReport
from .statistical_analyzer import (
    ComparisonResult,
    CorrelationResult,
    DescriptiveStats,
    StatisticalAnalysis,
    StatisticalAnalyzer,
    StrategyStat,
    TrendResult,
)
from .weakness_ranker import RankedWeakness, WeaknessRanker, WeaknessRanking

__all__ = [
    "ResearchDataset",
    "ResearchDataCollector",
    "StrategyObservation",
    "SimilarityObservation",
    "DescriptiveStats",
    "TrendResult",
    "CorrelationResult",
    "StrategyStat",
    "ComparisonResult",
    "StatisticalAnalysis",
    "StatisticalAnalyzer",
    "Finding",
    "FindingExtractor",
    "Figure",
    "FigureGenerator",
    "Paper",
    "PaperSection",
    "PaperWriter",
    "CompileResult",
    "LatexCompiler",
    "REAL",
    "FALLBACK",
    "SYNTHETIC",
    "UNVERIFIED",
    "ALL_PROVENANCE",
    "PROVENANCE_LABELS",
    "DataProvenanceTagger",
    "ProvenanceEntry",
    "ProvenanceReport",
    "EvidenceGrade",
    "EvidenceGrader",
    "Disclaimer",
    "IntegrityChecker",
    "IntegrityReport",
    "ConfidenceBreakdown",
    "ConfidenceScorer",
    "CommitRecord",
    "EvolutionPhase",
    "GitPhaseDetector",
    "PhaseDetectionResult",
    "classify_commit",
    "SelfResearchConfig",
    "SelfResearchReport",
    "SelfResearcher",
    "AdversarialReviewer",
    "Attack",
    "AuditReport",
    "ReviewContext",
    "ReviewContextLoader",
    "FATAL",
    "MAJOR",
    "MINOR",
    "ATTACK_LABELS",
    "SEVERITY_LABELS",
    "WeaknessRanker",
    "RankedWeakness",
    "WeaknessRanking",
    "RebuttalGenerator",
    "Rebuttal",
    "PaperRevisionEngine",
    "Revision",
    "RevisionResult",
]
