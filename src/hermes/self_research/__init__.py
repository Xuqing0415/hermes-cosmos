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
from .import_fix_benchmark import ImportFixReport, PairOutcome, evaluate_repository
from .import_fix_miner import ImportFixCase, MineReport, import_bindings, mine_repository
from .integrity_checker import Disclaimer, IntegrityChecker, IntegrityReport
from .latex_compiler import CompileResult, LatexCompiler
from .paper_revision_engine import (
    PaperRevisionEngine,
    Revision,
    RevisionResult,
)
from .paper_writer import Paper, PaperSection, PaperWriter
from .real_defect_benchmark import (
    BenchmarkReport,
    CaseResult,
    DefectCase,
    DefectSampler,
    GitRepo,
    RealDefectBenchmark,
    TestOutcome,
)
from .rebuttal_generator import (
    STRENGTH_LABELS,
    STRENGTH_NONE,
    STRENGTH_STRONG,
    STRENGTH_WEAK,
    Rebuttal,
    RebuttalGenerator,
)
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
from .success_claim_auditor import (
    ClaimAuditReport,
    ClaimCriterion,
    SuccessClaim,
    SuccessClaimAuditor,
)
from .unused_import_benchmark import RemovalOutcome, UnusedBenchmarkReport
from .unused_import_miner import RemovedImportCase, UnusedImportReport, dunder_all_names, unused_import_names
from .weakness_ranker import RankedWeakness, WeaknessRanker, WeaknessRanking

__all__ = [
    "ResearchDataset",
    "ResearchDataCollector",
    "StrategyObservation",
    "SimilarityObservation",
    "RealDefectBenchmark",
    "BenchmarkReport",
    "CaseResult",
    "DefectCase",
    "DefectSampler",
    "GitRepo",
    "TestOutcome",
    "DescriptiveStats",
    "TrendResult",
    "CorrelationResult",
    "StrategyStat",
    "ComparisonResult",
    "StatisticalAnalysis",
    "StatisticalAnalyzer",
    "Finding",
    "FindingExtractor",
    "ImportFixCase",
    "ImportFixReport",
    "MineReport",
    "PairOutcome",
    "evaluate_repository",
    "import_bindings",
    "mine_repository",
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
    "ClaimAuditReport",
    "ClaimCriterion",
    "SuccessClaim",
    "SuccessClaimAuditor",
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
    "STRENGTH_NONE",
    "STRENGTH_STRONG",
    "STRENGTH_WEAK",
    "STRENGTH_LABELS",
    "PaperRevisionEngine",
    "Revision",
    "RevisionResult",
    "RemovedImportCase",
    "UnusedImportReport",
    "UnusedBenchmarkReport",
    "RemovalOutcome",
    "dunder_all_names",
    "unused_import_names",
]
