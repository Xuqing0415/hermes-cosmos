from .abstract_pattern_extractor import (
    AbstractPattern,
    AbstractPatternExtractor,
    AbstractPatternType,
)
from .abstract_template_generator import (
    AbstractFixTemplate,
    AbstractTemplateGenerator,
)
from .auto_correction_engine import (
    AutoCorrectionEngine,
    CorrectionAction,
    CorrectionReport,
)
from .capability_benchmark import (
    BenchmarkReport,
    BenchmarkResult,
    CapabilityBenchmark,
)
from .cognitive_loop_engine import (
    CognitiveLoopEngine,
    CognitiveLoopResult,
)
from .cognitive_planner import (
    CognitivePlan,
    CognitivePlanner,
)
from .cross_domain_translator import (
    CrossDomainTranslator,
    DomainFix,
    UniversalFixTemplate,
)
from .cross_entity_learner import (
    CrossEntityLearner,
    CrossEntityLearningResult,
)
from .dashboard_cli import (
    render_dashboard,
)
from .event_pattern_learner import (
    EventPatternLearner,
    ExternalPattern,
)
from .evolution_compare_engine import (
    ComparisonReport,
    EvolutionCompareEngine,
    StageComparison,
)
from .evolution_storyteller import (
    EvolutionStoryteller,
    StoryChapter,
)
from .evolution_tracker import (
    EvolutionSnapshot,
    EvolutionTracker,
)
from .experiment_designer import (
    ExperimentDesigner,
    TestCase,
    TestCaseDifficulty,
)
from .external_insight_mapper import (
    ExternalInsight,
    ExternalInsightMapper,
)
from .gap_analyzer import (
    GapAnalysis,
    GapAnalyzer,
    GapReport,
)
from .gap_calibrator import (
    CalibrationSuggestion,
    GapCalibrator,
)
from .git_phase_detector import (
    CommitRecord,
    EvolutionPhase,
    GitPhaseDetector,
    PhaseDetectionResult,
    classify_commit,
)
from .insight_extractor import (
    Insight,
    InsightExtractor,
)
from .knowledge_amalgamator import (
    KnowledgeAmalgamator,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
)
from .mental_model_trainer import (
    MentalModelTrainer,
    PredictionModel,
)
from .mental_simulator import (
    MentalSimulator,
    SimulationResult,
)
from .notification_dispatcher import (
    NotificationDispatcher,
    NotificationEvent,
    get_dispatcher,
    notify,
)
from .pattern_discovery import (
    DiscoveredPattern,
    PatternDiscoveryEngine,
)
from .pattern_similarity_engine import (
    PatternCluster,
    PatternSimilarityEngine,
    SimilarityMatch,
)
from .recursive_insight_injector import (
    CalibrationResult,
    RecursiveInsightInjector,
)
from .report_generator import (
    ReportGenerator,
)
from .report_interpreter import (
    ReportInterpreter,
    Signal,
    SignalReport,
)
from .repository_miner import (
    ExternalEvent,
    RepoMiningReport,
    RepositoryMiner,
)
from .root_cause_analyzer import (
    RootCause,
    RootCauseAnalyzer,
)
from .self_guided_evolver import (
    EvolutionTarget,
    SelfGuidedEvolver,
)
from .self_improvement_policy import (
    EvolutionPolicy,
    SelfImprovementPolicy,
)
from .self_improvement_proposal import (
    Proposal,
    SelfImprovementProposal,
)
from .self_repository_miner import (
    EvolutionStage,
    SelfRepositoryMiner,
)
from .similarity_learner import (
    MigrationRecord,
    SimilarityLearner,
    SimilarityMatrix,
)
from .strategy_evaluator import (
    StrategyEvaluation,
    StrategyEvaluator,
    StrategyReport,
)
from .strategy_sandbox import (
    SandboxReport,
    StrategySandbox,
)
from .trend_detector import (
    TrendAlert,
    TrendDetector,
    TrendReport,
)
from .value_discovery import (
    ValueDiscovery,
    ValuePrinciple,
)
from .value_tracker import (
    TrackedChange,
    ValueTracker,
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
    "EvolutionPolicy",
    "SelfImprovementPolicy",
    "TestCase",
    "TestCaseDifficulty",
    "ExperimentDesigner",
    "BenchmarkResult",
    "BenchmarkReport",
    "CapabilityBenchmark",
    "GapAnalysis",
    "GapReport",
    "GapAnalyzer",
    "EvolutionSnapshot",
    "EvolutionTracker",
    "TrendAlert",
    "TrendReport",
    "TrendDetector",
    "RootCause",
    "RootCauseAnalyzer",
    "CorrectionAction",
    "CorrectionReport",
    "AutoCorrectionEngine",
    "ReportGenerator",
    "render_dashboard",
    "NotificationEvent",
    "NotificationDispatcher",
    "get_dispatcher",
    "notify",
    "StoryChapter",
    "EvolutionStoryteller",
    "Signal",
    "SignalReport",
    "ReportInterpreter",
    "Insight",
    "InsightExtractor",
    "Proposal",
    "SelfImprovementProposal",
    "CognitiveLoopResult",
    "CognitiveLoopEngine",
    "TrackedChange",
    "ValueTracker",
    "StrategyEvaluation",
    "StrategyReport",
    "StrategyEvaluator",
    "ValuePrinciple",
    "ValueDiscovery",
    "EvolutionTarget",
    "SelfGuidedEvolver",
    "PredictionModel",
    "MentalModelTrainer",
    "SimulationResult",
    "MentalSimulator",
    "SandboxReport",
    "StrategySandbox",
    "CognitivePlan",
    "CognitivePlanner",
    "ExternalEvent",
    "RepoMiningReport",
    "RepositoryMiner",
    "ExternalPattern",
    "EventPatternLearner",
    "ExternalInsight",
    "ExternalInsightMapper",
    "CrossEntityLearningResult",
    "CrossEntityLearner",
    "EvolutionStage",
    "SelfRepositoryMiner",
    "CommitRecord",
    "EvolutionPhase",
    "GitPhaseDetector",
    "PhaseDetectionResult",
    "classify_commit",
    "StageComparison",
    "ComparisonReport",
    "EvolutionCompareEngine",
    "CalibrationSuggestion",
    "GapCalibrator",
    "CalibrationResult",
    "RecursiveInsightInjector",
]
