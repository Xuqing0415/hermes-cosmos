"""Self-Researcher：让系统自动撰写关于自身的学术论文。"""

from .figure_generator import Figure, FigureGenerator
from .finding_extractor import Finding, FindingExtractor
from .latex_compiler import CompileResult, LatexCompiler
from .paper_writer import Paper, PaperSection, PaperWriter
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
    "SelfResearchConfig",
    "SelfResearchReport",
    "SelfResearcher",
]
