"""Self-Researcher 主编排器：把数据、统计、发现、图表与论文串成一条流水线。"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hermes.self_research.figure_generator import Figure, FigureGenerator
from hermes.self_research.finding_extractor import Finding, FindingExtractor
from hermes.self_research.latex_compiler import CompileResult, LatexCompiler
from hermes.self_research.paper_writer import Paper, PaperWriter
from hermes.self_research.research_data_collector import ResearchDataCollector, ResearchDataset
from hermes.self_research.statistical_analyzer import StatisticalAnalysis, StatisticalAnalyzer

FORMAT_EXTENSIONS = {"markdown": ".md", "latex": ".tex", "pdf": ".pdf"}


@dataclass
class SelfResearchConfig:
    output_dir: str = "papers"
    paper_path: Optional[str] = None
    formats: List[str] = field(default_factory=lambda: ["markdown"])
    prefer_png: bool = True
    compile_pdf: bool = False
    alpha: float = 0.05
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "paper_path": self.paper_path,
            "formats": list(self.formats),
            "prefer_png": self.prefer_png,
            "compile_pdf": self.compile_pdf,
            "alpha": self.alpha,
        }


@dataclass
class SelfResearchReport:
    title: str
    generated_at: str
    output_dir: str
    dataset_path: str
    analysis_path: str
    paper_paths: List[str] = field(default_factory=list)
    figure_paths: List[str] = field(default_factory=list)
    compile_results: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "output_dir": self.output_dir,
            "dataset_path": self.dataset_path,
            "analysis_path": self.analysis_path,
            "paper_paths": self.paper_paths,
            "figure_paths": self.figure_paths,
            "compile_results": self.compile_results,
            "findings": self.findings,
            "counts": self.counts,
            "duration_seconds": round(self.duration_seconds, 3),
        }


class SelfResearcher:
    """自动撰写关于自身演化的学术论文。"""

    VERSION = "1.0"

    def __init__(
        self,
        config: Optional[SelfResearchConfig] = None,
        collector: Optional[ResearchDataCollector] = None,
        analyzer: Optional[StatisticalAnalyzer] = None,
        extractor: Optional[FindingExtractor] = None,
        figure_generator: Optional[FigureGenerator] = None,
        writer: Optional[PaperWriter] = None,
        compiler: Optional[LatexCompiler] = None,
    ):
        self._config = config or SelfResearchConfig()
        self._collector = collector or ResearchDataCollector()
        self._analyzer = analyzer or StatisticalAnalyzer(alpha=self._config.alpha)
        self._extractor = extractor or FindingExtractor(alpha=self._config.alpha)
        self._figures = figure_generator or FigureGenerator(
            output_dir=os.path.join(self._config.output_dir, "figures"),
            prefer_png=self._config.prefer_png,
        )
        self._writer = writer or PaperWriter()
        self._compiler = compiler or LatexCompiler(output_dir=self._config.output_dir)

    def run(self) -> SelfResearchReport:
        started = time.perf_counter()

        dataset = self._collect()
        analysis = self._analyze(dataset)
        findings = self._extract(analysis, dataset)
        figures = self._figures_step(dataset, analysis)
        paper = self._write(dataset, analysis, findings, figures)
        compile_results = self._render(paper)

        dataset_path = self._save_json(os.path.join(self._config.output_dir, "dataset.json"), dataset.to_dict())
        analysis_path = self._save_json(
            os.path.join(self._config.output_dir, "analysis.json"),
            {"analysis": analysis.to_dict(), "findings": [item.to_dict() for item in findings]},
        )

        paper_paths = [result.output_path for result in compile_results]
        figure_paths = [figure.path for figure in figures if figure.path]

        report = SelfResearchReport(
            title=paper.title,
            generated_at=datetime.now(timezone.utc).isoformat(),
            output_dir=self._config.output_dir,
            dataset_path=dataset_path,
            analysis_path=analysis_path,
            paper_paths=paper_paths,
            figure_paths=figure_paths,
            compile_results=[result.to_dict() for result in compile_results],
            findings=[item.to_dict() for item in findings],
            counts=dataset.counts(),
            duration_seconds=time.perf_counter() - started,
        )

        self._log(
            f"[SelfResearcher] Paper generated: {paper_paths[0] if paper_paths else '(none)'} "
            f"({len(paper.sections)} sections, {len(figures)} figures, {len(paper.references)} references, "
            f"{report.duration_seconds:.2f}s)"
        )
        return report

    # ------------------------------------------------------------------ 各步骤

    def _collect(self) -> ResearchDataset:
        dataset = self._collector.collect()
        counts = dataset.counts()
        strategy_types = {item.get("strategy_type") for item in dataset.strategies}
        self._log(
            f"[SelfResearcher] Collecting data from {counts['snapshots']} snapshots, "
            f"{counts['strategies']} strategy observations over {len(strategy_types)} types, "
            f"{counts['phases']} phases..."
        )
        return dataset

    def _analyze(self, dataset: ResearchDataset) -> StatisticalAnalysis:
        self._log("[StatisticalAnalyzer] Computing metrics...")
        analysis = self._analyzer.analyze(dataset)

        trend = analysis.success_rate_trend
        if trend is not None:
            self._log(
                f"  - Success rate trend: {trend.total_change:+.3f} over {trend.n} snapshots "
                f"(slope={trend.slope:+.5f}, p={trend.p_value:.4f}"
                f"{', significant' if trend.significant else ''})"
            )
        if analysis.strategy_stats:
            top = analysis.strategy_stats[0]
            self._log(f"  - Top strategy: {top.strategy_type} ({top.avg_benefit:+.1%} avg gain, n={top.instances})")
        self._log(
            f"  - Mental model accuracy: {analysis.mental_model_accuracy:.0%} "
            f"({analysis.mental_model_samples} samples)"
        )
        return analysis

    def _extract(self, analysis: StatisticalAnalysis, dataset: ResearchDataset) -> List[Finding]:
        findings = self._extractor.extract(analysis, dataset)
        self._log(f"[FindingExtractor] Extracted {len(findings)} key findings:")
        for index, item in enumerate(findings, 1):
            self._log(f"  {index}. {item.statement}")
        return findings

    def _figures_step(self, dataset: ResearchDataset, analysis: StatisticalAnalysis) -> List[Figure]:
        figures = self._figures.generate_all(dataset, analysis)
        backend = "PNG" if any(figure.path for figure in figures) else "ascii"
        self._log(f"[FigureGenerator] Generated {len(figures)} figures ({backend}).")

        for figure in figures:
            if figure.path or not figure.ascii_art:
                continue
            path = os.path.join(self._figures.output_dir, f"{figure.name}.txt")
            self._save_text(path, figure.ascii_art)
            figure.path = path

        return figures

    def _write(
        self,
        dataset: ResearchDataset,
        analysis: StatisticalAnalysis,
        findings: List[Finding],
        figures: List[Figure],
    ) -> Paper:
        mode = "LLM" if self._writer.llm_enabled else "templates"
        self._log(f"[PaperWriter] Generating sections with {mode}...")
        return self._writer.write(
            dataset,
            analysis,
            [item.to_dict() for item in findings],
            [figure.to_dict() for figure in figures],
        )

    def _render(self, paper: Paper) -> List[CompileResult]:
        results: List[CompileResult] = []
        formats = list(self._config.formats) or ["markdown"]

        for index, fmt in enumerate(formats):
            if fmt == "pdf" or (self._config.compile_pdf and fmt == "latex"):
                target_format = "pdf"
            else:
                target_format = fmt

            target = self._target_path(target_format, index)
            result = self._compiler.render(paper, target, target_format)
            self._log(f"[LaTeXCompiler] Wrote {result.output_path}")
            if result.warnings:
                for warning in result.warnings:
                    self._log(f"  ! {warning.strip().splitlines()[0] if warning.strip() else ''}")
            results.append(result)

        return results

    def _target_path(self, fmt: str, index: int) -> Optional[str]:
        extension = FORMAT_EXTENSIONS.get(fmt, ".md")
        configured = self._config.paper_path

        if index == 0 and configured:
            if os.path.splitext(configured)[1].lower() == extension:
                return configured
            adjusted = os.path.splitext(configured)[0] + extension
            self._log(f"[SelfResearcher] Output extension adjusted to match format: {adjusted}")
            return adjusted

        return os.path.join(self._config.output_dir, f"paper{extension}")

    # ------------------------------------------------------------------ 工具

    def _log(self, message: str) -> None:
        if self._config.verbose:
            print(message)

    @staticmethod
    def _save_json(path: str, payload: Dict[str, Any]) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def _save_text(path: str, content: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path
