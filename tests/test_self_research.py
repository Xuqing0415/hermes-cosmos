"""Self-Researcher（方向 A）的单元测试与端到端测试。

覆盖范围：统计函数、ASCII 图表、发现提取规则、Markdown/LaTeX 渲染，
以及 SelfResearcher 流水线与 CLI 入口。
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import uuid

import pytest

from hermes.cross_domain import EvolutionTracker
from hermes.self_research import (
    FigureGenerator,
    FindingExtractor,
    LatexCompiler,
    PaperWriter,
    ResearchDataCollector,
    ResearchDataset,
    SelfResearchConfig,
    SelfResearcher,
    StatisticalAnalyzer,
)
from hermes.self_research.figure_generator import ascii_bar_chart, ascii_line_chart, ascii_scatter
from hermes.self_research.latex_compiler import latex_escape
from hermes.self_research.statistical_analyzer import (
    describe,
    linear_trend,
    pearson_correlation,
    student_t_two_sided_p,
    welch_t_test,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def work_dir():
    """临时工作目录。

    不用 pytest 的 tmp_path：受限环境下系统临时目录不可写。
    统一放在仓库内的 .tmp_test/self_research 下，用例结束后清理。
    """

    base = os.path.join(REPO_ROOT, ".tmp_test", "self_research")
    path = pathlib.Path(os.path.join(base, "run_" + uuid.uuid4().hex))
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _import_autotestgen():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import autotestgen

    return autotestgen


def _synthetic_dataset() -> ResearchDataset:
    """构造一份小而完整的数据集，用于验证统计与写作逻辑。"""

    snapshots = [
        {
            "id": index,
            "snapshot_index": index,
            "success_rate": 0.60 + 0.02 * index,
            "cross_domain_success": 0.40 + 0.01 * index,
            "pattern_coverage": 0.60 + 0.005 * index,
            "metadata": {"pattern_success_rates": {"boundary_check_missing": 0.8}} if index == 10 else {},
        }
        for index in range(1, 11)
    ]

    return ResearchDataset(
        generated_at="2026-01-01T00:00:00+00:00",
        snapshots=snapshots,
        strategies=[
            {"strategy_type": "reset_similarity", "target": "boundary", "benefit": 0.12},
            {"strategy_type": "reset_similarity", "target": "boundary", "benefit": 0.11},
            {"strategy_type": "reset_similarity", "target": "boundary", "benefit": 0.10},
            {"strategy_type": "increase_weight", "target": "boundary", "benefit": 0.01},
        ],
        similarity_pairs=[
            {
                "source_pattern": "k8s.a",
                "target_pattern": "default.b",
                "similarity": 0.80,
                "success_rate": 0.85,
                "domain": "k8s+default",
            },
            {
                "source_pattern": "k8s.b",
                "target_pattern": "default.c",
                "similarity": 0.75,
                "success_rate": 0.90,
                "domain": "k8s+default",
            },
            {
                "source_pattern": "k8s.c",
                "target_pattern": "default.d",
                "similarity": 0.85,
                "success_rate": 0.83,
                "domain": "k8s+default",
            },
        ],
        phases=[
            {
                "stage_id": 1,
                "name": "foundation",
                "commit_range": "a..b",
                "dominant_types": ["foundation"],
                "total_commits": 5,
            }
        ],
        policy={"update_count": 3, "mutation_rate": 0.18},
        mental_model={"accuracy": 0.8, "training_samples": 10},
        sources={"snapshot_db": True},
    )


def _build_sources(base: str) -> None:
    """在 base 目录下造出四个数据源文件，模拟真实的 loop_output 产物。"""

    loop_dir = os.path.join(base, "loop_output")
    os.makedirs(loop_dir, exist_ok=True)

    tracker = EvolutionTracker(os.path.join(loop_dir, "evolution_tracker.db"))
    tracker.record_snapshot(0.70, 0.50, 0.60, metadata={"pattern_success_rates": {"boundary_check_missing": 0.6}})
    tracker.record_snapshot(
        0.82,
        0.60,
        0.75,
        metadata={"corrected": True, "pattern_success_rates": {"boundary_check_missing": 0.85}},
    )

    events = [
        {
            "event_type": "external_insight_applied",
            "message": "提升模式权重",
            "metadata": {"old_weight": 0.1, "new_weight": 0.2, "target": "boundary_check_missing"},
        },
        {
            "event_type": "evolution_target_selected",
            "message": "选择演化目标 (预期收益: 12%)",
            "metadata": {"target": "boundary_check_missing"},
        },
        {
            "event_type": "proposal_applied",
            "message": "应用提案 (预期收益: 40%)",
            "metadata": {"target": "unknown"},
        },
    ]
    with open(os.path.join(loop_dir, "notifications.log"), "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    graph = {
        "nodes": [
            {"id": "knode-0001", "pattern_type": "resource_limit_missing", "domains": ["k8s"]},
            {"id": "knode-0002", "pattern_type": "boundary_check_missing", "domains": ["default"]},
        ],
        "edges": [
            {
                "source_id": "knode-0001",
                "target_id": "knode-0002",
                "similarity": 0.55,
                "domains": ["k8s", "default"],
            }
        ],
    }
    with open(os.path.join(loop_dir, "knowledge_graph.json"), "w", encoding="utf-8") as handle:
        json.dump(graph, handle, ensure_ascii=False)

    policy = {
        "pattern_weights": {"boundary_check_missing": 0.5},
        "mutation_rate": 0.18,
        "update_count": 3,
    }
    with open(os.path.join(loop_dir, "self_improvement_policy.json"), "w", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False)


class TestStatisticalAnalyzer:
    def test_describe_basics(self):
        stats = describe([1.0, 2.0, 3.0, 4.0])
        assert stats.count == 4
        assert stats.mean == pytest.approx(2.5)
        assert stats.minimum == 1.0
        assert stats.maximum == 4.0

    def test_describe_empty_series(self):
        assert describe([]).count == 0

    def test_linear_trend_detects_rising_series(self):
        trend = linear_trend([float(index) for index in range(10)])
        assert trend.slope == pytest.approx(1.0)
        assert trend.direction == "up"
        assert trend.significant is True
        assert trend.total_change == pytest.approx(9.0)

    def test_linear_trend_flat_series_is_not_significant(self):
        trend = linear_trend([0.5] * 6)
        assert trend.slope == pytest.approx(0.0)
        assert trend.direction == "flat"
        assert trend.significant is False

    def test_linear_trend_requires_three_points(self):
        trend = linear_trend([0.1, 0.2])
        assert trend.n == 2
        assert trend.p_value == 1.0
        assert trend.significant is False

    def test_pearson_correlation_perfect_positive(self):
        result = pearson_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        assert result.r == pytest.approx(1.0)
        assert result.significant is True

    def test_pearson_correlation_needs_three_pairs(self):
        result = pearson_correlation([1.0, 2.0], [2.0, 4.0])
        assert result.n == 2
        assert result.significant is False

    def test_welch_t_test_identical_samples(self):
        t_statistic, p_value = welch_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert t_statistic == pytest.approx(0.0)
        assert p_value == pytest.approx(1.0)

    def test_welch_t_test_separated_samples(self):
        t_statistic, p_value = welch_t_test([1.0, 1.1, 0.9], [5.0, 5.1, 4.9])
        assert abs(t_statistic) > 10
        assert p_value < 0.05

    def test_student_t_p_value_stays_in_unit_interval(self):
        for t_statistic in (-8.0, -1.0, 0.0, 1.0, 8.0):
            p_value = student_t_two_sided_p(t_statistic, 8)
            assert 0.0 <= p_value <= 1.0
        assert student_t_two_sided_p(0.0, 8) == pytest.approx(1.0)

    def test_analyze_summarizes_dataset(self):
        analysis = StatisticalAnalyzer().analyze(_synthetic_dataset())
        assert analysis.best_strategy == "reset_similarity"
        assert analysis.worst_strategy == "increase_weight"
        assert analysis.mental_model_accuracy == pytest.approx(0.8)
        assert analysis.mental_model_samples == 10
        assert analysis.success_rate.count == 10
        assert analysis.similarity_correlation is not None
        assert analysis.similarity_correlation.n == 3


class TestAsciiFigures:
    def test_line_chart_renders_markers(self):
        art = ascii_line_chart([(1, 0.1), (2, 0.5), (3, 0.9)], title="demo")
        assert "*" in art
        assert "demo" in art

    def test_line_chart_without_data(self):
        assert ascii_line_chart([]) == "（无数据）"

    def test_bar_chart_without_data(self):
        assert ascii_bar_chart([]) == "（无数据）"

    def test_bar_chart_draws_value_labels(self):
        art = ascii_bar_chart([("phase-1", 4.0), ("phase-2", 8.0)])
        assert "+4.000" in art
        assert "+8.000" in art

    def test_scatter_plots_points(self):
        art = ascii_scatter([(0.5, 0.6), (0.8, 0.9)], x_label="相似度", y_label="成功率")
        assert "*" in art

    def test_scatter_degenerate_range_falls_back_to_listing(self):
        art = ascii_scatter([(0.55, 0.65)], x_label="相似度", y_label="成功率", title="散点")
        assert "退化为单点" in art

    def test_generate_all_returns_three_figures(self):
        dataset = _synthetic_dataset()
        analysis = StatisticalAnalyzer().analyze(dataset)
        figures = FigureGenerator(prefer_png=False).generate_all(dataset, analysis)
        assert [figure.name for figure in figures] == [
            "success_rate_curve",
            "phase_comparison",
            "similarity_success_scatter",
        ]
        assert all(figure.ascii_art for figure in figures)


class TestFindingExtractor:
    def test_extracts_significant_strategy(self):
        dataset = _synthetic_dataset()
        analysis = StatisticalAnalyzer().analyze(dataset)
        findings = FindingExtractor().extract(analysis, dataset)
        strategies = [item for item in findings if item.category == "strategy"]
        assert any("reset_similarity" in item.statement and item.significant for item in strategies)
        assert all(item.finding_id.startswith("F-") for item in findings)

    def test_reports_strong_transfer_relation(self):
        dataset = _synthetic_dataset()
        findings = FindingExtractor().extract(StatisticalAnalyzer().analyze(dataset), dataset)
        assert any(item.metric == "transfer_strength" for item in findings)

    def test_reports_insufficient_transfer_sample(self):
        dataset = _synthetic_dataset()
        dataset.similarity_pairs = [
            {
                "source_pattern": "k8s.x",
                "target_pattern": "default.y",
                "similarity": 0.55,
                "success_rate": 0.60,
                "domain": "k8s+default",
            }
        ]
        findings = FindingExtractor().extract(StatisticalAnalyzer().analyze(dataset), dataset)
        assert any(item.metric == "similarity_pairs" for item in findings)
        assert not any(item.metric == "transfer_strength" for item in findings)

    def test_empty_dataset_yields_no_findings(self):
        empty = ResearchDataset()
        findings = FindingExtractor().extract(StatisticalAnalyzer().analyze(empty), empty)
        assert findings == []


class TestPaperRendering:
    @staticmethod
    def _paper():
        dataset = _synthetic_dataset()
        analysis = StatisticalAnalyzer().analyze(dataset)
        findings = FindingExtractor().extract(analysis, dataset)
        figures = FigureGenerator(prefer_png=False).generate_all(dataset, analysis)
        return PaperWriter().write(
            dataset,
            analysis,
            [item.to_dict() for item in findings],
            [figure.to_dict() for figure in figures],
        )

    def test_paper_has_expected_sections_and_references(self):
        paper = self._paper()
        headings = [section.heading for section in paper.sections]
        for title in ("引言", "相关工作", "系统设计", "实验设置", "结果与分析", "讨论", "结论"):
            assert any(title in heading for heading in headings)
        assert any("第一人称" in heading for heading in headings)
        assert any(section.kind == "appendix" for section in paper.sections)
        assert len(paper.references) >= 10
        assert paper.keywords

    def test_markdown_render_has_title_keywords_and_fences(self):
        rendered = LatexCompiler().to_markdown(self._paper())
        assert rendered.startswith("# ")
        assert "关键词" in rendered
        assert "```" in rendered

    def test_latex_render_is_a_document(self):
        rendered = LatexCompiler().to_latex(self._paper())
        assert "\\documentclass" in rendered
        assert "\\begin{abstract}" in rendered
        assert rendered.count("\\begin{verbatim}") == rendered.count("\\end{verbatim}")
        assert "\\includegraphics" not in rendered

    def test_latex_escape_covers_special_characters(self):
        escaped = latex_escape("a_b % c & d # e")
        for token in ("\\_", "\\%", "\\&", "\\#"):
            assert token in escaped

    def test_resolve_format_infers_from_extension(self):
        assert LatexCompiler.resolve_format(None, "auto") == "markdown"
        assert LatexCompiler.resolve_format("paper.tex", "auto") == "latex"
        assert LatexCompiler.resolve_format("paper.pdf", "auto") == "pdf"
        assert LatexCompiler.resolve_format("paper.md", "latex") == "latex"


class TestResearchDataCollector:
    def test_collects_all_available_sources(self, work_dir):
        base = str(work_dir)
        _build_sources(base)

        dataset = ResearchDataCollector(base_dir=base, repo_path=base).collect()

        assert dataset.counts()["snapshots"] == 2
        assert dataset.counts()["strategies"] == 3
        assert dataset.counts()["similarity_pairs"] == 1
        assert dataset.pattern_success_rates["boundary_check_missing"] == pytest.approx(0.85)
        assert dataset.sources["snapshot_db"] is True
        assert dataset.sources["git_history"] is False

    def test_missing_sources_degrade_gracefully(self, work_dir):
        base = str(work_dir / "nothing-here")
        dataset = ResearchDataCollector(base_dir=base, repo_path=base).collect()

        counts = dataset.counts()
        assert counts["snapshots"] == 0
        assert counts["strategies"] == 0
        assert counts["events"] == 0
        assert counts["similarity_pairs"] == 0
        for source in ("snapshot_db", "policy", "notification_log", "knowledge_graph", "git_history"):
            assert dataset.sources[source] is False


class TestSelfResearcherPipeline:
    def test_empty_environment_still_writes_a_paper(self, work_dir):
        empty = str(work_dir / "empty")
        output_dir = work_dir / "out"
        config = SelfResearchConfig(output_dir=str(output_dir), formats=["markdown"], verbose=False)
        collector = ResearchDataCollector(base_dir=empty, repo_path=empty)

        report = SelfResearcher(config=config, collector=collector).run()

        assert report.paper_paths
        assert os.path.exists(report.paper_paths[0])
        assert os.path.exists(report.dataset_path)
        assert os.path.exists(report.analysis_path)
        assert report.counts["snapshots"] == 0
        assert report.counts["strategies"] == 0
        assert not any(finding["category"] == "strategy" for finding in report.findings)

    def test_pipeline_with_synthetic_sources_writes_both_formats(self, work_dir):
        base = str(work_dir)
        _build_sources(base)
        config = SelfResearchConfig(
            output_dir=str(work_dir / "out"),
            formats=["markdown", "latex"],
            verbose=False,
        )
        collector = ResearchDataCollector(base_dir=base, repo_path=base)

        report = SelfResearcher(config=config, collector=collector).run()

        assert report.counts["snapshots"] == 2
        assert report.findings
        assert len(report.figure_paths) == 3
        assert {os.path.splitext(path)[1] for path in report.paper_paths} == {".md", ".tex"}
        for path in report.paper_paths:
            assert os.path.exists(path)


class TestSelfResearchCLI:
    def test_resolve_paper_formats(self):
        module = _import_autotestgen()
        assert module._resolve_paper_formats("both", None) == ["markdown", "latex"]
        assert module._resolve_paper_formats("pdf", None) == ["pdf"]
        assert module._resolve_paper_formats("auto", "paper.tex") == ["latex"]
        assert module._resolve_paper_formats("auto", None) == ["markdown"]

    def test_run_self_research_writes_artifacts(self, work_dir, capsys):
        module = _import_autotestgen()
        target = work_dir / "paper.md"

        module.run_self_research(str(target), "markdown", str(work_dir))

        assert target.exists()
        assert (work_dir / "dataset.json").exists()
        assert (work_dir / "analysis.json").exists()
        assert "SelfResearcher" in capsys.readouterr().out
