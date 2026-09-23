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
from datetime import datetime, timedelta, timezone

import pytest

from hermes.cross_domain import EvolutionTracker, GitPhaseDetector
from hermes.cross_domain.git_phase_detector import CommitRecord, classify_commit
from hermes.self_research import (
    FALLBACK,
    REAL,
    SYNTHETIC,
    UNVERIFIED,
    ConfidenceScorer,
    DataProvenanceTagger,
    EvidenceGrade,
    EvidenceGrader,
    FigureGenerator,
    FindingExtractor,
    IntegrityChecker,
    LatexCompiler,
    PaperWriter,
    ResearchDataCollector,
    ResearchDataset,
    SelfResearchConfig,
    SelfResearcher,
    StatisticalAnalyzer,
)
from hermes.self_research.data_provenance import worst
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


def _synthetic_commits(subjects, gap_days: float = 0.0, mid_gap_days: float = 0.0):
    """按给定提交主题造一段确定的提交历史。"""

    commits = []
    moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    middle = len(subjects) // 2
    for index, subject in enumerate(subjects):
        moment = moment + timedelta(days=gap_days)
        if index == middle:
            moment = moment + timedelta(days=mid_gap_days)
        commits.append(
            CommitRecord(
                sha=f"{index:040d}",
                date=moment,
                subject=subject,
                category=classify_commit(subject),
            )
        )
    return commits


class _FrozenCollector:
    """返回固定数据集的采集器替身，用于端到端验证给定来源标注下的论文输出。"""

    def __init__(self, dataset: ResearchDataset):
        self._dataset = dataset

    def collect(self) -> ResearchDataset:
        return self._dataset


ALL_SOURCES = {
    "snapshot_db": True,
    "notification_log": True,
    "policy": True,
    "knowledge_graph": True,
    "git_history": True,
}


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


class TestGitPhaseDetector:
    def test_unavailable_repository_reports_no_phases(self):
        result = GitPhaseDetector(str(pathlib.Path(REPO_ROOT) / "no-such-repo")).detect()
        assert result.git_unavailable is True
        assert result.phases == []
        assert result.reason
        assert result.commit_count == 0
        assert result.to_dict()["phases"] == []

    def test_empty_history_reports_no_phases(self):
        result = GitPhaseDetector(".").detect_from_commits([])
        assert result.git_unavailable is True
        assert result.phases == []

    def test_missing_git_never_fabricates_stage_counts(self):
        """回归：旧实现凭空给出 12/15/10 次提交的三个阶段。"""

        result = GitPhaseDetector("/nonexistent-repository").detect()
        assert [phase.total_commits for phase in result.phases] == []
        assert result.commit_count == 0

    def test_phases_follow_real_theme_changes(self):
        subjects = ["feat: 初始化模块"] * 10 + ["fix: 修复CI门禁"] * 10 + ["feat: 跨领域知识蒸馏"] * 6
        result = GitPhaseDetector(".").detect_from_commits(_synthetic_commits(subjects))

        assert result.git_unavailable is False
        assert result.method == "git_log_theme_and_gap"
        assert result.boundaries == [10, 20]
        assert [phase.stage_id for phase in result.phases] == [1, 2, 3]
        assert [phase.name for phase in result.phases] == ["基础构建期", "CI与修复期", "智能扩展期"]
        assert [phase.total_commits for phase in result.phases] == [10, 10, 6]
        assert sum(phase.total_commits for phase in result.phases) == len(subjects)
        assert all(phase.category_share > 0 for phase in result.phases)
        assert result.phases[0].boundary_signal == "start"

    def test_time_gap_alone_does_not_split_a_single_theme(self):
        """同一主题的两段历史即使间隔很久也合并为一个阶段（不做无依据的切分）。"""

        subjects = ["docs: 补充说明"] * 20
        result = GitPhaseDetector(".").detect_from_commits(
            _synthetic_commits(subjects, gap_days=0.5, mid_gap_days=30.0)
        )
        assert len(result.phases) == 1
        assert result.phases[0].total_commits == 20
        assert result.boundaries == []

    def test_detection_is_deterministic(self):
        subjects = ["feat: 新模块"] * 8 + ["fix: 修复门禁"] * 8
        commits = _synthetic_commits(subjects)
        first = GitPhaseDetector(".").detect_from_commits(commits)
        second = GitPhaseDetector(".").detect_from_commits(list(commits))
        assert first.to_dict() == second.to_dict()

    def test_phase_count_is_bounded_and_complete(self):
        subjects = ["feat: 新模块" if index % 2 else "fix: 修复门禁" for index in range(80)]
        result = GitPhaseDetector(".").detect_from_commits(_synthetic_commits(subjects))
        assert 1 <= len(result.phases) <= 6
        assert sum(phase.total_commits for phase in result.phases) == 80
        assert [phase.stage_id for phase in result.phases] == list(range(1, len(result.phases) + 1))


class TestDataProvenance:
    def test_undeclared_sources_are_missing_and_fields_unverified(self):
        dataset = ResearchDataset(snapshots=[{"success_rate": 0.5}])
        report = DataProvenanceTagger().tag(dataset)

        assert report.provenance_of("snapshots") == UNVERIFIED
        assert report.git_unavailable is True
        assert set(report.missing_sources) == {
            "snapshot_db",
            "notification_log",
            "policy",
            "knowledge_graph",
            "git_history",
        }
        assert report.count_by_provenance()[UNVERIFIED] == 1

    def test_absent_data_is_not_called_verified_or_unverified(self):
        """没有数据时不做推断：来源保持 REAL，备注写明“无数据”。"""

        report = DataProvenanceTagger().tag(ResearchDataset(sources=dict(ALL_SOURCES)))
        assert report.provenance_of("phases") == REAL
        assert "无数据" in report.entries["phases"].detail
        assert report.missing_sources == []

    def test_stamped_fallback_is_preserved(self):
        dataset = ResearchDataset(
            snapshots=[{"success_rate": 0.5}],
            provenance={"snapshots": FALLBACK},
            provenance_notes={"snapshots": "回退分支合成"},
        )
        report = DataProvenanceTagger().tag(dataset)
        assert report.provenance_of("snapshots") == FALLBACK
        assert report.entries["snapshots"].detail == "回退分支合成"
        assert any("降级数据" in note for note in report.notes)

    def test_worst_picks_the_least_trustworthy_label(self):
        assert worst([REAL, FALLBACK]) == FALLBACK
        assert worst([FALLBACK, SYNTHETIC]) == SYNTHETIC
        assert worst([UNVERIFIED, SYNTHETIC]) == SYNTHETIC
        assert worst([REAL, UNVERIFIED]) == UNVERIFIED
        assert worst([]) == UNVERIFIED

    def test_collector_stamps_real_sources(self, work_dir):
        base = str(work_dir)
        _build_sources(base)
        dataset = ResearchDataCollector(base_dir=base, repo_path=base).collect()
        report = DataProvenanceTagger().tag(dataset)

        assert report.provenance_of("snapshots") == REAL
        assert report.provenance_of("strategies") == REAL
        assert report.provenance_of("phases") == REAL
        assert "无数据" in report.entries["phases"].detail
        assert "git_history" in report.missing_sources


class TestEvidenceGrader:
    _grader = EvidenceGrader()

    @staticmethod
    def _report(**stamps) -> object:
        return DataProvenanceTagger().tag(ResearchDataset(sources=dict(ALL_SOURCES), provenance=dict(stamps)))

    def _grade(self, stamps, **finding):
        payload = {
            "finding_id": "F-01",
            "statement": "结论",
            "data_fields": ["strategies"],
            "sample_size": 5,
            "required_sample": 3,
        }
        payload.update(finding)
        return self._grader.grade(payload, self._report(**stamps))

    def test_all_real_and_enough_samples_earns_grade_a(self):
        grade = self._grade({"strategies": REAL})
        assert grade.grade == "A"
        assert grade.provenance == REAL

    def test_fallback_data_downgrades_to_b(self):
        assert self._grade({"strategies": FALLBACK}).grade == "B"

    def test_unverified_data_downgrades_to_b(self):
        assert self._grade({"strategies": UNVERIFIED}).grade == "B"

    def test_synthetic_data_downgrades_to_c(self):
        assert self._grade({"strategies": SYNTHETIC}).grade == "C"

    def test_insufficient_sample_downgrades_to_c(self):
        grade = self._grade({"strategies": REAL}, sample_size=2)
        assert grade.grade == "C"
        assert any("样本量不足" in reason for reason in grade.reasons)

    def test_caveat_downgrades_to_b(self):
        grade = self._grade({"strategies": REAL}, caveats=["以事件序号计时，属估计值"])
        assert grade.grade == "B"
        assert any("估计值" in reason for reason in grade.reasons)

    def test_unknown_field_falls_back_to_unverified(self):
        grade = self._grade({}, data_fields=["unknown_field"])
        assert grade.provenance == UNVERIFIED
        assert grade.grade == "B"

    def test_worst_of_several_fields_wins(self):
        grade = self._grade({"strategies": REAL, "phases": SYNTHETIC}, data_fields=["strategies", "phases"])
        assert grade.provenance == SYNTHETIC
        assert grade.grade == "C"

    def test_summarise_counts_every_grade(self):
        grades = [
            self._grade({"strategies": REAL}),
            self._grade({"strategies": FALLBACK}),
            self._grade({"strategies": SYNTHETIC}),
        ]
        assert EvidenceGrader.summarise(grades) == {"A": 1, "B": 1, "C": 1}


class TestIntegrityChecker:
    _checker = IntegrityChecker()

    @staticmethod
    def _report(**stamps):
        return DataProvenanceTagger().tag(ResearchDataset(sources=dict(ALL_SOURCES), provenance=dict(stamps)))

    @staticmethod
    def _grade(grade: str, provenance: str, category: str, finding_id: str = "F-01") -> EvidenceGrade:
        return EvidenceGrade(
            finding_id=finding_id,
            statement="结论",
            grade=grade,
            provenance=provenance,
            reasons=["原因"],
            category=category,
        )

    def test_grade_a_findings_produce_no_disclaimer(self):
        report = self._checker.check([self._grade("A", REAL, "strategy")], self._report())
        assert report.disclaimers == []
        assert report.critical_findings == []
        assert report.critical_count == 0
        assert report.is_clean is True

    def test_downgraded_critical_category_is_flagged(self):
        report = self._checker.check([self._grade("B", FALLBACK, "evolution", "F-07")], self._report())
        assert report.critical_count == 1
        assert report.critical_findings == ["F-07"]
        assert report.disclaimers[0].severity == "critical"
        assert "待真实数据验证" in report.disclaimers[0].text
        assert "5 结果与分析" in report.disclaimers[0].target
        assert report.is_clean is False

    def test_downgraded_non_critical_category_is_only_a_caution(self):
        report = self._checker.check([self._grade("C", SYNTHETIC, "strategy")], self._report())
        assert report.disclaimers[0].severity == "caution"
        assert report.critical_count == 0

    def test_missing_source_produces_a_data_warning(self):
        dataset = ResearchDataset(sources={**ALL_SOURCES, "git_history": False})
        report = self._checker.check([], DataProvenanceTagger().tag(dataset))
        assert any("git 历史不可用" in warning for warning in report.data_warnings)
        assert report.is_clean is False

    def test_summary_line_mentions_counts(self):
        grades = [self._grade("B", FALLBACK, "evolution")]
        report = self._checker.check(grades, self._report())
        line = IntegrityChecker.summary_line(report)
        assert "1 条" in line


class TestConfidenceScorer:
    _scorer = ConfidenceScorer()

    @staticmethod
    def _report(**sources):
        return DataProvenanceTagger().tag(ResearchDataset(sources={**ALL_SOURCES, **sources}))

    @staticmethod
    def _grade(grade: str) -> EvidenceGrade:
        return EvidenceGrade(
            finding_id="F-01",
            statement="结论",
            grade=grade,
            provenance=REAL,
            reasons=[],
            category="strategy",
        )

    def test_all_real_with_full_coverage_scores_one(self):
        breakdown = self._scorer.score([self._grade("A")], self._report())
        assert breakdown.score == pytest.approx(1.0)
        assert breakdown.coverage == pytest.approx(1.0)
        assert breakdown.suggestions == []

    def test_grade_weights_are_averaged(self):
        breakdown = self._scorer.score([self._grade("A"), self._grade("B"), self._grade("C")], self._report())
        assert breakdown.evidence_score == pytest.approx((1.0 + 0.6 + 0.3) / 3)
        assert breakdown.score == pytest.approx(0.7 * (1.0 + 0.6 + 0.3) / 3 + 0.3)

    def test_missing_source_lowers_coverage_and_suggests_a_fix(self):
        breakdown = self._scorer.score([self._grade("A")], self._report(git_history=False))
        assert breakdown.coverage == pytest.approx(0.8)
        assert breakdown.score == pytest.approx(0.7 + 0.3 * 0.8)
        assert breakdown.missing_sources == ["git_history"]
        assert any("补齐数据源" in suggestion for suggestion in breakdown.suggestions)
        assert any("git" in suggestion for suggestion in breakdown.suggestions)

    def test_critical_downgrades_apply_a_penalty(self):
        grades = [self._grade("A"), self._grade("B")]
        without = self._scorer.score(grades, self._report(), critical_count=0)
        with_penalty = self._scorer.score(grades, self._report(), critical_count=2)
        assert with_penalty.score == pytest.approx(without.score - 0.1)

    def test_no_findings_scores_zero_evidence(self):
        breakdown = self._scorer.score([], self._report())
        assert breakdown.evidence_score == 0.0
        assert breakdown.score == pytest.approx(0.3)

    def test_score_never_leaves_the_unit_interval(self):
        breakdown = self._scorer.score([self._grade("C")], self._report(git_history=False), critical_count=99)
        assert 0.0 <= breakdown.score <= 1.0


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

    def test_pipeline_reports_grades_confidence_and_no_integrity_file_by_default(self, work_dir):
        base = str(work_dir)
        _build_sources(base)
        config = SelfResearchConfig(output_dir=str(work_dir / "out"), formats=["markdown"], verbose=False)
        collector = ResearchDataCollector(base_dir=base, repo_path=base)

        report = SelfResearcher(config=config, collector=collector).run()

        assert report.grades
        assert set(report.grade_mix) == {"A", "B", "C"}
        assert 0.0 <= report.confidence <= 1.0
        assert report.integrity_path == ""
        # git 历史不可用：论文不能声称通过完整性检查
        assert report.integrity_clean is False

        dataset_payload = json.loads(pathlib.Path(report.dataset_path).read_text(encoding="utf-8"))
        assert dataset_payload["provenance"]["phases"] == REAL
        assert "无数据" in dataset_payload["provenance_notes"]["phases"]
        assert dataset_payload["sources"]["git_history"] is False

    def test_fallback_phase_data_forces_a_disclaimer_in_the_paper(self, work_dir):
        """关键结论（演化阶段）依赖回退数据时，论文必须显式声明。"""

        dataset = _synthetic_dataset()
        dataset.phases = [
            {
                "stage_id": 1,
                "name": "基础构建期",
                "commit_range": "a..b",
                "dominant_types": ["foundation"],
                "total_commits": 6,
            },
            {
                "stage_id": 2,
                "name": "CI与修复期",
                "commit_range": "c..d",
                "dominant_types": ["ci_build"],
                "total_commits": 9,
            },
        ]
        dataset.provenance = {"phases": FALLBACK}
        dataset.provenance_notes = {"phases": "git 不可用，阶段来自回退分支"}
        dataset.sources = {**ALL_SOURCES, "git_history": False}

        config = SelfResearchConfig(
            output_dir=str(work_dir / "out"),
            formats=["markdown"],
            verbose=False,
            integrity_report=True,
        )
        report = SelfResearcher(config=config, collector=_FrozenCollector(dataset)).run()

        evolution = [item for item in report.grades if item["category"] == "evolution"]
        assert evolution
        assert all(item["grade"] != "A" for item in evolution)
        assert report.integrity_clean is False
        assert any(item["severity"] == "critical" for item in report.disclaimers)
        assert report.confidence < 1.0
        assert os.path.exists(report.integrity_path)

        markdown = pathlib.Path(report.paper_paths[0]).read_text(encoding="utf-8")
        assert "待真实数据验证" in markdown
        assert "置信度" in markdown
        assert "数据来源" in markdown

    def test_real_phase_data_is_graded_a_and_not_disclaimed(self, work_dir):
        dataset = _synthetic_dataset()
        dataset.phases = [
            {
                "stage_id": 1,
                "name": "基础构建期",
                "commit_range": "a..b",
                "dominant_types": ["foundation"],
                "total_commits": 6,
            },
        ]
        dataset.provenance = {"phases": REAL}
        dataset.sources = dict(ALL_SOURCES)

        config = SelfResearchConfig(output_dir=str(work_dir / "out"), formats=["markdown"], verbose=False)
        report = SelfResearcher(config=config, collector=_FrozenCollector(dataset)).run()

        evolution = [item for item in report.grades if item["category"] == "evolution"]
        assert evolution and evolution[0]["grade"] == "A"
        assert all(item["finding_ids"] != evolution[0]["finding_id"] for item in report.disclaimers)


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
        assert not (work_dir / "integrity.json").exists()
        assert "SelfResearcher" in capsys.readouterr().out

    def test_run_self_research_with_integrity_writes_the_report(self, work_dir, capsys):
        module = _import_autotestgen()
        target = work_dir / "paper.md"

        module.run_self_research(str(target), "markdown", str(work_dir), True)

        assert target.exists()
        integrity = json.loads((work_dir / "integrity.json").read_text(encoding="utf-8"))
        assert "confidence" in integrity
        assert "provenance" in integrity
        assert "grades" in integrity
        assert "disclaimers" in integrity
        assert "数据源" in integrity["summary"]

        output = capsys.readouterr().out
        assert "DataProvenance" in output
        assert "EvidenceGrader" in output
        assert "ConfidenceScorer" in output
