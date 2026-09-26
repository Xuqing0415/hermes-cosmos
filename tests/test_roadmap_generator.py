"""进化路线图生成器（``RoadmapGenerator``）的单元测试。

覆盖三件事：空数据不崩且说明原因、已知输入下的排序与预算分配、以及「收益 / 人天 / ROI」
确实是按输入算出来的（不是硬编码常量）。

注意：``generate(None)`` 会去真的扫描仓库、拉取趋势（受限环境里就是等网络超时），
所以这里一律喂一个「所有数据源都命中但都为空」的 ``EMPTY_INPUT``：既走完了分支，
又让提取结果确定为空，从而稳定地落到默认候选池。
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import uuid

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hermes.roadmap_generator import (  # noqa: E402
    FeatureCandidate,
    FeatureCategory,
    FeatureSource,
    ImpactEstimator,
    ImpactScore,
    MultiObjectiveOptimizer,
    RoadmapConfig,
    RoadmapGenerator,
    WorkloadEstimate,
    WorkloadEstimator,
)

EMPTY_INPUT = {
    "audit_result": {"issues": []},
    "trend_result": {"top_recommendations": []},
    "smells": [],
    "task_stats": {},
    "proof_report": {},
}


def _candidate(title, category, priority_score, source=FeatureSource.SELF_AUDIT, tags=(), severity=None):
    metadata = {"tags": list(tags)}
    if severity is not None:
        metadata["severity"] = severity
    return FeatureCandidate(
        title=title,
        description="",
        source=source,
        category=category,
        priority_score=priority_score,
        metadata=metadata,
    )


@pytest.fixture
def work_dir():
    """临时工作目录（不用 tmp_path：受限环境下系统临时目录不可写）。"""

    base = os.path.join(REPO_ROOT, ".tmp_test", "roadmap_generator")
    path = pathlib.Path(os.path.join(base, "run_" + uuid.uuid4().hex))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _generator(work_dir, **overrides):
    settings = {"output_dir": str(work_dir), "generate_md": False}
    settings.update(overrides)
    return RoadmapGenerator(RoadmapConfig(**settings))


class TestEmptyInput:
    def test_no_candidates_falls_back_to_defaults_and_says_why(self, work_dir, capsys):
        report = _generator(work_dir).generate(EMPTY_INPUT)

        assert "未提取到候选特性，使用默认数据" in capsys.readouterr().out
        assert len(report.features) > 0

    def test_report_with_no_features_does_not_divide_by_zero(self, work_dir):
        report = _generator(work_dir)._create_report([], [], 0.0)

        assert report.summary["total_features"] == 0
        assert report.summary["avg_impact"] == 0
        assert report.summary["total_workload"] == 0
        assert report.recommendations == []

    def test_estimators_return_empty_mappings_for_no_candidates(self):
        assert ImpactEstimator().estimate([]) == {}
        assert WorkloadEstimator().estimate([]) == {}

    def test_optimizer_returns_empty_structures_for_no_candidates(self):
        optimizer = MultiObjectiveOptimizer()

        assert optimizer.prioritize([], {}, {}) == []
        assert optimizer.select_budget_portfolio([], {}, {}) == ([], 0.0)


class TestKnownInputsProduceExpectedOutput:
    def test_features_are_sorted_by_overall_score(self, work_dir):
        report = _generator(work_dir).generate(EMPTY_INPUT)

        scores = [feature.overall_score for feature in report.features]
        assert scores == sorted(scores, reverse=True)

    def test_portfolio_respects_the_budget(self, work_dir):
        report = _generator(work_dir, budget_days=8).generate(EMPTY_INPUT)

        assert 0 < report.summary["total_workload"] <= 8

    def test_budget_allocation_covers_exactly_the_portfolio(self, work_dir):
        report = _generator(work_dir, budget_days=8).generate(EMPTY_INPUT)

        allocated = sum(report.budget_allocation.values())
        assert allocated == pytest.approx(report.summary["total_workload"], abs=0.5)
        assert set(report.budget_allocation) <= {feature.candidate.category.value for feature in report.features}

    def test_markdown_report_lands_in_the_configured_directory(self, work_dir):
        report = _generator(work_dir, generate_md=True).generate(EMPTY_INPUT)

        latest = work_dir / "roadmap.md"
        timestamped = list(work_dir.glob("roadmap_*.md"))

        assert latest.exists()
        assert len(timestamped) == 1
        assert timestamped[0].read_text(encoding="utf-8") == latest.read_text(encoding="utf-8")

        text = latest.read_text(encoding="utf-8")
        assert report.version in text
        assert "路线图" in text

    def test_json_dump_round_trips_the_report(self, work_dir):
        generator = _generator(work_dir)
        report = generator.generate(EMPTY_INPUT)

        payload = json.loads(pathlib.Path(generator.save_json(report)).read_text(encoding="utf-8"))

        assert payload["version"] == report.version
        assert len(payload["features"]) == len(report.features)
        assert payload["features"][0]["workload"]["estimate"] > 0


class TestFieldsAreDerivedNotHardcoded:
    def test_every_feature_has_non_zero_derived_numbers(self, work_dir):
        report = _generator(work_dir).generate(EMPTY_INPUT)

        assert report.features
        for feature in report.features:
            assert feature.impact.overall > 0
            assert feature.workload.estimate > 0
            assert feature.overall_score > 0
            assert feature.roi > 0
            assert feature.risk_level in {"low", "medium", "high", "critical"}

    def test_roi_is_impact_over_workload_and_moves_with_the_inputs(self):
        optimizer = MultiObjectiveOptimizer()
        cheap = WorkloadEstimate(estimate=2.0, min_estimate=1.0, max_estimate=3.0, confidence=0.6)
        pricey = WorkloadEstimate(estimate=8.0, min_estimate=6.0, max_estimate=10.0, confidence=0.6)
        impact = ImpactScore(overall=50.0, confidence=0.6)

        roi_cheap = optimizer._calculate_roi(impact, cheap)
        roi_pricey = optimizer._calculate_roi(impact, pricey)

        assert roi_cheap > roi_pricey
        assert roi_cheap == pytest.approx(50.0 / 2.0 * 1.6)

    def test_roi_is_zero_when_the_workload_is_zero(self):
        roi = MultiObjectiveOptimizer()._calculate_roi(ImpactScore(overall=50.0), WorkloadEstimate(estimate=0.0))

        assert roi == 0.0

    def test_impact_grows_with_priority_score(self):
        estimator = ImpactEstimator()
        low = _candidate("低", FeatureCategory.PERFORMANCE, 30.0)
        high = _candidate("高", FeatureCategory.PERFORMANCE, 85.0)

        assert estimator._estimate_single(high).overall > estimator._estimate_single(low).overall

    def test_security_impact_grows_with_severity_metadata(self):
        estimator = ImpactEstimator()

        def impact(severity):
            candidate = _candidate(
                "安全", FeatureCategory.SECURITY, 70.0, source=FeatureSource.PROOF_FAILURE, severity=severity
            )
            return estimator._estimate_single(candidate).overall

        assert impact("critical") > impact("high") > impact("medium")

    def test_workload_depends_on_category_and_complexity_tags(self):
        estimator = WorkloadEstimator()

        plain = estimator._estimate_single(_candidate("x", FeatureCategory.SECURITY, 50.0))
        complex_ = estimator._estimate_single(_candidate("x", FeatureCategory.SECURITY, 50.0, tags=["rust"]))
        heavier_category = estimator._estimate_single(_candidate("x", FeatureCategory.TECH_DEBT, 50.0))

        assert complex_.estimate > plain.estimate
        assert heavier_category.estimate > plain.estimate
        assert plain.min_estimate < plain.estimate < plain.max_estimate
        assert plain.unit.value == "person_days"
