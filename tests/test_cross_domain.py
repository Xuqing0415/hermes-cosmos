import os
import tempfile

import pytest

from hermes.cross_domain import (
    AbstractPatternExtractor,
    AbstractPatternType,
    CognitivePlanner,
    CrossDomainTranslator,
    CrossEntityLearner,
    EventPatternLearner,
    EvolutionCompareEngine,
    EvolutionStage,
    ExternalInsightMapper,
    GapCalibrator,
    KnowledgeAmalgamator,
    MentalModelTrainer,
    MentalSimulator,
    PatternSimilarityEngine,
    RecursiveInsightInjector,
    RepositoryMiner,
    SelfGuidedEvolver,
    SelfRepositoryMiner,
    StrategyEvaluator,
    StrategySandbox,
    TrackedChange,
    ValueDiscovery,
    ValueTracker,
)
from hermes.cross_domain.git_phase_detector import CATEGORY_LABELS
from hermes.cross_domain.mental_model_trainer import PredictionModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestAbstractPatternExtractor:
    def test_extract_pattern_from_mlir(self):
        extractor = AbstractPatternExtractor()
        pain_point = {
            "id": "mlir-test-1",
            "type": "index_out_of_bounds",
            "severity": "high",
            "message": "Index out of bounds detected in memref.load",
            "location": "test.mlir",
            "context": {"operation": "memref.load"},
        }

        pattern = extractor.extract_pattern("mlir", pain_point)

        assert pattern is not None
        assert pattern.pattern_type == AbstractPatternType.BOUNDARY_CHECK_MISSING
        assert pattern.domain == "mlir"
        assert pattern.confidence > 0.5

    def test_extract_pattern_from_k8s(self):
        extractor = AbstractPatternExtractor()
        pain_point = {
            "id": "k8s-test-1",
            "type": "missing_resources",
            "severity": "medium",
            "message": "deployment missing resource limits",
            "location": "deploy.yaml",
            "context": {"deployment": "api"},
        }

        pattern = extractor.extract_pattern("k8s", pain_point)

        assert pattern is not None
        assert pattern.pattern_type == AbstractPatternType.RESOURCE_LIMIT_MISSING
        assert pattern.domain == "k8s"

    def test_extract_pattern_from_python(self):
        extractor = AbstractPatternExtractor()
        pain_point = {
            "id": "py-test-1",
            "type": "null_pointer",
            "severity": "high",
            "message": "Potential null pointer dereference",
            "location": "handler.py",
            "context": {"function": "process"},
        }

        pattern = extractor.extract_pattern("default", pain_point)

        assert pattern is not None
        assert pattern.pattern_type == AbstractPatternType.BOUNDARY_CHECK_MISSING
        assert pattern.domain == "default"

    def test_extract_patterns_batch(self):
        extractor = AbstractPatternExtractor()
        pain_points = [
            {"id": "1", "type": "index_out_of_bounds", "severity": "high", "message": "out of bounds"},
            {"id": "2", "type": "missing_resources", "severity": "medium", "message": "missing limits"},
        ]

        patterns = extractor.extract_patterns("mlir", pain_points)

        assert len(patterns) == 2


class TestPatternSimilarityEngine:
    def test_calculate_similarity_same_type(self):
        extractor = AbstractPatternExtractor()
        engine = PatternSimilarityEngine()

        pattern_a = extractor.extract_pattern(
            "mlir",
            {
                "id": "a",
                "type": "index_out_of_bounds",
                "severity": "high",
                "message": "array index out of bounds",
                "context": {},
            },
        )
        pattern_b = extractor.extract_pattern(
            "default",
            {"id": "b", "type": "null_pointer", "severity": "high", "message": "null pointer access", "context": {}},
        )

        if pattern_a and pattern_b:
            similarity = engine.calculate_similarity(pattern_a, pattern_b)
            assert 0 <= similarity <= 1.0

    def test_find_cross_domain_similarities(self):
        extractor = AbstractPatternExtractor()
        engine = PatternSimilarityEngine()

        patterns = [
            extractor.extract_pattern(
                "mlir",
                {
                    "id": "mlir-1",
                    "type": "index_out_of_bounds",
                    "severity": "high",
                    "message": "missing bounds check",
                    "context": {},
                },
            ),
            extractor.extract_pattern(
                "k8s",
                {
                    "id": "k8s-1",
                    "type": "missing_resources",
                    "severity": "medium",
                    "message": "missing resource limits",
                    "context": {},
                },
            ),
            extractor.extract_pattern(
                "default",
                {
                    "id": "py-1",
                    "type": "null_pointer",
                    "severity": "high",
                    "message": "missing null check",
                    "context": {},
                },
            ),
        ]

        patterns = [p for p in patterns if p is not None]
        engine.add_patterns(patterns)

        similarities = engine.find_cross_domain_similarities(threshold=0.3)
        assert isinstance(similarities, list)

    def test_cluster_patterns(self):
        extractor = AbstractPatternExtractor()
        engine = PatternSimilarityEngine()

        patterns = [
            extractor.extract_pattern(
                "mlir",
                {
                    "id": "mlir-1",
                    "type": "index_out_of_bounds",
                    "severity": "high",
                    "message": "out of bounds",
                    "context": {},
                },
            ),
            extractor.extract_pattern(
                "default",
                {"id": "py-1", "type": "null_pointer", "severity": "high", "message": "null pointer", "context": {}},
            ),
        ]

        patterns = [p for p in patterns if p is not None]
        engine.add_patterns(patterns)

        clusters = engine.cluster_patterns()
        assert len(clusters) >= 1


class TestCrossDomainTranslator:
    def test_translate_fix(self):
        translator = CrossDomainTranslator()

        fix = translator.translate_fix("k8s", "default", AbstractPatternType.RESOURCE_LIMIT_MISSING)

        assert fix is not None
        assert fix.domain == "default"
        assert fix.fix_description is not None

    def test_generate_universal_fix(self):
        translator = CrossDomainTranslator()

        template = translator.generate_universal_fix(AbstractPatternType.BOUNDARY_CHECK_MISSING)

        assert template is not None
        assert len(template.domain_fixes) >= 3
        assert template.cross_domain_insight != ""

    def test_translate_from_pattern(self):
        extractor = AbstractPatternExtractor()
        translator = CrossDomainTranslator()

        pattern = extractor.extract_pattern(
            "mlir",
            {
                "id": "test",
                "type": "index_out_of_bounds",
                "severity": "high",
                "message": "out of bounds",
                "context": {},
            },
        )

        if pattern:
            fix = translator.translate_from_pattern(pattern, "k8s")
            assert fix is not None
            assert fix.domain == "k8s"


class TestKnowledgeAmalgamator:
    def test_amalgamate_patterns(self):
        extractor = AbstractPatternExtractor()
        amalgamator = KnowledgeAmalgamator()

        patterns = [
            extractor.extract_pattern(
                "mlir",
                {
                    "id": "mlir-1",
                    "type": "index_out_of_bounds",
                    "severity": "high",
                    "message": "out of bounds",
                    "context": {},
                },
            ),
            extractor.extract_pattern(
                "k8s",
                {
                    "id": "k8s-1",
                    "type": "missing_resources",
                    "severity": "medium",
                    "message": "missing limits",
                    "context": {},
                },
            ),
        ]

        patterns = [p for p in patterns if p is not None]
        count = amalgamator.amalgamate_patterns(patterns)

        assert count == len(patterns)
        assert amalgamator.get_pattern_count() >= 1

    def test_save_and_load(self):
        extractor = AbstractPatternExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            amalgamator = KnowledgeAmalgamator(temp_path)

            patterns = [
                extractor.extract_pattern(
                    "mlir",
                    {
                        "id": "test",
                        "type": "index_out_of_bounds",
                        "severity": "high",
                        "message": "test pattern",
                        "context": {},
                    },
                ),
            ]

            patterns = [p for p in patterns if p is not None]
            amalgamator.amalgamate_patterns(patterns)
            amalgamator.save()

            assert os.path.exists(temp_path)

            new_amalgamator = KnowledgeAmalgamator(temp_path)
            new_amalgamator.load()

            assert new_amalgamator.get_pattern_count() == amalgamator.get_pattern_count()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_universal_fixes(self):
        extractor = AbstractPatternExtractor()
        amalgamator = KnowledgeAmalgamator()

        patterns = [
            extractor.extract_pattern(
                "mlir",
                {"id": "mlir-1", "type": "index_out_of_bounds", "severity": "high", "message": "test", "context": {}},
            ),
            extractor.extract_pattern(
                "k8s",
                {"id": "k8s-1", "type": "missing_resources", "severity": "medium", "message": "test", "context": {}},
            ),
        ]

        patterns = [p for p in patterns if p is not None]
        amalgamator.amalgamate_patterns(patterns)

        fixes = amalgamator.get_universal_fixes()
        assert len(fixes) >= 1


class TestValueTracker:
    def test_analyze_history_no_tracker(self):
        tracker = ValueTracker()
        changes = tracker.analyze_history()
        assert len(changes) == 0

    def test_effective_changes_empty(self):
        tracker = ValueTracker()
        effective = tracker.get_effective_changes()
        assert len(effective) == 0


class TestStrategyEvaluator:
    def test_evaluate_empty_changes(self):
        evaluator = StrategyEvaluator()
        report = evaluator.evaluate([])
        assert report.total_strategies_analyzed == 0

    def test_evaluate_with_mixed_changes(self):
        changes = [
            TrackedChange(
                change_type="correction",
                target="auto_correction",
                description="test correction",
                applied_at_snapshot=1,
                before_value=0.70,
                after_value=0.82,
                actual_benefit=0.12,
                effective=True,
            ),
            TrackedChange(
                change_type="increase_weight",
                target="type_mismatch",
                description="test weight",
                applied_at_snapshot=2,
                before_value=0.1,
                after_value=0.15,
                actual_benefit=0.05,
                effective=True,
            ),
        ]
        evaluator = StrategyEvaluator()
        report = evaluator.evaluate(changes)
        assert report.total_strategies_analyzed == 2
        assert len(report.evaluations) >= 2


class TestValueDiscovery:
    def test_discover_principles(self):
        from hermes.cross_domain.strategy_evaluator import StrategyEvaluation, StrategyReport

        report = StrategyReport(
            evaluations=[
                StrategyEvaluation(
                    strategy_type="correction",
                    target="global",
                    instances=2,
                    avg_benefit=0.12,
                    median_benefit=0.12,
                    success_rate=1.0,
                    total_positive=2,
                    total_negative=0,
                    recommendation="有效策略",
                ),
            ]
        )
        discovery = ValueDiscovery()
        principles = discovery.discover(report)
        assert len(principles) >= 3
        assert all(hasattr(p, "principle_id") for p in principles)


class TestSelfGuidedEvolver:
    def test_evolver_no_tracker(self):
        evolver = SelfGuidedEvolver()
        result = evolver.run_value_driven_cycle()
        assert "principles" in result
        assert "targets" in result
        assert len(result["targets"]) >= 1

    def test_evolver_targets_have_expected_benefit(self):
        evolver = SelfGuidedEvolver()
        result = evolver.run_value_driven_cycle()
        for target in result["targets"]:
            assert "expected_benefit" in target
            assert target["expected_benefit"] > 0


class TestMentalModelTrainer:
    def test_train_no_tracker(self):
        trainer = MentalModelTrainer()
        model = trainer.train()
        assert model.training_samples == 0
        # 样本不足时不允许编造准确率：只标记为未评测
        assert model.estimated is True
        assert model.to_dict()["estimated"] is True

    def test_predict_returns_value(self):
        trainer = MentalModelTrainer()
        trainer.train()
        result = trainer.predict_success_rate(
            {"success_rate": 0.7, "cross_domain_success": 0.5, "pattern_coverage": 0.5},
            "reset_similarity",
            "type_mismatch",
            0.0,
        )
        assert 0 <= result <= 1.0

    def test_feature_importance(self):
        trainer = MentalModelTrainer()
        model = trainer.train()
        assert len(model.feature_importance) >= 5


class TestMentalSimulator:
    def test_simulate_reset_similarity(self):
        trainer = MentalModelTrainer()
        simulator = MentalSimulator(trainer)
        result = simulator.simulate("reset_similarity", "type_mismatch")
        assert result.strategy_type == "reset_similarity"
        assert result.target == "type_mismatch"
        assert result.predicted_delta != 0

    def test_simulate_candidates_returns_list(self):
        trainer = MentalModelTrainer()
        simulator = MentalSimulator(trainer)
        results = simulator.simulate_candidates()
        assert len(results) >= 3
        assert results[0].predicted_delta >= results[-1].predicted_delta  # sorted


class TestStrategySandbox:
    def test_evaluate_all(self):
        trainer = MentalModelTrainer()
        simulator = MentalSimulator(trainer)
        sandbox = StrategySandbox(simulator)
        report = sandbox.evaluate_all()
        assert report.best_candidate is not None
        assert len(report.rankings) >= 3

    def test_recommendation(self):
        trainer = MentalModelTrainer()
        simulator = MentalSimulator(trainer)
        sandbox = StrategySandbox(simulator)
        report = sandbox.evaluate_all()
        assert len(report.recommended_action) > 0


class TestCognitivePlanner:
    def test_plan_no_tracker(self):
        planner = CognitivePlanner()
        plan = planner.plan()
        assert hasattr(plan, "selected_strategy")
        assert hasattr(plan, "sandbox_report")


class TestRepositoryMiner:
    def test_mine_demo(self):
        miner = RepositoryMiner()
        report = miner.mine("DEMO")
        assert report.total_events >= 10
        assert len(report.events_by_type) >= 3

    def test_mine_local_repo_honors_max_events(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(repo_root, ".git")):
            pytest.skip("git working tree not available")

        miner = RepositoryMiner()
        report = miner.mine(repo_root, max_events=3)

        assert report.mining_method == "git_log"
        assert 1 <= report.total_events <= 3

    def test_get_events_by_type(self):
        miner = RepositoryMiner()
        miner.mine("DEMO")
        events = miner.get_events_by_type("boundary_fix")
        assert len(events) >= 1


class TestEventPatternLearner:
    def test_learn_from_report(self):
        miner = RepositoryMiner()
        report = miner.mine("DEMO")
        learner = EventPatternLearner()
        patterns = learner.learn(report)
        assert len(patterns) >= 3

    def test_mapped_type_present(self):
        miner = RepositoryMiner()
        report = miner.mine("DEMO")
        learner = EventPatternLearner()
        patterns = learner.learn(report)
        mapped_types = [p.mapped_type for p in patterns]
        assert "boundary_check_missing" in mapped_types


class TestExternalInsightMapper:
    def test_map_patterns(self):
        miner = RepositoryMiner()
        report = miner.mine("DEMO")
        learner = EventPatternLearner()
        patterns = learner.learn(report)
        mapper = ExternalInsightMapper()
        insights = mapper.map_patterns(patterns)
        assert len(insights) >= 2

    def test_insight_order(self):
        miner = RepositoryMiner()
        report = miner.mine("DEMO")
        learner = EventPatternLearner()
        patterns = learner.learn(report)
        mapper = ExternalInsightMapper()
        insights = mapper.map_patterns(patterns)
        for i in range(1, len(insights)):
            assert insights[i - 1].estimated_impact >= insights[i].estimated_impact


class TestCrossEntityLearner:
    def test_learn_from_repo(self):
        from hermes.cross_domain.self_improvement_policy import SelfImprovementPolicy

        learner = CrossEntityLearner(policy_loader=SelfImprovementPolicy())
        result = learner.learn_from("DEMO")
        assert result.report.total_events >= 10
        assert len(result.patterns) >= 3


class _StubSnapshot:
    def __init__(self, success_rate: float):
        self.success_rate = success_rate


class _StubTracker:
    """只需要 get_recent_snapshots 的 EvolutionTracker 替身。"""

    def __init__(self, snapshots):
        self._snapshots = list(snapshots)

    def get_recent_snapshots(self, n: int = 100):
        return self._snapshots[:n]


class _StubTrainer:
    """返回固定预测值的 MentalModelTrainer 替身（接口与真实实现一致）。"""

    def __init__(self, predicted: float):
        self._predicted = predicted
        self._model = PredictionModel(accuracy=0.9, feature_importance={}, training_samples=8)

    def get_model(self):
        return self._model

    def predict_success_rate(self, current_state, strategy_type, target, adjustment):
        return self._predicted


def _stub_snapshots(count: int):
    return [_StubSnapshot(0.60 + 0.05 * index) for index in range(count)]


def _stub_stages(count: int):
    return [
        EvolutionStage(
            stage_id=index + 1,
            name=f"测试阶段{index + 1}",
            description="",
            commit_range="a..b",
            total_commits=2,
            dominant_types=["foundation"],
        )
        for index in range(count)
    ]


class TestSelfRepositoryMiner:
    def test_missing_git_returns_no_stages(self):
        """git 不可用时返回空列表并显式标记，绝不返回编造的阶段。"""

        miner = SelfRepositoryMiner("/nonexistent")
        stages = miner.mine_self()
        assert stages == []
        assert miner.git_unavailable is True
        assert miner.unavailable_reason
        assert miner.get_stages() == []

    def test_missing_git_never_fabricates_commit_counts(self):
        """旧实现在 git 不可用时返回 12/15/10 次提交的硬编码阶段。"""

        miner = SelfRepositoryMiner("/nonexistent")
        payload = miner.to_dict()
        assert payload["stages"] == []
        assert payload["git_unavailable"] is True
        assert [stage.total_commits for stage in miner.mine_self()] == []

    def test_real_repository_stages_are_data_driven(self):
        """在真实仓库上：阶段提交数之和等于真实提交数，阶段名来自类型标签。"""

        miner = SelfRepositoryMiner(REPO_ROOT)
        stages = miner.mine_self()
        detection = miner.detection
        assert detection is not None

        if detection.git_unavailable:
            assert stages == []
            return

        assert stages
        assert sum(stage.total_commits for stage in stages) == detection.commit_count
        assert [stage.stage_id for stage in stages] == list(range(1, len(stages) + 1))
        for stage in stages:
            assert stage.name.rstrip("期") in CATEGORY_LABELS.values() or any(
                label in stage.name for label in CATEGORY_LABELS.values()
            )
            assert stage.start_date is not None


class TestEvolutionCompareEngine:
    def test_compare_without_snapshots_does_not_fabricate(self):
        """旧实现在没有快照时用 0.45/0.65/0.72 之类的演示数值拼出对比。"""

        engine = EvolutionCompareEngine()
        report = engine.compare(_stub_stages(3))
        assert report.comparisons == []
        assert report.overall_accuracy == 0.0
        assert report.model_bias == "no_data"
        assert report.notes

    def test_compare_without_stages_does_not_fabricate(self):
        engine = EvolutionCompareEngine(tracker=_StubTracker(_stub_snapshots(6)))
        report = engine.compare([])
        assert report.comparisons == []
        assert report.model_bias == "no_data"

    def test_compare_without_trained_model_does_not_fabricate(self):
        engine = EvolutionCompareEngine(tracker=_StubTracker(_stub_snapshots(6)), trainer=None)
        report = engine.compare(_stub_stages(3))
        assert report.comparisons == []
        assert report.model_bias == "no_model"
        assert report.notes

    def test_compare_fewer_snapshots_than_stages_does_not_fabricate(self):
        engine = EvolutionCompareEngine(tracker=_StubTracker(_stub_snapshots(2)), trainer=_StubTrainer(0.75))
        report = engine.compare(_stub_stages(3))
        assert report.comparisons == []
        assert report.model_bias == "no_data"

    def test_compare_uses_real_snapshot_means(self):
        engine = EvolutionCompareEngine(tracker=_StubTracker(_stub_snapshots(6)), trainer=_StubTrainer(0.75))
        report = engine.compare(_stub_stages(3))

        assert len(report.comparisons) == 3
        assert report.alignment == "snapshot_index_range"
        assert report.notes

        means = [comp.actual_success_rate for comp in report.comparisons]
        assert means == pytest.approx([0.625, 0.725, 0.825])
        assert all(comp.snapshot_count == 2 for comp in report.comparisons)
        assert all(comp.predicted_success_rate == pytest.approx(0.75) for comp in report.comparisons)
        for comp in report.comparisons:
            assert comp.over_under in ("overestimated", "underestimated", "accurate")
            assert comp.accuracy == pytest.approx(1.0 - abs(comp.delta))

        assert report.model_bias == "mixed"
        assert report.overall_accuracy == pytest.approx((0.875 + 0.975 + 0.925) / 3)


class TestGapCalibrator:
    def test_analyze_balanced(self):
        from hermes.cross_domain.evolution_compare_engine import ComparisonReport, StageComparison

        report = ComparisonReport(model_bias="balanced")
        report.comparisons = [
            StageComparison(
                stage_id=1,
                stage_name="test",
                actual_success_rate=0.65,
                predicted_success_rate=0.66,
                delta=0.01,
                accuracy=0.99,
                over_under="accurate",
                evidence="test",
            ),
        ]
        calibrator = GapCalibrator()
        suggestions = calibrator.analyze(report)
        assert len(suggestions) >= 1
        assert any("fine_tune" in s.target for s in suggestions)

    def test_analyze_conservative(self):
        from hermes.cross_domain.evolution_compare_engine import ComparisonReport

        report = ComparisonReport(model_bias="conservative")
        calibrator = GapCalibrator()
        suggestions = calibrator.analyze(report)
        assert len(suggestions) >= 1


class TestRecursiveInsightInjector:
    def test_inject_no_trainer(self):
        from hermes.cross_domain.evolution_compare_engine import ComparisonReport

        report = ComparisonReport(model_bias="balanced")
        injector = RecursiveInsightInjector()
        result = injector.inject(report)
        assert result.calibrations_applied == 0
        assert "已校准" in result.message
