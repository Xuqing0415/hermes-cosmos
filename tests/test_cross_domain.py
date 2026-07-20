import pytest
import tempfile
import os

from hermes.cross_domain import (
    AbstractPatternExtractor,
    AbstractPatternType,
    PatternSimilarityEngine,
    CrossDomainTranslator,
    KnowledgeAmalgamator,
    TrackedChange,
    ValueTracker,
    StrategyEvaluator,
    ValueDiscovery,
    EvolutionTarget,
    SelfGuidedEvolver,
    MentalModelTrainer,
    MentalSimulator,
    StrategySandbox,
    CognitivePlanner,
    RepositoryMiner,
    EventPatternLearner,
    ExternalInsightMapper,
    CrossEntityLearner,
    SelfRepositoryMiner,
    EvolutionCompareEngine,
    GapCalibrator,
    RecursiveInsightInjector,
)


class TestAbstractPatternExtractor:
    def test_extract_pattern_from_mlir(self):
        extractor = AbstractPatternExtractor()
        pain_point = {
            "id": "mlir-test-1",
            "type": "index_out_of_bounds",
            "severity": "high",
            "message": "Index out of bounds detected in memref.load",
            "location": "test.mlir",
            "context": {"operation": "memref.load"}
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
            "context": {"deployment": "api"}
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
            "context": {"function": "process"}
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
        
        pattern_a = extractor.extract_pattern("mlir", {
            "id": "a", "type": "index_out_of_bounds", "severity": "high", 
            "message": "array index out of bounds", "context": {}
        })
        pattern_b = extractor.extract_pattern("default", {
            "id": "b", "type": "null_pointer", "severity": "high", 
            "message": "null pointer access", "context": {}
        })
        
        if pattern_a and pattern_b:
            similarity = engine.calculate_similarity(pattern_a, pattern_b)
            assert 0 <= similarity <= 1.0

    def test_find_cross_domain_similarities(self):
        extractor = AbstractPatternExtractor()
        engine = PatternSimilarityEngine()
        
        patterns = [
            extractor.extract_pattern("mlir", {
                "id": "mlir-1", "type": "index_out_of_bounds", "severity": "high",
                "message": "missing bounds check", "context": {}
            }),
            extractor.extract_pattern("k8s", {
                "id": "k8s-1", "type": "missing_resources", "severity": "medium",
                "message": "missing resource limits", "context": {}
            }),
            extractor.extract_pattern("default", {
                "id": "py-1", "type": "null_pointer", "severity": "high",
                "message": "missing null check", "context": {}
            }),
        ]
        
        patterns = [p for p in patterns if p is not None]
        engine.add_patterns(patterns)
        
        similarities = engine.find_cross_domain_similarities(threshold=0.3)
        assert isinstance(similarities, list)

    def test_cluster_patterns(self):
        extractor = AbstractPatternExtractor()
        engine = PatternSimilarityEngine()
        
        patterns = [
            extractor.extract_pattern("mlir", {
                "id": "mlir-1", "type": "index_out_of_bounds", "severity": "high",
                "message": "out of bounds", "context": {}
            }),
            extractor.extract_pattern("default", {
                "id": "py-1", "type": "null_pointer", "severity": "high",
                "message": "null pointer", "context": {}
            }),
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
        
        pattern = extractor.extract_pattern("mlir", {
            "id": "test", "type": "index_out_of_bounds", "severity": "high",
            "message": "out of bounds", "context": {}
        })
        
        if pattern:
            fix = translator.translate_from_pattern(pattern, "k8s")
            assert fix is not None
            assert fix.domain == "k8s"


class TestKnowledgeAmalgamator:
    def test_amalgamate_patterns(self):
        extractor = AbstractPatternExtractor()
        amalgamator = KnowledgeAmalgamator()
        
        patterns = [
            extractor.extract_pattern("mlir", {
                "id": "mlir-1", "type": "index_out_of_bounds", "severity": "high",
                "message": "out of bounds", "context": {}
            }),
            extractor.extract_pattern("k8s", {
                "id": "k8s-1", "type": "missing_resources", "severity": "medium",
                "message": "missing limits", "context": {}
            }),
        ]
        
        patterns = [p for p in patterns if p is not None]
        count = amalgamator.amalgamate_patterns(patterns)
        
        assert count == len(patterns)
        assert amalgamator.get_pattern_count() >= 1

    def test_save_and_load(self):
        extractor = AbstractPatternExtractor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            amalgamator = KnowledgeAmalgamator(temp_path)
            
            patterns = [
                extractor.extract_pattern("mlir", {
                    "id": "test", "type": "index_out_of_bounds", "severity": "high",
                    "message": "test pattern", "context": {}
                }),
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
            extractor.extract_pattern("mlir", {
                "id": "mlir-1", "type": "index_out_of_bounds", "severity": "high",
                "message": "test", "context": {}
            }),
            extractor.extract_pattern("k8s", {
                "id": "k8s-1", "type": "missing_resources", "severity": "medium",
                "message": "test", "context": {}
            }),
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
                change_type="correction", target="auto_correction",
                description="test correction", applied_at_snapshot=1,
                before_value=0.70, after_value=0.82, actual_benefit=0.12, effective=True,
            ),
            TrackedChange(
                change_type="increase_weight", target="type_mismatch",
                description="test weight", applied_at_snapshot=2,
                before_value=0.1, after_value=0.15, actual_benefit=0.05, effective=True,
            ),
        ]
        evaluator = StrategyEvaluator()
        report = evaluator.evaluate(changes)
        assert report.total_strategies_analyzed == 2
        assert len(report.evaluations) >= 2


class TestValueDiscovery:
    def test_discover_principles(self):
        from hermes.cross_domain.strategy_evaluator import StrategyReport, StrategyEvaluation
        report = StrategyReport(
            evaluations=[
                StrategyEvaluation(
                    strategy_type="correction", target="global", instances=2,
                    avg_benefit=0.12, median_benefit=0.12, success_rate=1.0,
                    total_positive=2, total_negative=0,
                    recommendation="有效策略",
                ),
            ]
        )
        discovery = ValueDiscovery()
        principles = discovery.discover(report)
        assert len(principles) >= 3
        assert all(hasattr(p, 'principle_id') for p in principles)


class TestSelfGuidedEvolver:
    def test_evolver_no_tracker(self):
        evolver = SelfGuidedEvolver()
        result = evolver.run_value_driven_cycle()
        assert 'principles' in result
        assert 'targets' in result
        assert len(result['targets']) >= 1

    def test_evolver_targets_have_expected_benefit(self):
        evolver = SelfGuidedEvolver()
        result = evolver.run_value_driven_cycle()
        for target in result['targets']:
            assert 'expected_benefit' in target
            assert target['expected_benefit'] > 0


class TestMentalModelTrainer:
    def test_train_no_tracker(self):
        trainer = MentalModelTrainer()
        model = trainer.train()
        assert model.training_samples == 0
        assert 0.5 <= model.accuracy <= 1.0

    def test_predict_returns_value(self):
        trainer = MentalModelTrainer()
        trainer.train()
        result = trainer.predict_success_rate(
            {"success_rate": 0.7, "cross_domain_success": 0.5, "pattern_coverage": 0.5},
            "reset_similarity", "type_mismatch", 0.0
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
        assert hasattr(plan, 'selected_strategy')
        assert hasattr(plan, 'sandbox_report')


class TestRepositoryMiner:
    def test_mine_demo(self):
        miner = RepositoryMiner()
        report = miner.mine("DEMO")
        assert report.total_events >= 10
        assert len(report.events_by_type) >= 3

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


class TestSelfRepositoryMiner:
    def test_mine_self_synthetic(self):
        miner = SelfRepositoryMiner("/nonexistent")
        stages = miner.mine_self()
        assert len(stages) == 3
        assert stages[0].name == "基础构建期"

    def test_get_stages(self):
        miner = SelfRepositoryMiner("/nonexistent")
        miner.mine_self()
        stages = miner.get_stages()
        assert len(stages) == 3


class TestEvolutionCompareEngine:
    def test_compare_no_tracker(self):
        miner = SelfRepositoryMiner("/nonexistent")
        stages = miner.mine_self()
        engine = EvolutionCompareEngine()
        report = engine.compare(stages)
        assert len(report.comparisons) == 3
        assert report.overall_accuracy > 0

    def test_comparison_has_deltas(self):
        miner = SelfRepositoryMiner("/nonexistent")
        stages = miner.mine_self()
        engine = EvolutionCompareEngine()
        report = engine.compare(stages)
        for comp in report.comparisons:
            assert comp.over_under in ("overestimated", "underestimated", "accurate")


class TestGapCalibrator:
    def test_analyze_balanced(self):
        from hermes.cross_domain.evolution_compare_engine import ComparisonReport, StageComparison
        report = ComparisonReport(model_bias="balanced")
        report.comparisons = [
            StageComparison(
                stage_id=1, stage_name="test",
                actual_success_rate=0.65, predicted_success_rate=0.66,
                delta=0.01, accuracy=0.99, over_under="accurate",
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