import pytest
import tempfile
import os
import shutil

from hermes.self_refactor import (
    ArchAnalyzer,
    SmellDetector,
    RefactorPlanGenerator,
    BehaviorVerifier,
    RefactorRollback,
    SelfRefactorEngine,
    SmellType,
    SmellSeverity,
    RefactorPlan,
    CodeLocation,
    SmellInstance
)


class TestArchAnalyzer:
    def test_init(self):
        analyzer = ArchAnalyzer()
        assert analyzer.source_dir == "src/hermes"
    
    def test_analyze_file(self):
        analyzer = ArchAnalyzer()
        test_file = "src/hermes/self_refactor/types.py"
        if os.path.exists(test_file):
            metrics = analyzer.analyze_file(test_file)
            assert metrics.loc > 0
            assert metrics.cyclomatic_complexity >= 0
            assert metrics.function_count >= 0
            assert metrics.class_count >= 0
    
    def test_analyze_module(self):
        analyzer = ArchAnalyzer()
        module_path = "src/hermes/self_refactor"
        if os.path.exists(module_path):
            metrics = analyzer.analyze_module(module_path)
            assert metrics.total_loc > 0
            assert metrics.total_functions > 0
            assert metrics.coupling_score >= 0
            assert metrics.cohesion_score >= 0
    
    def test_get_function_signature(self):
        analyzer = ArchAnalyzer()
        test_file = "src/hermes/self_refactor/arch_analyzer.py"
        if os.path.exists(test_file):
            sig = analyzer.get_function_signature(test_file, "analyze_file")
            assert sig is not None
            assert len(sig) == 16


class TestSmellDetector:
    def test_init(self):
        detector = SmellDetector()
        assert detector.detected_smells == []
    
    def test_detect_all(self):
        detector = SmellDetector()
        smells = detector.detect_all()
        assert isinstance(smells, list)
    
    def test_get_top_smells(self):
        detector = SmellDetector()
        smells = detector.detect_all()
        top = detector.get_top_smells(3)
        assert len(top) <= 3
    
    def test_get_smells_by_type(self):
        detector = SmellDetector()
        detector.detect_all()
        smells = detector.get_smells_by_type(SmellType.LONG_FUNCTION)
        assert isinstance(smells, list)
    
    def test_get_smells_by_severity(self):
        detector = SmellDetector()
        detector.detect_all()
        smells = detector.get_smells_by_severity(SmellSeverity.HIGH)
        assert isinstance(smells, list)


class TestRefactorPlanGenerator:
    def test_init(self):
        generator = RefactorPlanGenerator()
        assert generator.analyzer is not None
    
    def test_generate_plans(self):
        detector = SmellDetector()
        smells = detector.detect_all()
        generator = RefactorPlanGenerator()
        plans = generator.generate_plans(smells[:3])
        assert isinstance(plans, list)
    
    def test_generate_plan_for_long_function(self):
        smell = SmellInstance(
            smell_type=SmellType.LONG_FUNCTION,
            severity=SmellSeverity.MEDIUM,
            description="Test long function",
            locations=[CodeLocation("test.py", 1, 60)],
            metadata={'function_name': 'test_func', 'complexity': 15, 'line_count': 60},
            priority=60.0
        )
        generator = RefactorPlanGenerator()
        plan = generator._generate_extract_function_plan("test123", smell)
        assert plan is not None
        assert len(plan.actions) > 0
    
    def test_evaluate_plan_risk(self):
        generator = RefactorPlanGenerator()
        smell = SmellInstance(
            smell_type=SmellType.LONG_FUNCTION,
            severity=SmellSeverity.LOW,
            description="Test",
            locations=[CodeLocation("test.py", 1, 5)],
            priority=10.0
        )
        plan = RefactorPlan(
            plan_id="test",
            smell=smell,
            actions=[],
            estimated_effort=1.0,
            risk_level="low"
        )
        risk = generator.evaluate_plan_risk(plan)
        assert risk in ["low", "medium", "high"]
    
    def test_get_plan_summary(self):
        smell = SmellInstance(
            smell_type=SmellType.DUPLICATE_CODE,
            severity=SmellSeverity.MEDIUM,
            description="Test",
            locations=[],
            priority=50.0
        )
        plan = RefactorPlan(
            plan_id="test",
            smell=smell,
            actions=[],
            estimated_effort=2.0,
            risk_level="medium"
        )
        generator = RefactorPlanGenerator()
        summary = generator.get_plan_summary(plan)
        assert " ID" in summary
        assert "" in summary


class TestBehaviorVerifier:
    def test_init(self):
        verifier = BehaviorVerifier()
        assert verifier.test_dir == "tests"
    
    def test_compute_code_hash(self):
        verifier = BehaviorVerifier()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass")
            temp_file = f.name
        try:
            hash_val = verifier.compute_code_hash(temp_file)
            assert len(hash_val) == 16
        finally:
            os.unlink(temp_file)
    
    def test_compare_code_hashes(self):
        verifier = BehaviorVerifier()
        hashes1 = {"file1.py": "abc123", "file2.py": "def456"}
        hashes2 = {"file1.py": "abc123", "file2.py": "xyz789"}
        changed = verifier.compare_code_hashes(hashes1, hashes2)
        assert "file2.py" in changed
        assert "file1.py" not in changed


class TestRefactorRollback:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rollback = RefactorRollback(tmpdir)
            assert os.path.exists(tmpdir)
    
    def test_create_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rollback = RefactorRollback(tmpdir)
            smell = SmellInstance(
                smell_type=SmellType.LONG_FUNCTION,
                severity=SmellSeverity.LOW,
                description="Test",
                locations=[],
                priority=10.0
            )
            plan = RefactorPlan(
                plan_id="test_backup",
                smell=smell,
                actions=[],
                estimated_effort=1.0,
                risk_level="low"
            )
            backup_path = rollback.create_backup(plan)
            assert os.path.exists(backup_path)
    
    def test_list_backups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rollback = RefactorRollback(tmpdir)
            backups = rollback.list_backups()
            assert isinstance(backups, list)
    
    def test_get_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rollback = RefactorRollback(tmpdir)
            history = rollback.get_history()
            assert isinstance(history, list)


class TestSelfRefactorEngine:
    def test_init(self):
        engine = SelfRefactorEngine(auto_apply=False)
        assert engine.source_dir == "src/hermes"
        assert engine.auto_apply is False
    
    def test_detect_smells(self):
        engine = SelfRefactorEngine()
        smells = engine.detect_smells()
        assert isinstance(smells, list)
    
    def test_generate_plans(self):
        engine = SelfRefactorEngine()
        smells = engine.detect_smells()
        plans = engine.generate_plans(smells[:2])
        assert isinstance(plans, list)
    
    def test_generate_health_report(self):
        engine = SelfRefactorEngine()
        report = engine.generate_health_report()
        assert report.total_smells >= 0
        assert 0 <= report.overall_health_score <= 1.0
        assert isinstance(report.recommendations, list)
    
    def test_get_top_smells(self):
        engine = SelfRefactorEngine()
        smells = engine.get_top_smells(5)
        assert len(smells) <= 5
    
    def test_get_refactor_history(self):
        engine = SelfRefactorEngine()
        history = engine.get_refactor_history()
        assert isinstance(history, list)
    
    def test_set_auto_apply(self):
        engine = SelfRefactorEngine()
        engine.set_auto_apply(True)
        assert engine.auto_apply is True
        engine.set_auto_apply(False)
        assert engine.auto_apply is False


class TestIntegration:
    def test_full_pipeline(self):
        engine = SelfRefactorEngine(auto_apply=False)
        report = engine.generate_health_report()
        assert report is not None
        
        smells = engine.detect_smells()
        assert isinstance(smells, list)
        
        plans = engine.generate_plans(smells[:2])
        assert isinstance(plans, list)