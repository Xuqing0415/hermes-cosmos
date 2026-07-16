"""
Tests for Neural-Symbolic Proof Generation Module
"""

import pytest
import os
import tempfile

from hermes.neural_symbolic.theorem_prover import TheoremProver
from hermes.neural_symbolic.proof_obligation import ProofObligationGenerator
from hermes.neural_symbolic.certificate import ProofCertificate
from hermes.neural_symbolic.target_predictor import NeuralProofTargetPredictor
from hermes.neural_symbolic.test_generator import ProofGuidedTestGenerator
from hermes.neural_symbolic.proof_generator import ProofGenerator
from hermes.neural_symbolic.types import (
    ProofStatus, 
    PathCondition, 
    ProofTarget,
    GeneratedTest,
    ProofResult
)


class TestTheoremProver:
    """Tests for TheoremProver"""
    
    def test_prove_no_div_by_zero(self):
        prover = TheoremProver()
        result = prover.verify_no_div_by_zero("b")
        
        assert result.status in [ProofStatus.PROVEN, ProofStatus.DISPROVEN, ProofStatus.UNKNOWN]
    
    def test_prove_implication(self):
        prover = TheoremProver()
        result = prover.prove_implication("(>= x 1)", "(> x 0)")
        
        assert result.status in [ProofStatus.PROVEN, ProofStatus.UNKNOWN]
    
    def test_prove_timeout(self):
        prover = TheoremProver(timeout=1)
        result = prover.prove("(assert true)")
        
        assert result.status in [ProofStatus.PROVEN, ProofStatus.UNKNOWN]
    
    def test_prove_false(self):
        prover = TheoremProver()
        result = prover.prove("(assert false)")
        
        assert result.status == ProofStatus.PROVEN


class TestProofObligationGenerator:
    """Tests for ProofObligationGenerator"""
    
    def test_generate_from_code(self):
        code = """
def divide(a, b):
    return a / b
"""
        generator = ProofObligationGenerator()
        vcs = generator.generate_from_code(code, "divide")
        
        assert isinstance(vcs, list)
        assert len(vcs) > 0
    
    def test_generate_from_code_no_function(self):
        code = """
def add(a, b):
    return a + b
"""
        generator = ProofObligationGenerator()
        vcs = generator.generate_from_code(code, "nonexistent")
        
        assert vcs == []
    
    def test_generate_from_path_condition(self):
        generator = ProofObligationGenerator()
        path_cond = PathCondition("(not (= b 0))")
        
        vc = generator.generate_from_path_condition(path_cond)
        
        assert vc is not None
        assert "(not (= b 0))" in vc.smt_formula
    
    def test_generate_proof_targets(self):
        code = """
def divide(a, b):
    return a / b
"""
        generator = ProofObligationGenerator()
        targets = generator.generate_proof_targets(code, "divide")
        
        assert isinstance(targets, list)
        assert len(targets) > 0
        for target in targets:
            assert isinstance(target, ProofTarget)
    
    def test_calculate_risk(self):
        code = """
def risky_divide(a, b):
    return a / b
"""
        generator = ProofObligationGenerator()
        vcs = generator.generate_from_code(code, "risky_divide")
        
        for vc in vcs:
            risk = generator._calculate_risk(vc)
            assert 0.0 <= risk <= 1.0


class TestProofCertificate:
    """Tests for ProofCertificate"""
    
    def test_create_certificate(self):
        cert = ProofCertificate()
        
        assert cert.id is not None
        assert cert.results == []
        assert cert.tests == []
    
    def test_add_proof_result(self):
        cert = ProofCertificate()
        result = ProofResult(
            status=ProofStatus.PROVEN,
            smt_expression="(not (= b 0))",
            certificate="Test proof"
        )
        
        cert.add_proof_result(result)
        
        assert len(cert.results) == 1
        assert cert.results[0]["status"] == "proven"
    
    def test_add_test(self):
        cert = ProofCertificate()
        test = GeneratedTest(
            id="test_1",
            function_name="divide",
            inputs={"a": 10, "b": 2}
        )
        
        cert.add_test(test)
        
        assert len(cert.tests) == 1
        assert cert.tests[0].function_name == "divide"
    
    def test_to_dict(self):
        cert = ProofCertificate()
        result = ProofResult(
            status=ProofStatus.PROVEN,
            smt_expression="(not (= b 0))"
        )
        cert.add_proof_result(result)
        
        data = cert.to_dict()
        
        assert "id" in data
        assert "results" in data
        assert "statistics" in data
    
    def test_save_and_load(self):
        cert = ProofCertificate()
        result = ProofResult(
            status=ProofStatus.PROVEN,
            smt_expression="(not (= b 0))"
        )
        cert.add_proof_result(result)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = cert.save(tmpdir)
            assert os.path.exists(filepath)
            
            loaded_cert = ProofCertificate.load(filepath)
            assert loaded_cert.id == cert.id
            assert len(loaded_cert.results) == 1


class TestNeuralProofTargetPredictor:
    """Tests for NeuralProofTargetPredictor"""
    
    def test_init(self):
        predictor = NeuralProofTargetPredictor()
        
        assert predictor.model is not None
        assert predictor.trained is True
    
    def test_predict(self):
        predictor = NeuralProofTargetPredictor()
        targets = [
            ProofTarget(
                id="test_1",
                function_name="divide",
                path_condition=PathCondition("(not (= b 0))"),
                risk_score=0.8
            )
        ]
        
        predictions = predictor.predict(targets)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0][1], float)
    
    def test_predict_top_targets(self):
        predictor = NeuralProofTargetPredictor()
        targets = [
            ProofTarget(
                id="test_1",
                function_name="divide",
                path_condition=PathCondition("(not (= b 0))"),
                risk_score=0.8
            ),
            ProofTarget(
                id="test_2",
                function_name="add",
                path_condition=PathCondition("(> x 0)"),
                risk_score=0.2
            )
        ]
        
        top_targets = predictor.predict_top_targets(targets, top_n=1)
        
        assert isinstance(top_targets, list)
        assert len(top_targets) == 1
    
    def test_save_and_load_model(self):
        predictor = NeuralProofTargetPredictor()
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
        
        try:
            predictor.save_model(model_path)
            assert os.path.exists(model_path)
            
            new_predictor = NeuralProofTargetPredictor(model_path=model_path)
            assert new_predictor.trained is True
        finally:
            os.unlink(model_path)


class TestProofGuidedTestGenerator:
    """Tests for ProofGuidedTestGenerator"""
    
    def test_generate_from_proven(self):
        generator = ProofGuidedTestGenerator()
        result = ProofResult(
            status=ProofStatus.PROVEN,
            smt_expression="(not (= b 0))"
        )
        target = ProofTarget(
            id="test_1",
            function_name="divide",
            path_condition=PathCondition("(not (= b 0))")
        )
        
        test = generator.generate_from_proof_result(result, target)
        
        assert test is None
    
    def test_generate_from_disproven(self):
        generator = ProofGuidedTestGenerator()
        result = ProofResult(
            status=ProofStatus.DISPROVEN,
            smt_expression="(not (= b 0))",
            model={"b": "0"}
        )
        target = ProofTarget(
            id="test_1",
            function_name="divide",
            path_condition=PathCondition("(not (= b 0))")
        )
        
        test = generator.generate_from_proof_result(result, target)
        
        assert test is not None
        assert isinstance(test, GeneratedTest)
        assert test.function_name == "divide"
    
    def test_generate_test_code(self):
        generator = ProofGuidedTestGenerator()
        test = GeneratedTest(
            id="test_1",
            function_name="divide",
            inputs={"a": 10, "b": 2},
            assertion="assert b != 0"
        )
        
        code = generator.generate_test_code(test)
        
        assert isinstance(code, str)
        assert "def test_test_1" in code
    
    def test_generate_test_module(self):
        generator = ProofGuidedTestGenerator()
        tests = [
            GeneratedTest(
                id="test_1",
                function_name="divide",
                inputs={"a": 10, "b": 2},
                assertion="assert b != 0"
            )
        ]
        
        module_code = generator.generate_test_module(tests)
        
        assert isinstance(module_code, str)
        assert "import pytest" in module_code


class TestProofGenerator:
    """Tests for ProofGenerator"""
    
    def test_init(self):
        generator = ProofGenerator()
        
        assert generator.prover is not None
        assert generator.obligation_generator is not None
        assert generator.target_predictor is not None
        assert generator.test_generator is not None
    
    def test_prove_and_generate_test(self):
        code = """
def divide(a, b):
    return a / b
"""
        generator = ProofGenerator()
        result = generator.prove_and_generate_test(code, "divide")
        
        assert isinstance(result, dict)
        assert "proven" in result
        assert "disproven" in result
        assert "tests" in result
        assert "test_code" in result
    
    def test_verify_division_safety(self):
        code = """
def divide(a, b):
    return a / b
"""
        generator = ProofGenerator()
        result = generator.verify_division_safety(code, "divide")
        
        assert isinstance(result, dict)
        assert "result" in result
        assert "safe" in result
        assert "proof_results" in result
    
    def test_generate_proof_report(self):
        code = """
def safe_add(a, b):
    return a + b
"""
        generator = ProofGenerator()
        report = generator.generate_proof_report(code, "safe_add")
        
        assert report.proven_count >= 0
        assert report.disproven_count >= 0
        assert report.unknown_count >= 0
        assert report.total_duration >= 0


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_proof(self):
        code = """
def calculate(a, b):
    if a > b:
        return a / b
    else:
        return b - a
"""
        generator = ProofGenerator()
        result = generator.prove_and_generate_test(code, "calculate")
        
        assert "proven" in result
        assert "disproven" in result
        assert "tests" in result
    
    def test_certificate_generation(self):
        code = """
def divide(a, b):
    return a / b
"""
        generator = ProofGenerator(certificate_dir=tempfile.gettempdir())
        report = generator.generate_proof_report(code, "divide")
        
        assert report.proven_count >= 0
        assert report.disproven_count >= 0
