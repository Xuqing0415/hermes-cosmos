import pytest
import tempfile
import os

from hermes.cross_domain import (
    AbstractPatternExtractor,
    AbstractPatternType,
    PatternSimilarityEngine,
    CrossDomainTranslator,
    KnowledgeAmalgamator,
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