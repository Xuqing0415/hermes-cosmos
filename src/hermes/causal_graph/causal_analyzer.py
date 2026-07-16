from typing import List, Dict, Optional, Any
from .types import CausalNode, CausalEdge, CausalGraph, CausalExplanation, NodeType
import re

class CausalAnalyzer:
    """Analyzes task results and generates causal explanations for debugging."""
    
    def __init__(self):
        self.explanation_history: List[CausalExplanation] = []
        self.pattern_db: Dict[str, Dict[str, Any]] = self._initialize_pattern_db()
    
    def _initialize_pattern_db(self) -> Dict[str, Dict[str, Any]]:
        return {
            "ZeroDivisionError": {
                "root_cause": "Division by zero without guard check",
                "intermediate": "Missing input validation",
                "symptom": "Runtime crash",
                "recommendation": "Add guard: if divisor == 0: raise ValueError"
            },
            "TypeError": {
                "root_cause": "Type mismatch in operation",
                "intermediate": "Missing type checking",
                "symptom": "Runtime type error",
                "recommendation": "Add type validation or conversion"
            },
            "AttributeError": {
                "root_cause": "Accessing attribute on None or wrong type",
                "intermediate": "Missing null check",
                "symptom": "Attribute access failure",
                "recommendation": "Add None check before attribute access"
            },
            "IndexError": {
                "root_cause": "Array index out of bounds",
                "intermediate": "Missing bounds check",
                "symptom": "Runtime index error",
                "recommendation": "Validate index before access"
            },
            "KeyError": {
                "root_cause": "Dictionary key not found",
                "intermediate": "Missing key existence check",
                "symptom": "Runtime key error",
                "recommendation": "Use dict.get() or check key existence"
            },
            "ImportError": {
                "root_cause": "Module not found or not installed",
                "intermediate": "Missing dependency",
                "symptom": "Import failure",
                "recommendation": "Install missing package or fix import path"
            },
            "performance": {
                "root_cause": "Inefficient algorithm or data structure",
                "intermediate": "O(n^2) or worse complexity",
                "symptom": "Slow execution time",
                "recommendation": "Optimize algorithm or use caching"
            }
        }
    
    def analyze_error(self, error_type: str, error_message: str, 
                      context: Dict[str, Any] = None) -> CausalExplanation:
        context = context or {}
        pattern = self.pattern_db.get(error_type, {
            "root_cause": f"Unknown error: {error_type}",
            "intermediate": "Insufficient error context",
            "symptom": error_message,
            "recommendation": "Add more logging and context"
        })
        
        graph = CausalGraph()
        
        root_node = CausalNode(
            node_id="root",
            label=pattern["root_cause"],
            node_type=NodeType.ROOT_CAUSE,
            description=f"Root cause of {error_type}",
            evidence=[error_message],
            confidence=0.8
        )
        graph.add_node(root_node)
        
        intermediate_node = CausalNode(
            node_id="intermediate",
            label=pattern["intermediate"],
            node_type=NodeType.INTERMEDIATE,
            description="Intermediate cause",
            confidence=0.7
        )
        graph.add_node(intermediate_node)
        graph.add_edge(CausalEdge("root", "intermediate", "causes", 0.8))
        
        symptom_node = CausalNode(
            node_id="symptom",
            label=pattern["symptom"],
            node_type=NodeType.SYMPTOM,
            description=error_message,
            confidence=0.9
        )
        graph.add_node(symptom_node)
        graph.add_edge(CausalEdge("intermediate", "symptom", "manifests_as", 0.8))
        
        for key, value in context.items():
            if isinstance(value, str) and value:
                obs_node = CausalNode(
                    node_id=f"obs_{key}",
                    label=f"{key}: {value[:50]}",
                    node_type=NodeType.OBSERVATION,
                    description=value,
                    confidence=0.6
                )
                graph.add_node(obs_node)
                graph.add_edge(CausalEdge("symptom", obs_node.node_id, "observed_in", 0.5))
        
        explanation_text = self._generate_explanation(graph, error_type, pattern)
        recommendations = [pattern["recommendation"]]
        
        explanation = CausalExplanation(
            graph=graph,
            root_causes=[root_node],
            explanation=explanation_text,
            recommendations=recommendations,
            confidence=0.75
        )
        
        self.explanation_history.append(explanation)
        return explanation
    
    def _generate_explanation(self, graph: CausalGraph, error_type: str, 
                              pattern: Dict[str, Any]) -> str:
        lines = [f"Causal Analysis for {error_type}:", ""]
        lines.append(f"Root Cause: {pattern['root_cause']}")
        lines.append(f"Intermediate: {pattern['intermediate']}")
        lines.append(f"Symptom: {pattern['symptom']}")
        lines.append("")
        lines.append("Causal Chain:")
        for edge in graph.edges:
            source = graph.get_node(edge.source_id)
            target = graph.get_node(edge.target_id)
            if source and target:
                lines.append(f"  [{source.label}] --{edge.relation}--> [{target.label}]")
        return "\n".join(lines)
    
    def analyze_test_failure(self, test_result: Dict[str, Any]) -> CausalExplanation:
        error_type = test_result.get("error_type", "Unknown")
        error_message = test_result.get("error", test_result.get("error_message", ""))
        context = {
            "function": test_result.get("function_name", ""),
            "test_name": test_result.get("test_name", ""),
            "input": str(test_result.get("input", ""))
        }
        return self.analyze_error(error_type, error_message, context)
    
    def analyze_task_result(self, task_result: Dict[str, Any]) -> CausalExplanation:
        if task_result.get("error"):
            return self.analyze_error("RuntimeError", str(task_result["error"]), task_result)
        
        failed_tests = task_result.get("failed_tests", [])
        if failed_tests:
            first_failure = failed_tests[0]
            return self.analyze_test_failure(first_failure)
        
        return CausalExplanation(
            graph=CausalGraph(),
            root_causes=[],
            explanation="No errors detected in task result",
            recommendations=[],
            confidence=1.0
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_analyses": len(self.explanation_history),
            "pattern_count": len(self.pattern_db),
            "avg_confidence": sum(e.confidence for e in self.explanation_history) / len(self.explanation_history) if self.explanation_history else 0.0
        }
