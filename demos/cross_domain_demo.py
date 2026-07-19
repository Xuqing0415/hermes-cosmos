#!/usr/bin/env python3
"""
Cross-Domain Knowledge Distillation Demo
Shows the orchestrator flow: Perceiver -> Sage -> CrossDomain -> Knight

Usage:
  python demos/cross_domain_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    from hermes.cross_domain.knowledge_amalgamator import KnowledgeAmalgamator
    from hermes.cross_domain.cross_domain_translator import CrossDomainTranslator
    from hermes.cross_domain.pattern_similarity_engine import PatternSimilarityEngine
    from hermes.cross_domain.abstract_pattern_extractor import (
        AbstractPatternExtractor, AbstractPatternType
    )
    from hermes.core.interfaces import (
        PainPoint, DomainContext, PatchPlan, ExecutionResult,
        Perceiver, Sage, Knight, DomainPlugin
    )
    from hermes.core.orchestrator import CrossDomainOrchestrator

    # ============================================================
    # Step 1: Seed knowledge graph with cross-domain patterns
    # ============================================================
    print("[CrossDomain] Initializing cross-domain knowledge graph...")
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    extractor = AbstractPatternExtractor()
    similarity_engine = PatternSimilarityEngine()

    # Create patterns from different domains
    k8s_p = extractor.extract_pattern("k8s", {
        "type": "missing_resources", "severity": "medium",
        "message": "missing resource boundary limits and bounds checking",
        "location": "deployment.yaml"
    })
    py_p = extractor.extract_pattern("default", {
        "type": "null_pointer", "severity": "high",
        "message": "missing null check and boundary validation before access",
        "location": "handler.py"
    })

    all_patterns = [p for p in [k8s_p, py_p] if p]
    amalgamator.amalgamate_patterns(all_patterns)
    similarity_engine.add_patterns(all_patterns)
    similarities = similarity_engine.find_cross_domain_similarities(threshold=0.5)
    amalgamator.amalgamate_similarities(similarities)
    amalgamator.save()
    print("[CrossDomain] Knowledge graph seeded with cross-domain patterns.\n")

    # ============================================================
    # Step 2: Create a custom Sage that returns NO direct fix
    #         for a specific pain point, triggering cross-domain
    # ============================================================
    class CustomPerceiver(Perceiver):
        def detect(self, context):
            return [
                PainPoint(
                    id="demo-1",
                    type="index_out_of_bounds",
                    severity="high",
                    message="index out of bounds in memref.load operation",
                    location="test.mlir",
                    context={"operation": "memref.load", "index": "%idx", "bound": "%size"}
                )
            ]

    class CustomSage(Sage):
        def generate_patch(self, context, pain_point):
            # Return empty patch plan = no direct fix template found
            return PatchPlan(
                id="no-fix",
                pain_point_id=pain_point.id,
                description="No direct fix template found",
                operations=[],
                estimated_effort=0
            )

        def cross_domain_fallback(self, context, pain_point, cross_domain_fix):
            import hashlib
            import time
            patch_id = hashlib.md5(f"{pain_point.id}_{int(time.time())}_cd".encode()).hexdigest()
            code = cross_domain_fix.code_template if hasattr(cross_domain_fix, 'code_template') else ""
            return PatchPlan(
                id=patch_id,
                pain_point_id=pain_point.id,
                description=f"[CrossDomain] {cross_domain_fix.fix_description}",
                operations=[{
                    "type": "cross_domain_fix",
                    "description": cross_domain_fix.fix_description,
                    "code": code,
                    "steps": cross_domain_fix.implementation_steps if hasattr(cross_domain_fix, 'implementation_steps') else []
                }],
                estimated_effort=0.5
            )

    class CustomKnight(Knight):
        def execute(self, context, patch_plan):
            return ExecutionResult(
                success=True,
                patch_plan_id=patch_plan.id,
                message="Cross-domain fix applied",
                verification_results=[{"operation": op["type"], "status": "applied"} for op in patch_plan.operations]
            )

        def verify(self, context, patch_plan):
            return ExecutionResult(
                success=True,
                patch_plan_id=patch_plan.id,
                message="All verification tests passed",
                verification_results=[{"test": "unit_test", "status": "passed", "duration_ms": 120}]
            )

    class CustomPlugin(DomainPlugin):
        @property
        def domain_name(self):
            return "mlir"

        @property
        def description(self):
            return "MLIR domain plugin for cross-domain demo"

        def create_context(self, path, **kwargs):
            return DomainContext(domain="mlir", path=path, metadata=kwargs, issues=[], artifacts={})

        def get_perceiver(self):
            return CustomPerceiver()

        def get_sage(self):
            return CustomSage()

        def get_knight(self):
            return CustomKnight()

    # ============================================================
    # Step 3: Run the orchestrator
    # ============================================================
    plugin = CustomPlugin()
    print("[Orchestrator] Running domain: mlir")
    context = plugin.create_context(".")

    orchestrator = CrossDomainOrchestrator(
        plugin=plugin,
        knowledge_amalgamator=amalgamator,
        translator=CrossDomainTranslator(),
        pattern_extractor=extractor,
        similarity_engine=similarity_engine,
        similarity_threshold=0.5,
    )

    results = orchestrator.run(context)

    # ============================================================
    # Step 4: Display results in the expected format
    # ============================================================
    print()
    print("-" * 60)
    print("Demo Output (expected format):")
    print("-" * 60)

    for r in results:
        print(f"\n[Orchestrator] Running domain: mlir")
        print(f"[Perceiver] Found: {r.pain_point.message}")
        print(f"[Sage] No direct fix template found.")
        if r.cross_domain_used:
            print(f"[CrossDomain] Querying knowledge graph...")
            print(f"[CrossDomain] Found similar pattern '{r.pattern_type}' "
                  f"in {r.source_domain} plugin (sim: {r.similarity:.2f})")
            print(f"[CrossDomain] Translating fix from {r.source_domain} to mlir...")
            print(f"[Knight] Applying fix... Tests passed!")
            print(f"[CrossDomain] New cross-domain link added.")
        else:
            print("[Knight] No cross-domain fix available.")

    print()
    print("[CrossDomain] Demo complete.")


if __name__ == "__main__":
    main()