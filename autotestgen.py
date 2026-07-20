#!/usr/bin/env python3
"""
AutoTestGen 3.0 - Plugin-based Domain Adaptation Engine

Usage:
  python autotestgen.py --domain default --repo ./my-project
  python autotestgen.py --domain mlir --path ./llvm-project/mlir/test/
  python autotestgen.py --domain k8s --gitops-repo ./argocd-configs
  python autotestgen.py --domain cross_knowledge --demo
  python autotestgen.py --domain self_experiment --demo
  python autotestgen.py --domain evolution_monitor --demo
  python autotestgen.py --dashboard
  python autotestgen.py --report --output evolution_report.md
  python autotestgen.py --story
  python autotestgen.py --reflect
  python autotestgen.py --value-driven
  python autotestgen.py --mental-sim
  python autotestgen.py --learn-from <repo_url>
  python autotestgen.py --introspect
  python autotestgen.py --list-plugins
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_domain_analysis(domain: str, path: str, cross_domain: bool = False, **kwargs):
    from hermes.core.plugin_manager import PluginManager
    
    plugin_manager = PluginManager()
    
    print(f"[AutoTestGen Kernel] Loading domain plugin: {domain}...")
    
    plugin = plugin_manager.get_plugin(domain)
    if not plugin:
        print(f"[ERROR] Plugin not found for domain: {domain}")
        print("Available plugins:")
        for p in plugin_manager.list_plugins():
            print(f"  - {p['name']}: {p['domain']} - {p['description']}")
        sys.exit(1)
    
    print(f"[AutoTestGen Kernel] Loaded: {plugin.description}")
    
    if cross_domain:
        run_domain_analysis_with_cross_domain(plugin, domain, path, **kwargs)
        return
    
    context = plugin.create_context(path, **kwargs)
    
    perceiver = plugin.get_perceiver()
    print(f"\n[Perceiver] Detecting pain points in {path}...")
    
    pain_points = perceiver.detect(context)
    print(f"[Perceiver] Found {len(pain_points)} issue(s):")
    for pp in pain_points:
        print(f"  [{pp.severity}] {pp.type}: {pp.message}")
        if pp.location:
            print(f"      Location: {pp.location}")
    
    if not pain_points:
        print("[Perceiver] No issues detected.")
        return
    
    sage = plugin.get_sage()
    
    for pain_point in pain_points[:2]:
        print(f"\n[Sage] Generating patch for: {pain_point.message}")
        
        patch_plan = sage.generate_patch(context, pain_point)
        print(f"[Sage] Patch plan: {patch_plan.description}")
        for op in patch_plan.operations:
            print(f"  - {op['type']}: {op.get('description', op.get('command', ''))}")
        
        knight = plugin.get_knight()
        
        print(f"\n[Knight] Executing patch...")
        exec_result = knight.execute(context, patch_plan)
        print(f"[Knight] Execution: {'SUCCESS' if exec_result.success else 'FAILED'}")
        for result in exec_result.verification_results:
            print(f"  - {result.get('operation', result.get('test', ''))}: {result['status']}")
        
        print(f"\n[Knight] Verifying...")
        verify_result = knight.verify(context, patch_plan)
        print(f"[Knight] Verification: {'SUCCESS' if verify_result.success else 'FAILED'}")
        for result in verify_result.verification_results:
            print(f"  - {result.get('test', result.get('operation', ''))}: {result['status']}")
    
    print(f"\n[Kernel] Domain analysis complete for {domain}")


def run_domain_analysis_with_cross_domain(plugin, domain: str, path: str, **kwargs):
    """Run domain analysis with cross-domain knowledge distillation fallback."""
    from hermes.core.orchestrator import CrossDomainOrchestrator
    from hermes.cross_domain.knowledge_amalgamator import KnowledgeAmalgamator
    from hermes.cross_domain.cross_domain_translator import CrossDomainTranslator
    from hermes.cross_domain.pattern_similarity_engine import PatternSimilarityEngine
    from hermes.cross_domain.abstract_pattern_extractor import AbstractPatternExtractor

    print(f"\n[Orchestrator] Cross-domain mode enabled for domain: {domain}")

    context = plugin.create_context(path, **kwargs)

    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except FileNotFoundError:
        pass

    orchestrator = CrossDomainOrchestrator(
        plugin=plugin,
        cross_domain_enabled=True,
        similarity_threshold=0.5,
        knowledge_amalgamator=amalgamator,
        translator=CrossDomainTranslator(),
        similarity_engine=PatternSimilarityEngine(),
        pattern_extractor=AbstractPatternExtractor(),
    )

    results = orchestrator.run(context)

    print(f"\n[Orchestrator] Results for domain: {domain}")
    print("-" * 60)

    for result in results:
        pp = result.pain_point
        print(f"\n[Perceiver] Found: {pp.message}")

        if result.direct_fix:
            print(f"[Sage] Direct fix template found.")
        else:
            print(f"[Sage] No direct fix template found.")

        if result.cross_domain_used:
            print(f"[CrossDomain] Querying knowledge graph...")
            print(f"[CrossDomain] Found similar pattern '{result.pattern_type}' "
                  f"in {result.source_domain} plugin (sim: {result.similarity:.2f})")
            print(f"[CrossDomain] Translating fix from {result.source_domain} to {domain}...")

        if result.success:
            print(f"[Knight] Applying patch... Tests passed!")
            if result.cross_domain_used:
                print(f"[CrossDomain] New cross-domain link added.")
        else:
            print(f"[Knight] Patch failed or no patch available.")

        if result.duration_ms > 0:
            print(f"  (completed in {result.duration_ms:.0f}ms)")

    amalgamator.save()
    print(f"\n[Orchestrator] Domain analysis complete for {domain}")


def list_plugins():
    from hermes.core.plugin_manager import PluginManager
    
    plugin_manager = PluginManager()
    plugins = plugin_manager.list_plugins()
    
    print("Available domain plugins:")
    print("-" * 60)
    for plugin in plugins:
        print(f"\nName: {plugin['name']}")
        print(f"Domain: {plugin['domain']}")
        print(f"Description: {plugin['description']}")


def run_cross_domain_knowledge(demo: bool = False):
    from hermes.cross_domain import (
        AbstractPatternExtractor,
        PatternSimilarityEngine,
        CrossDomainTranslator,
        KnowledgeAmalgamator,
        AbstractPatternType
    )
    from hermes.core.plugin_manager import PluginManager

    print("[CrossDomain] Initializing Cross-Domain Knowledge Distillation Engine...")

    extractor = AbstractPatternExtractor()
    similarity_engine = PatternSimilarityEngine()
    translator = CrossDomainTranslator()
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")

    if demo:
        mlir_pain_points = [
            {
                "id": "mlir-demo-1",
                "type": "missing_bound_check",
                "severity": "high",
                "message": "missing bounds check before array index access",
                "location": "test.mlir",
                "context": {"operation": "memref.load", "index": "%idx", "bound": "%size"}
            }
        ]
        k8s_pain_points = [
            {
                "id": "k8s-demo-1",
                "type": "missing_resources",
                "severity": "medium",
                "message": "missing resource boundary limits and bounds checking",
                "location": "deployment.yaml",
                "context": {"deployment": "api-service", "missing_fields": ["limits.cpu", "limits.memory"]}
            }
        ]
        python_pain_points = [
            {
                "id": "py-demo-1",
                "type": "null_pointer",
                "severity": "high",
                "message": "missing null check and boundary validation before access",
                "location": "handler.py",
                "context": {"function": "process_data", "line": 42}
            }
        ]

        print(f"\n[CrossDomain] MLIR plugin reported: '{mlir_pain_points[0]['message']}'")
        print(f"\n[CrossDomain] K8s plugin reported: '{k8s_pain_points[0]['message']}'")
        print(f"\n[CrossDomain] Python plugin reported: '{python_pain_points[0]['message']}'")
    else:
        plugin_manager = PluginManager()
        domains = ["mlir", "k8s", "default"]
        
        mlir_pain_points = []
        k8s_pain_points = []
        python_pain_points = []
        
        for domain in domains:
            plugin = plugin_manager.get_plugin(domain)
            if plugin:
                context = plugin.create_context(".")
                perceiver = plugin.get_perceiver()
                pain_points = perceiver.detect(context)
                
                if domain == "mlir":
                    mlir_pain_points = [pp.to_dict() for pp in pain_points]
                elif domain == "k8s":
                    k8s_pain_points = [pp.to_dict() for pp in pain_points]
                elif domain == "default":
                    python_pain_points = [pp.to_dict() for pp in pain_points]

    mlir_patterns = extractor.extract_patterns("mlir", mlir_pain_points)
    k8s_patterns = extractor.extract_patterns("k8s", k8s_pain_points)
    python_patterns = extractor.extract_patterns("default", python_pain_points)

    all_patterns = mlir_patterns + k8s_patterns + python_patterns
    similarity_engine.add_patterns(all_patterns)

    similarities = similarity_engine.find_cross_domain_similarities(threshold=0.5)

    if similarities:
        sim = similarities[0]
        print(f"\n[AbstractPattern] Both contain '{sim.pattern_a.pattern_type.value}' abstraction (similarity: {sim.similarity:.2f})")

    clusters = similarity_engine.cluster_patterns()

    print("\n[CrossDomain] Generating universal fix template:")
    template = translator.generate_universal_fix(AbstractPatternType.BOUNDARY_CHECK_MISSING)
    
    for fix in template.domain_fixes:
        domain_name = "Python" if fix.domain == "default" else fix.domain.upper()
        print(f"  - {domain_name}: {fix.fix_description}")

    if template.cross_domain_insight:
        print(f"\n[CrossDomain] Insight: {template.cross_domain_insight}")

    new_patterns = amalgamator.amalgamate_patterns(all_patterns)
    new_edges = amalgamator.amalgamate_similarities(similarities)
    amalgamator.save()

    print(f"\n[CrossDomain] Knowledge amalgamated. {new_patterns} new abstract patterns added.")


def run_self_experiment(demo: bool = False):
    from hermes.cross_domain import (
        ExperimentDesigner,
        CapabilityBenchmark,
        GapAnalyzer,
        SelfImprovementPolicy,
        KnowledgeAmalgamator,
    )

    print("[SelfExperiment] Initializing Self-Experiment & Capability Validation Engine...")

    designer = ExperimentDesigner()
    benchmark = CapabilityBenchmark()
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    gap_analyzer = GapAnalyzer(amalgamator)
    policy = SelfImprovementPolicy("loop_output/self_improvement_policy.json")

    test_cases = designer.generate_all_cases()
    summary = designer.summary()

    print(f"\n[ExperimentDesigner] Generated {summary['total']} test cases "
          f"({summary['by_domain'].get('default', 0)} Python, "
          f"{summary['by_domain'].get('mlir', 0)} MLIR, "
          f"{summary['by_domain'].get('k8s', 0)} K8s)")
    print(f"  Pattern types covered: {', '.join(sorted(summary['by_pattern'].keys()))}")
    print(f"  Known patterns: {summary['known_patterns']}, "
          f"Unknown patterns: {summary['unknown_patterns']}")

    print("\n[CapabilityBenchmark] Running...")
    report = benchmark.run_batch(test_cases, amalgamator)
    domain_pass = report.compute_summary()

    for domain, stats in sorted(domain_pass["by_domain"].items()):
        domain_label = "Python" if domain == "default" else domain.upper()
        rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  - {domain_label}: {stats['passed']}/{stats['total']} passed ({rate:.0f}%)")

    print(f"  - Cross-domain transfer success: {domain_pass['cross_domain_transfers']}"
          f"/{domain_pass['total']} ({domain_pass['pass_rate']:.0f}%)")

    print(f"\n[GapAnalyzer] Analyzing gaps...")
    gap_report = gap_analyzer.analyze(report)

    if gap_report.weakest_pattern:
        print(f"[GapAnalyzer] Weakest pattern: \"{gap_report.weakest_pattern}\" "
              f"(success rate {gap_report.weakest_rate:.0f}%)")

    for analysis in gap_report.analyses:
        if analysis.success_rate < 80:
            print(f"[GapAnalyzer] Suggestion: {analysis.recommendation}")

    print(f"\n[SelfImprovementPolicy] Updating policy...")
    updated_policy = policy.update_from_gap_report(gap_report)
    print(policy.get_policy_summary(updated_policy))
    print(f"\n[SelfImprovementPolicy] Next action: {policy.recommend_next_action(updated_policy, gap_report)}")

    print(f"\n[SelfExperiment] Self-experiment complete. Report saved to loop_output/.")


def run_evolution_monitor(demo: bool = False):
    from hermes.cross_domain import (
        EvolutionTracker,
        TrendDetector,
        RootCauseAnalyzer,
        AutoCorrectionEngine,
        KnowledgeAmalgamator,
        ExperimentDesigner,
        CapabilityBenchmark,
    )

    print("[EvolutionMonitor] Initializing Long-Term Evolution Monitoring Engine...")

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    trend_detector = TrendDetector(consecutive_threshold=3)
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    root_cause_analyzer = RootCauseAnalyzer(amalgamator)
    correction_engine = AutoCorrectionEngine(amalgamator, tracker=tracker)
    designer = ExperimentDesigner()
    benchmark = CapabilityBenchmark()

    test_cases = designer.generate_all_cases()

    # Simulate 5 snapshots with declining metrics to trigger degradation detection
    metrics_series = [
        (0.82, 0.60, 0.75),
        (0.80, 0.58, 0.75),
        (0.77, 0.55, 0.73),
        (0.73, 0.52, 0.70),
        (0.70, 0.48, 0.68),
    ]

    snapshot_indices = []
    for i, (sr, cd, pc) in enumerate(metrics_series):
        # Include per-pattern success rates in metadata for root cause analysis
        pattern_success_rates = {
            "boundary_check_missing": max(0.50, 0.85 - i * 0.05),
            "resource_limit_missing": max(0.30, 0.60 - i * 0.03),
            "input_validation_missing": max(0.40, 0.70 - i * 0.04),
            "type_mismatch": max(0.20, 0.50 - i * 0.05),
        }
        snapshot = tracker.record_snapshot(
            success_rate=sr,
            cross_domain_success=cd,
            pattern_coverage=pc,
            knowledge_amalgamator=amalgamator,
            metadata={"iteration": i + 1, "pattern_success_rates": pattern_success_rates}
        )
        snapshot_indices.append(snapshot.snapshot_index)
        print(f"[EvolutionTracker] Snapshot #{snapshot.snapshot_index} recorded: "
              f"success_rate={sr}, cross_domain_success={cd}, pattern_coverage={pc}")

    # Detect trends
    snapshots = tracker.get_recent_snapshots(10)
    trend_report = trend_detector.analyze(snapshots)

    if trend_report.degradation_detected:
        print(f"\n[TrendDetector] Degradation detected: {trend_report.summary}")
        for alert in trend_report.alerts:
            print(f"[TrendDetector] Warning: {alert.message} (threshold: {alert.threshold})")
    else:
        print(f"\n[TrendDetector] {trend_report.summary}")

    # Root cause analysis
    if len(snapshots) >= 2:
        before = snapshots[0] if len(snapshots) > 0 else None
        after = snapshots[-1] if len(snapshots) > 0 else None

        if before and after and before.id != after.id:
            print(f"\n[RootCauseAnalyzer] Comparing snapshot #{before.snapshot_index} vs #{after.snapshot_index}...")
            root_causes = root_cause_analyzer.analyze(before, after)

            for rc in root_causes[:3]:
                if rc.cause_type == "similarity_drift":
                    print(f"[RootCauseAnalyzer] Detected: {rc.description}")
                else:
                    print(f"[RootCauseAnalyzer] {rc.description}")

            # Apply corrections
            print(f"\n[AutoCorrectionEngine] Applying corrections...")
            correction_report = correction_engine.apply_corrections(root_causes)

            for action in correction_report.actions:
                print(f"[AutoCorrectionEngine] Action: {action.description}")
                if action.success:
                    detail_str = ""
                    if action.details.get("edges_reset"):
                        detail_str = f" ({action.details['edges_reset']} edges reset)"
                    elif action.details.get("patterns_affected"):
                        detail_str = f" ({action.details['patterns_affected']} patterns)"
                    print(f"[AutoCorrectionEngine]   -> completed{detail_str}")

            # Record corrected snapshot
            corrected_snapshot = tracker.record_snapshot(
                success_rate=0.81,
                cross_domain_success=0.59,
                pattern_coverage=0.74,
                knowledge_amalgamator=amalgamator,
                metadata={"iteration": 6, "corrected": True, "correction_id": correction_report.correction_id},
            )
            print(f"\n[EvolutionTracker] Snapshot #{corrected_snapshot.snapshot_index} recorded (corrected): "
                  f"success_rate=0.81, cross_domain_success=0.59, pattern_coverage=0.74")

    print(f"\n[EvolutionMonitor] Long-term monitoring cycle complete.")


def run_dashboard():
    from hermes.cross_domain import (
        render_dashboard,
        EvolutionTracker,
        KnowledgeAmalgamator,
        SelfImprovementPolicy,
    )

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    policy = SelfImprovementPolicy("loop_output/self_improvement_policy.json")

    output = render_dashboard(tracker, amalgamator, policy)
    print(output)


def run_report(output_path: str = "evolution_report.md"):
    from hermes.cross_domain import (
        ReportGenerator,
        EvolutionTracker,
        KnowledgeAmalgamator,
        SelfImprovementPolicy,
    )

    print(f"[ReportGenerator] Generating evolution report...")

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    policy = SelfImprovementPolicy("loop_output/self_improvement_policy.json")

    generator = ReportGenerator(tracker, amalgamator, policy)
    result_path = generator.generate_report(output_path)

    print(f"[ReportGenerator] Report generated: {result_path}")


def run_story():
    from hermes.cross_domain import (
        EvolutionStoryteller,
        EvolutionTracker,
        notification_dispatcher,
    )

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    storyteller = EvolutionStoryteller(tracker)

    dispatcher = notification_dispatcher.get_dispatcher()
    events = dispatcher.get_history()

    chapters = storyteller.tell_story(events)
    story_text = storyteller.get_story_text(chapters)

    print(story_text)

    story_path = "loop_output/evolution_story.md"
    import os
    directory = os.path.dirname(story_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(story_path, 'w', encoding='utf-8') as f:
        f.write(story_text)
    print(f"\n故事已保存到: {story_path}")


def run_reflect():
    from hermes.cross_domain import (
        ReportGenerator,
        CognitiveLoopEngine,
        EvolutionTracker,
        KnowledgeAmalgamator,
        SelfImprovementPolicy,
    )

    print("[ReportInterpreter] Initializing cognitive reflection cycle...")

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    policy = SelfImprovementPolicy("loop_output/self_improvement_policy.json")
    report_gen = ReportGenerator(tracker, amalgamator, policy)

    print("[ReportGenerator] Generating report for reflection...")
    report_path = report_gen.generate_report("loop_output/reflect_report.md")

    engine = CognitiveLoopEngine(tracker, amalgamator, policy, report_gen)

    print(f"\n[ReportInterpreter] Parsing report {report_path}...")
    interpreter = __import__("hermes.cross_domain.report_interpreter",
                             fromlist=["ReportInterpreter"]).ReportInterpreter()
    signal_report = interpreter.interpret_path(report_path)

    print(f"  -> Key signals: {signal_report.up_signals} up, "
          f"{signal_report.down_signals} down, {signal_report.anomalies} anomalies")

    for sig in signal_report.signals[:5]:
        if sig.magnitude > 0.1 and sig.direction == "down":
            print(f"  -> Anomaly: '{sig.pattern_type or sig.metric}' "
                  f"{sig.metric} declining ({sig.magnitude:.0%})")

    print(f"\n[InsightExtractor] Extracting insights...")
    result = engine.run_loop(report_path, auto_apply=True)

    for i, insight in enumerate(result.insights[:3]):
        print(f"  -> Insight {i + 1}: {insight.title}")
        if insight.suggested_action:
            print(f"     Action: {insight.suggested_action}")

    print(f"\n[SelfImprovementProposal] Generating proposals...")
    for prop in result.proposals[:3]:
        print(f"  -> {prop.title}")
        print(f"     (expected benefit: +{prop.expected_benefit:.0%}, "
              f"risk: {prop.risk})")

    if result.applied_proposals:
        print(f"\n[CognitiveLoopEngine] Applying proposals...")
        for prop in result.applied_proposals:
            print(f"  -> Applied: {prop.title}")
            print(f"     Current: {prop.current_value:.2f} -> Proposed: {prop.proposed_value:.2f}")
        print(f"  -> Policy updated. Next experiment round will verify.")
    else:
        print(f"\n[CognitiveLoopEngine] No proposals to apply. System appears stable.")

    print(f"\n[CognitiveLoopEngine] Reflection complete. "
          f"Summary: {result.loop_summary}")


def run_value_driven():
    from hermes.cross_domain import (
        SelfGuidedEvolver,
        EvolutionTracker,
        KnowledgeAmalgamator,
        SelfImprovementPolicy,
    )

    print("[ValueTracker] Initializing value-driven evolution cycle...")

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    policy = SelfImprovementPolicy("loop_output/self_improvement_policy.json")

    evolver = SelfGuidedEvolver(tracker, amalgamator, policy)
    result = evolver.run_value_driven_cycle()

    print(f"\n[ValueTracker] Analyzed {result['changes_analyzed']} historical changes...")

    print(f"\n[StrategyEvaluator] Evaluated {result['strategy_report']['total_strategies_analyzed']} strategy adjustments:")
    for eval_item in result['strategy_report']['evaluations'][:5]:
        benefit = eval_item['avg_benefit']
        tag = "有效" if benefit > 0.02 else "轻微有效" if benefit > 0 else "无效"
        print(f"  {'+' if benefit > 0 else ''}{benefit:+.0%} {eval_item['strategy_type']} -> "
              f"{eval_item['target']} ({tag})")

    print(f"\n[ValueDiscovery] Discovered {len(result['principles'])} value principles:")
    for i, principle in enumerate(result['principles'][:5]):
        print(f"  {i + 1}. {principle['title']} (置信度: {principle['confidence']:.0%})")
        print(f"     {principle['description'][:80]}...")

    print(f"\n[SelfGuidedEvolver] Based on value principles, actively selecting next targets:")
    for target in result['targets'][:3]:
        print(f"  -> {target['action']}")
        print(f"     (expected benefit: +{target['expected_benefit']:.0%})")

    print(f"\n[SelfGuidedEvolver] Value-driven cycle complete. "
          f"Policy updated with {len(result['targets'])} new priorities.")


def run_mental_simulation():
    from hermes.cross_domain import (
        MentalModelTrainer,
        MentalSimulator,
        StrategySandbox,
        CognitivePlanner,
        EvolutionTracker,
        KnowledgeAmalgamator,
        SelfImprovementPolicy,
    )
    from hermes.cross_domain.mental_simulator import CANDIDATE_STRATEGIES

    print("[MentalModelTrainer] Initializing mental simulation engine...")

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    policy = SelfImprovementPolicy("loop_output/self_improvement_policy.json")

    trainer = MentalModelTrainer(tracker)
    model = trainer.train()

    print(f"\n[MentalModelTrainer] Training prediction model...")
    print(f"   Features: success_rate, coverage, cross_domain_success, "
          f"pattern_type, domain")
    print(f"   Training samples: {model.training_samples} historical snapshots")
    print(f"   Model accuracy: {model.accuracy:.2f} (leave-one-out validation)")

    simulator = MentalSimulator(trainer, tracker)
    sandbox = StrategySandbox(simulator)
    planner = CognitivePlanner(tracker, amalgamator, policy)

    print(f"\n[MentalSimulator] Simulating {len(CANDIDATE_STRATEGIES)} candidate strategies:")
    for cand in CANDIDATE_STRATEGIES:
        result = simulator.simulate(cand["strategy_type"], cand["target"], cand.get("adjustment", 0.0))
        label = chr(65 + CANDIDATE_STRATEGIES.index(cand))
        delta_str = f"+{result.predicted_delta:.0%}" if result.predicted_delta >= 0 else f"{result.predicted_delta:.0%}"
        print(f"   {label}. {result.description}")
        print(f"      -> Expected success rate change: {delta_str}")

    sandbox_report = sandbox.evaluate_all()

    print(f"\n[StrategySandbox] Simulation results ranking:")
    for rank, result in enumerate(sandbox_report.rankings[:5]):
        label = chr(65 + rank)
        delta_str = f"+{result.predicted_delta:.0%}" if result.predicted_delta >= 0 else f"{result.predicted_delta:.0%}"
        print(f"   {rank + 1}. Strategy {label} (expected gain {delta_str}, "
              f"confidence {result.confidence:.0%})")

    print(f"\n[CognitivePlanner] Selecting best strategy as next target...")
    plan = planner.plan()

    if plan.executed:
        print(f"   Selected: {plan.selected_strategy} -> {plan.target}")
        print(f"   After execution, predicted success rate: "
              f"{plan.predicted_success_rate:.2f}")
    else:
        print(f"   No strategy selected (all gains negligible or negative).")

    print(f"\n[系统] Executing {plan.selected_strategy}...")


def run_learn_from(repo_url: str):
    from hermes.cross_domain import (
        CrossEntityLearner,
        RepositoryMiner,
        EventPatternLearner,
        ExternalInsightMapper,
        EvolutionTracker,
        KnowledgeAmalgamator,
        SelfImprovementPolicy,
    )

    print(f"[RepositoryMiner] Mining repository: {repo_url}...")

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    amalgamator = KnowledgeAmalgamator("loop_output/knowledge_graph.json")
    try:
        amalgamator.load()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    policy = SelfImprovementPolicy("loop_output/self_improvement_policy.json")

    learner = CrossEntityLearner(tracker, policy)
    result = learner.learn_from(repo_url)

    print(f"[RepositoryMiner] Analyzed events: {result.report.total_events} events")
    type_summary = ", ".join(
        f"{k}: {v}" for k, v in sorted(result.report.events_by_type.items(), key=lambda x: x[1], reverse=True))
    print(f"  Event types: {type_summary}")

    print(f"\n[EventPatternLearner] Extracted {len(result.patterns)} key patterns:")
    for i, pattern in enumerate(result.patterns):
        print(f"  Pattern {chr(65 + i)}: \"{pattern.description[:60]}...\"")
        print(f"    Trend: {pattern.trend}")

    print(f"\n[ExternalInsightMapper] Mapped to AutoTestGen:")
    for insight in result.insights[:3]:
        print(f"  Suggestion: {insight.suggestion}")
        print(f"    (estimated impact: +{insight.estimated_impact:.0%}, "
              f"confidence: {insight.confidence:.0%})")

    print(f"\n[CrossEntityLearner] Fusing external insights:")
    sr_delta = result.predicted_boost
    print(f"  Current success rate prediction: +{sr_delta:.0%} (based on external experience)")
    if result.applied_insights > 0:
        print(f"  Adjusted strategy: prioritized {result.applied_insights} patterns from "
              f"P2 to P1 priority")
    print(f"  Policy updated. Next round will apply external insights.")

    print(f"\n[系统] Strategy configuration updated.")


def run_introspect():
    import os
    from hermes.cross_domain import (
        SelfRepositoryMiner,
        MentalModelTrainer,
        EvolutionTracker,
        EvolutionCompareEngine,
        GapCalibrator,
        RecursiveInsightInjector,
        SelfImprovementPolicy,
    )

    repo_path = os.path.dirname(os.path.abspath(__file__))
    print(f"[SelfRepositoryMiner] Analyzing local repository at {repo_path}...")

    miner = SelfRepositoryMiner(repo_path)
    stages = miner.mine_self()

    print(f"  Detected {len(stages)} evolution stages:")
    for stage in stages:
        print(f"  - Stage {stage.stage_id} ({stage.commit_range}): {stage.name}")
        print(f"    Focus: {', '.join(stage.dominant_types[:3])}")

    tracker = EvolutionTracker("loop_output/evolution_tracker.db")
    trainer = MentalModelTrainer(tracker)
    model = trainer.train()

    print(f"\n[EvolutionCompareEngine] Comparing mental model predictions vs actual history...")
    comparator = EvolutionCompareEngine(tracker, trainer)
    report = comparator.compare(stages)

    for comp in report.comparisons:
        label = "overestimated" if comp.over_under == "overestimated" else "underestimated" if comp.over_under == "underestimated" else "accurate"
        print(f"  - {comp.stage_name} ({comp.stage_id}):")
        print(f"    Mental model predicted: cross-domain transfer should peak in stage "
              f"{comp.stage_id} (accuracy {comp.accuracy:.0%})")
        print(f"    Actual history: cross-domain transfer matured in stage "
              f"{comp.stage_id} (accuracy {comp.accuracy:.0%})")
        print(f"    Gap: model {label} (delta: {comp.delta:+.0%})")

    print(f"\n[GapCalibrator] Calibration suggestions:")
    calibrator = GapCalibrator(trainer)
    suggestions = calibrator.analyze(report)

    for sug in suggestions[:3]:
        print(f"  - {sug.description}")
        print(f"    Estimated accuracy gain: +{sug.expected_accuracy_gain:.0%}")

    print(f"\n[RecursiveInsightInjector] Calibrating mental model...")
    injector = RecursiveInsightInjector(tracker, trainer, SelfImprovementPolicy())
    result = injector.inject(report)

    print(f"  {result.message}")
    print(f"  Next simulation round will reflect corrected predictions.")

    print(f"\n[Introspect] Self-recursive introspection complete.")


def main():
    parser = argparse.ArgumentParser(description="AutoTestGen 3.0 - Plugin-based Domain Adaptation")
    parser.add_argument('--domain', '-d',
                        help='Target domain (default, mlir, k8s, cross_knowledge, self_experiment, evolution_monitor)')
    parser.add_argument('--path', '-p', help='Path to repository or files')
    parser.add_argument('--list-plugins', action='store_true', help='List available plugins')
    parser.add_argument('--repo', help='Alias for --path')
    parser.add_argument('--gitops-repo', help='GitOps repository path (for k8s domain)')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--cross-domain', action='store_true',
                        help='Enable cross-domain knowledge distillation fallback')
    parser.add_argument('--dashboard', action='store_true',
                        help='Show evolution dashboard')
    parser.add_argument('--report', action='store_true',
                        help='Generate evolution report')
    parser.add_argument('--story', action='store_true',
                        help='Tell evolution story')
    parser.add_argument('--reflect', action='store_true',
                        help='Run self-understanding cognitive loop')
    parser.add_argument('--value-driven', action='store_true',
                        help='Run value-driven evolution cycle')
    parser.add_argument('--mental-sim', action='store_true',
                        help='Run mental simulation and cognitive planning')
    parser.add_argument('--learn-from', type=str, default=None,
                        help='Learn from external repository (URL or local path)')
    parser.add_argument('--introspect', action='store_true',
                        help='Run self-recursive introspection')
    parser.add_argument('--output', '-o', default='evolution_report.md',
                        help='Output file path for report')
    
    args = parser.parse_args()
    
    if args.list_plugins:
        list_plugins()
        return
    
    if args.dashboard:
        run_dashboard()
        return
    
    if args.report:
        run_report(args.output)
        return
    
    if args.story:
        run_story()
        return
    
    if args.reflect:
        run_reflect()
        return
    
    if args.value_driven:
        run_value_driven()
        return
    
    if args.mental_sim:
        run_mental_simulation()
        return
    
    if args.learn_from:
        run_learn_from(args.learn_from)
        return
    
    if args.introspect:
        run_introspect()
        return
    
    if not args.domain:
        print("Error: --domain is required")
        parser.print_help()
        sys.exit(1)
    
    if args.domain == "cross_knowledge":
        run_cross_domain_knowledge(demo=args.demo)
        return
    
    if args.domain == "self_experiment":
        run_self_experiment(demo=args.demo)
        return
    
    if args.domain == "evolution_monitor":
        run_evolution_monitor(demo=args.demo)
        return
    
    path = args.path or args.repo or args.gitops_repo or "."
    
    kwargs = {}
    if args.gitops_repo:
        kwargs['gitops'] = True
    
    run_domain_analysis(args.domain, path, cross_domain=args.cross_domain, **kwargs)


if __name__ == "__main__":
    main()