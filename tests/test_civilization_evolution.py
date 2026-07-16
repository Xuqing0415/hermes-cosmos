import pytest
import tempfile
import os
import time

from hermes.agent_civilization import (
    Constitution,
    ConstitutionAmendmentSystem,
    ClauseType,
    AmendmentStatus,
    AgentRole,
    RoleInventor,
    RoleGenerator,
    RoleValidator,
    CivilizationMigration,
    CivilizationCoordinator,
    TestOfficerAgent,
    Fixer,
    Prover,
    LanguagePriest,
    MigrationApostle,
    CausalOracle,
    ArchJudge,
    ResourceOverseer,
    ValidationStatus
)


class TestConstitution:
    def test_init(self):
        constitution = Constitution()
        assert constitution.version.value == "v1"
        assert len(constitution.clauses) > 0
    
    def test_get_clause(self):
        constitution = Constitution()
        clause = constitution.get_clause(ClauseType.TASK_ASSIGNMENT)
        assert clause is not None
        assert clause.title == "任务竞标规则"
    
    def test_evaluate_task_assignment(self):
        constitution = Constitution()
        candidates = [
            {"agent_id": "agent1", "capability_match": 0.8, "success_rate": 0.9},
            {"agent_id": "agent2", "capability_match": 0.9, "success_rate": 0.7}
        ]
        result = constitution.evaluate_task_assignment("test", candidates)
        assert result is not None
    
    def test_check_phase_advancement(self):
        constitution = Constitution()
        agent_counts = {"test_officer": 2, "fixer": 1, "prover": 1}
        result = constitution.check_phase_advancement(agent_counts, 0)
        assert result is not None
    
    def test_to_dict_and_back(self):
        constitution = Constitution()
        data = constitution.to_dict()
        restored = Constitution.from_dict(data)
        assert restored.version == constitution.version
        assert len(restored.clauses) == len(constitution.clauses)
    
    def test_update_clause(self):
        constitution = Constitution()
        constitution.update_clause("task_assign_001", {"weight_capability_match": 0.7})
        clause = constitution.get_clause(ClauseType.TASK_ASSIGNMENT)
        assert clause.rules["weight_capability_match"] == 0.7


class TestConstitutionAmendment:
    def test_init(self):
        constitution = Constitution()
        amendment_system = ConstitutionAmendmentSystem(constitution)
        assert amendment_system.constitution == constitution
    
    def test_propose_amendment(self):
        constitution = Constitution()
        amendment_system = ConstitutionAmendmentSystem(constitution)
        
        amendment = amendment_system.propose_amendment(
            proposer_id="agent1",
            proposer_role=AgentRole.TEST_OFFICER,
            title="Test Amendment",
            description="Test description",
            clause_type=ClauseType.TASK_ASSIGNMENT,
            changes={"new_rule": "value"}
        )
        
        assert amendment is not None
        assert amendment.amendment_id is not None
        assert amendment.status == AmendmentStatus.PROPOSED
    
    def test_start_voting(self):
        constitution = Constitution()
        amendment_system = ConstitutionAmendmentSystem(constitution)
        
        amendment = amendment_system.propose_amendment(
            proposer_id="agent1",
            proposer_role=AgentRole.TEST_OFFICER,
            title="Test Amendment",
            description="Test description",
            clause_type=ClauseType.TASK_ASSIGNMENT,
            changes={"new_rule": "value"}
        )
        
        result = amendment_system.start_voting(amendment.amendment_id, ["agent1", "agent2"])
        assert result is True
        assert amendment.status == AmendmentStatus.VOTING
    
    def test_cast_vote(self):
        constitution = Constitution()
        amendment_system = ConstitutionAmendmentSystem(constitution)
        
        amendment = amendment_system.propose_amendment(
            proposer_id="agent1",
            proposer_role=AgentRole.TEST_OFFICER,
            title="Test Amendment",
            description="Test description",
            clause_type=ClauseType.TASK_ASSIGNMENT,
            changes={"new_rule": "value"}
        )
        
        amendment_system.start_voting(amendment.amendment_id, ["agent1", "agent2"])
        result = amendment_system.cast_vote(amendment.amendment_id, "agent1", "yes")
        assert result is True
    
    def test_amendment_passes(self):
        constitution = Constitution()
        amendment_system = ConstitutionAmendmentSystem(constitution)
        
        amendment = amendment_system.propose_amendment(
            proposer_id="agent1",
            proposer_role=AgentRole.TEST_OFFICER,
            title="Test Amendment",
            description="Test description",
            clause_type=ClauseType.TASK_ASSIGNMENT,
            changes={"new_rule": "value"}
        )
        
        amendment_system.start_voting(amendment.amendment_id, ["agent1", "agent2"])
        amendment_system.cast_vote(amendment.amendment_id, "agent1", "yes")
        amendment_system.cast_vote(amendment.amendment_id, "agent2", "yes")
        
        assert amendment.status == AmendmentStatus.PASSED
    
    def test_get_voting_status(self):
        constitution = Constitution()
        amendment_system = ConstitutionAmendmentSystem(constitution)
        
        amendment = amendment_system.propose_amendment(
            proposer_id="agent1",
            proposer_role=AgentRole.TEST_OFFICER,
            title="Test Amendment",
            description="Test description",
            clause_type=ClauseType.TASK_ASSIGNMENT,
            changes={"new_rule": "value"}
        )
        
        status = amendment_system.get_voting_status(amendment.amendment_id)
        assert "amendment_id" in status
        assert "status" in status
    
    def test_get_stats(self):
        constitution = Constitution()
        amendment_system = ConstitutionAmendmentSystem(constitution)
        
        stats = amendment_system.get_stats()
        assert "total_amendments" in stats
        assert "active_amendments" in stats


class TestRoleInventor:
    def test_init(self):
        inventor = RoleInventor()
        assert inventor is not None
    
    def test_analyze_failures(self):
        inventor = RoleInventor()
        
        failed_tasks = [
            {"task_id": "task1", "error_type": "performance", "error_message": "slow execution"},
            {"task_id": "task2", "error_type": "performance", "error_message": "slow execution"},
            {"task_id": "task3", "error_type": "performance", "error_message": "slow execution"}
        ]
        
        patterns = inventor.analyze_failures(failed_tasks)
        assert len(patterns) >= 1
    
    def test_generate_role_definition(self):
        inventor = RoleInventor()
        
        from hermes.agent_civilization.role_inventor import FailurePattern
        
        pattern = FailurePattern(
            pattern_id="",
            task_type="performance_test",
            error_type="performance",
            error_message="Execution time exceeded threshold, performance issue detected",
            frequency=5,
            affected_tasks=["task1", "task2", "task3"]
        )
        
        role_def = inventor.generate_role_definition(pattern)
        assert role_def is not None
        assert role_def.name == "PerformanceWatcher"
        assert "performance_analysis" in role_def.capabilities
    
    def test_detect_code_complexity(self):
        inventor = RoleInventor()
        
        from hermes.agent_civilization.role_inventor import FailurePattern
        
        pattern = FailurePattern(
            pattern_id="",
            task_type="analysis",
            error_type="complexity",
            error_message="Cyclomatic complexity too high, god function detected",
            frequency=4,
            affected_tasks=["task1", "task2"]
        )
        
        role_def = inventor.generate_role_definition(pattern)
        assert role_def is not None
        assert role_def.name == "ComplexityAnalyzer"
    
    def test_detect_security_issues(self):
        inventor = RoleInventor()
        
        from hermes.agent_civilization.role_inventor import FailurePattern
        
        pattern = FailurePattern(
            pattern_id="",
            task_type="security",
            error_type="security",
            error_message="Security vulnerability detected, injection risk",
            frequency=3,
            affected_tasks=["task1", "task2", "task3"]
        )
        
        role_def = inventor.generate_role_definition(pattern)
        assert role_def is not None
        assert role_def.name == "SecurityGuard"
    
    def test_get_stats(self):
        inventor = RoleInventor()
        
        stats = inventor.get_stats()
        assert "total_failure_patterns" in stats
        assert "total_role_definitions" in stats


class TestRoleGenerator:
    def test_generate_agent_class(self):
        from hermes.agent_civilization.role_inventor import RoleDefinition
        
        role_def = RoleDefinition(
            role_id="",
            name="TestRole",
            category=None,
            description="Test role",
            responsibilities=["do_something"],
            capabilities=["capability_a", "capability_b"],
            input_spec={"input_key": "description"},
            output_spec={"output_key": "description"},
            required_skills=["skill_a"]
        )
        
        agent_class = RoleGenerator.generate_agent_class(role_def)
        assert agent_class is not None
        assert agent_class.__name__ == "TestRole"
    
    def test_create_agent_instance(self):
        from hermes.agent_civilization.role_inventor import RoleDefinition
        
        role_def = RoleDefinition(
            role_id="",
            name="TestInstance",
            category=None,
            description="Test instance",
            responsibilities=["test"],
            capabilities=["test_capability"],
            input_spec={"test": "input"},
            output_spec={"result": "output"},
            required_skills=["testing"]
        )
        
        agent = RoleGenerator.create_agent_instance(role_def)
        assert agent is not None
        assert agent.name == "TestInstance"
        assert "test_capability" in agent.capabilities
    
    def test_execute_task(self):
        from hermes.agent_civilization.role_inventor import RoleDefinition
        
        role_def = RoleDefinition(
            role_id="",
            name="ExecutorRole",
            category=None,
            description="Executor",
            responsibilities=["execute"],
            capabilities=["execute_task"],
            input_spec={"data": "input data"},
            output_spec={"result": "output"},
            required_skills=["execution"]
        )
        
        agent = RoleGenerator.create_agent_instance(role_def)
        result = agent.execute_task("execute_task", {"data": "test"})
        assert result is not None
        assert "capability" in result
    
    def test_generate_performance_watcher(self):
        agent_class = RoleGenerator.generate_performance_watcher()
        assert agent_class is not None
        assert agent_class.__name__ == "PerformanceWatcher"
        
        agent = agent_class()
        assert "performance_analysis" in agent.capabilities


class TestRoleValidator:
    def test_init(self):
        validator = RoleValidator()
        assert validator is not None
    
    def test_validate_role(self):
        from hermes.agent_civilization.role_inventor import RoleDefinition
        
        role_def = RoleDefinition(
            role_id="",
            name="ValidatorTest",
            category=None,
            description="Test",
            responsibilities=["test"],
            capabilities=["test_cap"],
            input_spec={"input": "test"},
            output_spec={"output": "test"},
            required_skills=["test"]
        )
        
        agent = RoleGenerator.create_agent_instance(role_def)
        validator = RoleValidator()
        
        result = validator.validate_role(agent, role_def)
        assert result is not None
        assert result.validation_id is not None
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED]
    
    def test_validate_performance_watcher(self):
        agent_class = RoleGenerator.generate_performance_watcher()
        agent = agent_class()
        
        validator = RoleValidator()
        result = validator.validate_performance_watcher(agent)
        
        assert result is not None
        assert result.tests_run >= 3
    
    def test_get_validation_result(self):
        from hermes.agent_civilization.role_inventor import RoleDefinition
        
        role_def = RoleDefinition(
            role_id="",
            name="GetResultTest",
            category=None,
            description="Test",
            responsibilities=["test"],
            capabilities=["test_cap"],
            input_spec={"input": "test"},
            output_spec={"output": "test"},
            required_skills=["test"]
        )
        
        agent = RoleGenerator.create_agent_instance(role_def)
        validator = RoleValidator()
        
        result = validator.validate_role(agent, role_def)
        retrieved = validator.get_validation_result(result.validation_id)
        
        assert retrieved is not None
        assert retrieved.validation_id == result.validation_id
    
    def test_get_stats(self):
        validator = RoleValidator()
        
        stats = validator.get_stats()
        assert "total_validations" in stats
        assert "passed" in stats
        assert "failed" in stats


class TestCivilizationMigration:
    def test_init(self):
        coordinator = CivilizationCoordinator()
        migration = CivilizationMigration(coordinator)
        assert migration.coordinator == coordinator
    
    def test_export_state(self):
        coordinator = CivilizationCoordinator()
        migration = CivilizationMigration(coordinator)
        
        state = migration.export_state()
        assert state is not None
        assert "version" in state
        assert "civilization" in state
    
    def test_export_to_file(self):
        coordinator = CivilizationCoordinator()
        migration = CivilizationMigration(coordinator)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            result = migration.export_to_file(filepath)
            assert result is True
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_import_from_file(self):
        coordinator = CivilizationCoordinator()
        migration = CivilizationMigration(coordinator)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            migration.export_to_file(filepath)
            
            new_coordinator = CivilizationCoordinator()
            new_migration = CivilizationMigration(new_coordinator)
            
            result = new_migration.import_from_file(filepath)
            assert result is True
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_get_state_summary(self):
        coordinator = CivilizationCoordinator()
        migration = CivilizationMigration(coordinator)
        
        summary = migration.get_state_summary()
        assert "phase" in summary
        assert "total_agents" in summary
        assert "constitution_version" in summary


class TestCivilizationCoordinatorIntegration:
    def test_init_with_constitution(self):
        constitution = Constitution()
        coordinator = CivilizationCoordinator(constitution)
        
        assert coordinator.constitution == constitution
    
    def test_select_assignee_with_constitution(self):
        coordinator = CivilizationCoordinator()
        
        officer = TestOfficerAgent()
        coordinator.register_agent(officer)
        
        assignee = coordinator.select_assignee("test_generation")
        assert assignee is not None
    
    def test_phase_advancement_with_constitution(self):
        coordinator = CivilizationCoordinator()
        
        officer = TestOfficerAgent()
        fixer = Fixer()
        coordinator.register_agent(officer)
        coordinator.register_agent(fixer)
        
        assert coordinator.state.phase.value == "bronze_age"
    
    def test_get_stats_includes_constitution(self):
        coordinator = CivilizationCoordinator()
        
        stats = coordinator.get_stats()
        assert "constitution_version" in stats
        assert "amendment_count" in stats


class TestRoleInventionPipeline:
    def test_full_pipeline(self):
        RoleGenerator.clear_generated_classes()
        
        inventor = RoleInventor()
        
        failed_tasks = [
            {"task_id": "t1", "error_type": "performance", "error_message": "slow execution time", "task_type": "test"},
            {"task_id": "t2", "error_type": "performance", "error_message": "slow execution time", "task_type": "test"},
            {"task_id": "t3", "error_type": "performance", "error_message": "slow execution time", "task_type": "test"},
            {"task_id": "t4", "error_type": "performance", "error_message": "slow execution time", "task_type": "test"},
            {"task_id": "t5", "error_type": "performance", "error_message": "slow execution time", "task_type": "test"}
        ]
        
        patterns = inventor.analyze_failures(failed_tasks)
        
        assert len(patterns) >= 1
        
        pattern = patterns[0]
        role_def = inventor.generate_role_definition(pattern)
        
        assert role_def is not None
        assert role_def.name == "PerformanceWatcher"
        
        agent_class = RoleGenerator.generate_agent_class(role_def)
        agent = agent_class()
        
        assert agent is not None
        
        validator = RoleValidator()
        result = validator.validate_role(agent, role_def)
        
        assert result is not None
        assert result.status == ValidationStatus.PASSED


class TestLanguagePriest:
    def test_init(self):
        priest = LanguagePriest()
        assert priest.role == AgentRole.LANGUAGE_PRIEST
        assert "language_design" in priest.capabilities
    
    def test_design_grammar(self):
        priest = LanguagePriest()
        result = priest.execute_task("language_design", {"requirements": "performance security"})
        assert "new_rules" in result
        assert len(result["new_rules"]) >= 1
    
    def test_evolve_grammar(self):
        priest = LanguagePriest()
        feedbacks = [
            {"rule_id": "rule_001", "test_pass_rate": 0.3, "usage_count": 10, 
             "error_patterns": ["syntax error"], "suggestion": "TEST <function> WITH <input> EXPECT <output> [TIMEOUT]"}
        ]
        result = priest.execute_task("grammar_evolution", {"feedbacks": feedbacks})
        assert "version" in result
        assert result["version"] > "1.0"
    
    def test_syntax_validation(self):
        priest = LanguagePriest()
        result = priest.execute_task("syntax_validation", {"expression": "TEST my_func WITH x=1 EXPECT 2"})
        assert "valid" in result


class TestMigrationApostle:
    def test_init(self):
        apostle = MigrationApostle()
        assert apostle.role == AgentRole.MIGRATION_APOSTLE
        assert "cross_language" in apostle.capabilities
    
    def test_migrate_code_python_to_rust(self):
        apostle = MigrationApostle()
        result = apostle.execute_task("cross_language", {
            "source_code": "def add(a, b):\n    return a + b\nassert add(1, 2) == 3",
            "source_lang": "python",
            "target_lang": "rust"
        })
        assert "migrated_code" in result
        assert result["status"] == "success"
    
    def test_migrate_code_python_to_go(self):
        apostle = MigrationApostle()
        result = apostle.execute_task("cross_language", {
            "source_code": "def add(a, b):\n    return a + b",
            "source_lang": "python",
            "target_lang": "go"
        })
        assert "migrated_code" in result
    
    def test_knowledge_transfer(self):
        apostle = MigrationApostle()
        result = apostle.execute_task("knowledge_transfer", {
            "knowledge_items": [{"content": "assert x == 1"}],
            "source_lang": "python",
            "target_lang": "rust"
        })
        assert "transferred" in result


class TestCausalOracle:
    def test_init(self):
        oracle = CausalOracle()
        assert oracle.role == AgentRole.CAUSAL_ORACLE
        assert "causal_analysis" in oracle.capabilities
    
    def test_analyze_zero_division_error(self):
        oracle = CausalOracle()
        result = oracle.execute_task("causal_analysis", {
            "error_type": "ZeroDivisionError",
            "error_message": "division by zero",
            "context": {"function": "divide", "input": [1, 0]}
        })
        assert "root_causes" in result
        assert "recommendations" in result
    
    def test_detect_root_cause(self):
        oracle = CausalOracle()
        result = oracle.execute_task("root_cause_detection", {
            "task_result": {"error": "ZeroDivisionError: division by zero"}
        })
        assert "root_causes" in result
    
    def test_generate_explanation(self):
        oracle = CausalOracle()
        failed_tests = [{"test_name": "test_div", "error_type": "ZeroDivisionError"}]
        result = oracle.execute_task("explanation_generation", {"failed_tests": failed_tests})
        assert "explanations" in result


class TestArchJudge:
    def test_init(self):
        judge = ArchJudge()
        assert judge.role == AgentRole.ARCHITECTURE_JUDGE
        assert "architecture_analysis" in judge.capabilities
    
    def test_analyze_architecture(self):
        judge = ArchJudge()
        test_code = """def simple_func(a, b):
    return a + b
"""
        result = judge.execute_task("architecture_analysis", {"source_code": test_code})
        assert "loc" in result
        assert "cyclomatic_complexity" in result
    
    def test_detect_smells(self):
        judge = ArchJudge()
        long_func = """def very_long_function(a, b, c, d, e):
    result = a + b
    result = result * c
    result = result - d
    result = result / e
    if a > b:
        result = result + 1
    if c < d:
        result = result - 1
    if e > 0:
        result = result * 2
    return result
"""
        result = judge.execute_task("smell_detection", {"source_code": long_func})
        assert "smells" in result
    
    def test_assess_health(self):
        judge = ArchJudge()
        simple_code = """def simple_func(a):
    return a * 2
"""
        result = judge.execute_task("health_assessment", {"source_code": simple_code})
        assert "health_score" in result
        assert "status" in result


class TestResourceOverseer:
    def test_init(self):
        overseer = ResourceOverseer()
        assert overseer.role == AgentRole.RESOURCE_OVERSEER
        assert "resource_management" in overseer.capabilities
    
    def test_manage_resources(self):
        overseer = ResourceOverseer()
        agents = [{"agent_id": "a1", "task_count": 2, "success_rate": 0.8}]
        tasks = [{"task_id": "t1"}, {"task_id": "t2"}]
        result = overseer.execute_task("resource_management", {"agents": agents, "tasks": tasks})
        assert "total_agents" in result
        assert "pending_tasks" in result
    
    def test_load_balancing(self):
        overseer = ResourceOverseer()
        agents = [
            {"agent_id": "a1", "task_count": 0, "success_rate": 0.9, "capabilities": ["test_generation"]},
            {"agent_id": "a2", "task_count": 3, "success_rate": 0.7, "capabilities": ["test_generation"]}
        ]
        result = overseer.execute_task("load_balancing", {"agents": agents, "task_type": "test_generation"})
        assert "recommended_agent" in result
        assert result["recommended_agent"] == "a1"
    
    def test_strategy_optimization(self):
        overseer = ResourceOverseer()
        task_results = [{"success": True} for _ in range(10)]
        result = overseer.execute_task("strategy_optimization", {"task_results": task_results})
        assert "strategy" in result
        assert "success_rate" in result


class TestFullCivilizationIntegration:
    def test_register_all_roles(self):
        coordinator = CivilizationCoordinator()
        
        officers = [TestOfficerAgent() for _ in range(2)]
        fixers = [Fixer() for _ in range(2)]
        provers = [Prover() for _ in range(2)]
        priest = LanguagePriest()
        apostle = MigrationApostle()
        oracle = CausalOracle()
        judge = ArchJudge()
        overseer = ResourceOverseer()
        
        for agent in officers + fixers + provers + [priest, apostle, oracle, judge, overseer]:
            coordinator.register_agent(agent)
        
        assert coordinator.state.total_agents == 11
        assert coordinator.state.phase.value == "iron_age"
    
    def test_multi_role_collaboration(self):
        coordinator = CivilizationCoordinator()
        
        officer = TestOfficerAgent()
        fixer = Fixer()
        oracle = CausalOracle()
        judge = ArchJudge()
        
        coordinator.register_agent(officer)
        coordinator.register_agent(fixer)
        coordinator.register_agent(oracle)
        coordinator.register_agent(judge)
        
        stats = coordinator.get_stats()
        assert stats["phase"] == "bronze_age"
        assert stats["total_agents"] == 4
    
    def test_civilization_sandbox_simulation(self):
        coordinator = CivilizationCoordinator()
        
        officer = TestOfficerAgent()
        fixer = Fixer()
        prover = Prover()
        oracle = CausalOracle()
        judge = ArchJudge()
        
        coordinator.register_agent(officer)
        coordinator.register_agent(fixer)
        coordinator.register_agent(prover)
        coordinator.register_agent(oracle)
        coordinator.register_agent(judge)
        
        completed_tasks = 0
        for i in range(50):
            task = coordinator.publish_task(
                title=f"Task {i}",
                description=f"Test task {i}",
                task_type="test_generation",
                input_data={"source_code": "def func(a): return a"},
                priority=1
            )
            
            assignee = coordinator.select_assignee("test_generation")
            if assignee:
                coordinator.assign_task(task.task_id, assignee)
                coordinator.complete_task(task.task_id, {"success": True})
                completed_tasks += 1
        
        assert completed_tasks > 0
        stats = coordinator.get_stats()
        assert stats["task_stats"]["completed"] == completed_tasks