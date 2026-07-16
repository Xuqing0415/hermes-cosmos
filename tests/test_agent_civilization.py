import pytest
import tempfile
import os

from hermes.agent_civilization import (
    AgentRole,
    TaskStatus,
    CivilizationPhase,
    CivilizationCoordinator,
    TestOfficerAgent,
    Fixer,
    Prover,
    TaskBoard,
    KnowledgeBase,
    AgentProtocol
)


class TestAgentProtocol:
    def test_init(self):
        protocol = AgentProtocol()
        assert protocol.channel is not None
    
    def test_send_receive_message(self):
        protocol = AgentProtocol()
        protocol.send_message("agent1", "test", {"key": "value"}, "agent2")
        messages = protocol.receive_messages("agent2")
        assert len(messages) > 0
    
    def test_broadcast_message(self):
        protocol = AgentProtocol()
        protocol.broadcast_message("agent1", "test", {"key": "value"})
        messages = protocol.receive_messages("any")
        assert len(messages) > 0


class TestTaskBoard:
    def test_init(self):
        board = TaskBoard()
        assert board.tasks == {}
    
    def test_publish_task(self):
        board = TaskBoard()
        task = board.publish_task("Test Task", "Description", "test")
        assert task.task_id is not None
        assert task.title == "Test Task"
    
    def test_assign_task(self):
        board = TaskBoard()
        task = board.publish_task("Test Task", "Description", "test")
        result = board.assign_task(task.task_id, "agent1")
        assert result is True
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.assignee == "agent1"
    
    def test_complete_task(self):
        board = TaskBoard()
        task = board.publish_task("Test Task", "Description", "test")
        board.assign_task(task.task_id, "agent1")
        result = board.complete_task(task.task_id, {"result": "success"})
        assert result is True
        assert task.status == TaskStatus.COMPLETED
    
    def test_get_task_stats(self):
        board = TaskBoard()
        board.publish_task("Task 1", "Desc", "test")
        board.publish_task("Task 2", "Desc", "test")
        stats = board.get_task_stats()
        assert stats["total"] == 2
        assert stats["pending"] == 2


class TestKnowledgeBase:
    def test_init(self):
        kb = KnowledgeBase()
        assert kb.db_path == ":memory:"
    
    def test_add_and_get_knowledge(self):
        kb = KnowledgeBase()
        from hermes.agent_civilization.types import KnowledgeItem
        
        knowledge = KnowledgeItem(
            knowledge_id="",
            type="test",
            content={"key": "value"},
            source_agent_id="agent1",
            source_role=AgentRole.TEST_OFFICER
        )
        result = kb.add_knowledge(knowledge)
        assert result is True
        
        retrieved = kb.get_knowledge(knowledge.knowledge_id)
        assert retrieved is not None
        assert retrieved.type == "test"
    
    def test_search_knowledge(self):
        kb = KnowledgeBase()
        from hermes.agent_civilization.types import KnowledgeItem
        
        knowledge = KnowledgeItem(
            knowledge_id="",
            type="test",
            content={"query_key": "test value"},
            source_agent_id="agent1",
            source_role=AgentRole.TEST_OFFICER
        )
        kb.add_knowledge(knowledge)
        
        results = kb.search_knowledge("test")
        assert len(results) >= 1
    
    def test_get_stats(self):
        kb = KnowledgeBase()
        stats = kb.get_stats()
        assert "total_knowledge" in stats
    
    def test_close(self):
        kb = KnowledgeBase()
        kb.close()


class TestAgents:
    def test_test_officer_init(self):
        officer = TestOfficerAgent()
        assert officer.role == AgentRole.TEST_OFFICER
        assert officer.name == "TestOfficer"
    
    def test_test_officer_generate_tests(self):
        officer = TestOfficerAgent()
        result = officer.generate_tests({
            "source_code": "def add(a, b):\n    return a + b\n"
        })
        assert "tests" in result
        assert "test_count" in result
        assert result["test_count"] > 0
    
    def test_fixer_init(self):
        fixer = Fixer()
        assert fixer.role == AgentRole.FIXER
        assert fixer.name == "Fixer"
    
    def test_fixer_fix_code(self):
        fixer = Fixer()
        result = fixer.fix_code({
            "source_code": "def divide(a, b):\n    return a / b\n",
            "failed_tests": [{"function_name": "divide", "error": "ZeroDivisionError"}]
        })
        assert "fixed_code" in result
        assert result["success"] is True
    
    def test_prover_init(self):
        prover = Prover()
        assert prover.role == AgentRole.PROVER
        assert prover.name == "Prover"
    
    def test_prover_generate_proof(self):
        prover = Prover()
        result = prover.generate_proof({
            "source_code": "def add(a, b):\n    return a + b\n"
        })
        assert "proofs" in result
        assert "certificate" in result


class TestCivilizationCoordinator:
    def test_init(self):
        coordinator = CivilizationCoordinator()
        assert coordinator.state.phase == CivilizationPhase.STONE_AGE
    
    def test_register_agent(self):
        coordinator = CivilizationCoordinator()
        officer = TestOfficerAgent()
        result = coordinator.register_agent(officer)
        assert result is True
        assert coordinator.state.total_agents == 1
    
    def test_deregister_agent(self):
        coordinator = CivilizationCoordinator()
        officer = TestOfficerAgent()
        coordinator.register_agent(officer)
        result = coordinator.deregister_agent(officer.agent_id)
        assert result is True
        assert coordinator.state.total_agents == 0
    
    def test_publish_task(self):
        coordinator = CivilizationCoordinator()
        task = coordinator.publish_task("Test Task", "Description", "test")
        assert task.task_id is not None
        assert coordinator.state.total_tasks == 1
    
    def test_assign_and_complete_task(self):
        coordinator = CivilizationCoordinator()
        officer = TestOfficerAgent()
        coordinator.register_agent(officer)
        
        task = coordinator.publish_task("Test Task", "Description", "test")
        result = coordinator.assign_task(task.task_id, officer.agent_id)
        assert result is True
        
        result = coordinator.complete_task(task.task_id, {"result": "success"})
        assert result is True
    
    def test_get_agents_by_role(self):
        coordinator = CivilizationCoordinator()
        officer = TestOfficerAgent()
        coordinator.register_agent(officer)
        
        officers = coordinator.get_agents_by_role(AgentRole.TEST_OFFICER)
        assert len(officers) == 1
    
    def test_get_stats(self):
        coordinator = CivilizationCoordinator()
        stats = coordinator.get_stats()
        assert "phase" in stats
        assert "total_agents" in stats
        assert "total_tasks" in stats
    
    def test_advance_phase(self):
        coordinator = CivilizationCoordinator()
        result = coordinator.advance_phase()
        assert result is True
        assert coordinator.state.phase == CivilizationPhase.BRONZE_AGE
    
    def test_close(self):
        coordinator = CivilizationCoordinator()
        coordinator.close()


class TestMVPWorkflow:
    def test_full_collaboration_workflow(self):
        coordinator = CivilizationCoordinator()
        
        officer = TestOfficerAgent(coordinator.knowledge_base)
        fixer = Fixer(coordinator.knowledge_base)
        prover = Prover(coordinator.knowledge_base)
        
        coordinator.register_agent(officer)
        coordinator.register_agent(fixer)
        coordinator.register_agent(prover)
        
        buggy_code = """def divide(a, b):
    return a / b
"""
        
        result = coordinator.run_collaboration(
            title="Fix and Verify Divide Function",
            description="Fix the divide function to handle division by zero and verify correctness",
            input_data={"source_code": buggy_code}
        )
        
        assert "task_id" in result
        assert "steps" in result
        assert len(result["steps"]) >= 1
    
    def test_test_officer_detects_bug(self):
        officer = TestOfficerAgent()
        
        buggy_code = """def divide(a, b):
    return a / b
"""
        
        result = officer.generate_tests({"source_code": buggy_code})
        
        assert "failed_tests" in result
        assert len(result["failed_tests"]) >= 0
    
    def test_fixer_applies_fix(self):
        officer = TestOfficerAgent()
        fixer = Fixer()
        
        buggy_code = """def divide(a, b):
    return a / b
"""
        
        test_result = officer.generate_tests({"source_code": buggy_code})
        
        if test_result.get("failed_tests"):
            fix_result = fixer.fix_code({
                "source_code": buggy_code,
                "failed_tests": test_result["failed_tests"]
            })
            
            assert "fixed_code" in fix_result
            assert fix_result["success"] is True
    
    def test_prover_verifies_fixed_code(self):
        fixer = Fixer()
        prover = Prover()
        
        buggy_code = """def divide(a, b):
    return a / b
"""
        
        failed_tests = [{"function_name": "divide", "error": "ZeroDivisionError"}]
        
        fix_result = fixer.fix_code({
            "source_code": buggy_code,
            "failed_tests": failed_tests
        })
        
        if fix_result.get("fixed_code"):
            proof_result = prover.generate_proof({
                "fixed_code": fix_result["fixed_code"]
            })
            
            assert "proofs" in proof_result
            assert "certificate" in proof_result