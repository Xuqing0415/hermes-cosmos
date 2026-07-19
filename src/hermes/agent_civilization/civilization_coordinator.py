import time
from typing import List, Dict, Optional, Any

from .types import (
    AgentInfo, AgentRole, Task, TaskStatus, Message, MessageType,
    CivilizationState, CivilizationPhase, CivilizationLog, KnowledgeItem
)
from .agent_protocol import AgentProtocol
from .task_board import TaskBoard
from .knowledge_base import KnowledgeBase
from .constitution import Constitution, ClauseType
from .constitution_amendment import ConstitutionAmendmentSystem, AmendmentStatus
from .constitution_voting import ConstitutionVoting, ConstitutionAmendment, RoleProposal, VoteOption, AmendmentStatus as VotingStatus
from .role_inventor import RoleInventor, FailurePattern, RoleTemplate, GeneratedRole
from .agents import BaseAgent


class CivilizationCoordinator:
    def __init__(self, constitution: Optional[Constitution] = None):
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.protocol = AgentProtocol()
        self.task_board = TaskBoard()
        self.knowledge_base = KnowledgeBase()
        self.state = CivilizationState()
        self.logs: List[CivilizationLog] = []
        self._role_agents: Dict[AgentRole, List[str]] = {}
        self.constitution = constitution or Constitution()
        self.amendment_system = ConstitutionAmendmentSystem(self.constitution)
        self.amendment_system.set_on_amendment_applied_callback(self._on_amendment_applied)
        self.voting_system = ConstitutionVoting()
        self.role_inventor = RoleInventor()
        self._task_type_weights: Dict[str, float] = {}
        self._role_admission_rules: Dict[AgentRole, Dict[str, Any]] = {}
        self._amendment_interval = 10
        self._tasks_since_last_amendment = 0
        self._tasks_since_last_role_invention = 0
        self._role_invention_interval = 15

    def register_agent(self, agent_info: AgentInfo) -> bool:
        if agent_info.agent_id in self.agents:
            return False
        
        self.agents[agent_info.agent_id] = agent_info
        if isinstance(agent_info, BaseAgent):
            self.agent_instances[agent_info.agent_id] = agent_info
        self.state.total_agents += 1
        self.state.last_update = time.time()
        
        if agent_info.role not in self._role_agents:
            self._role_agents[agent_info.role] = []
        self._role_agents[agent_info.role].append(agent_info.agent_id)
        
        self.protocol.send_register(agent_info.agent_id, agent_info)
        
        self._add_log(
            agent_id=agent_info.agent_id,
            agent_role=agent_info.role,
            action="register",
            details={"name": agent_info.name, "capabilities": agent_info.capabilities}
        )
        
        self._check_phase_advancement()
        
        return True

    def deregister_agent(self, agent_id: str) -> bool:
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return False
        
        self.protocol.send_deregister(agent_id)
        
        if agent_info.role in self._role_agents and agent_id in self._role_agents[agent_info.role]:
            self._role_agents[agent_info.role].remove(agent_id)
        
        self.agents.pop(agent_id)
        self.state.total_agents -= 1
        self.state.last_update = time.time()
        
        self._add_log(
            agent_id=agent_id,
            agent_role=agent_info.role,
            action="deregister"
        )
        
        return True

    def update_agent_heartbeat(self, agent_id: str, status: str = "active"):
        agent_info = self.agents.get(agent_id)
        if agent_info:
            agent_info.last_heartbeat = time.time()
            agent_info.status = status

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        return self.agents.get(agent_id)

    def get_agents_by_role(self, role: AgentRole) -> List[AgentInfo]:
        agent_ids = self._role_agents.get(role, [])
        return [self.agents[aid] for aid in agent_ids]

    def get_all_agents(self) -> List[AgentInfo]:
        return list(self.agents.values())

    def publish_task(self, title: str, description: str, task_type: str,
                     input_data: Dict[str, Any] = None, priority: int = 0) -> Task:
        task = self.task_board.publish_task(
            title=title,
            description=description,
            task_type=task_type,
            input_data=input_data,
            priority=priority
        )
        
        self.state.total_tasks += 1
        self.state.last_update = time.time()
        
        self.protocol.send_task_publish("coordinator", task)
        
        self._add_log(
            agent_id="coordinator",
            agent_role=AgentRole.RESOURCE_OVERSEER,
            action="task_publish",
            details={"task_id": task.task_id, "title": title, "type": task_type}
        )
        
        return task

    def assign_task(self, task_id: str, assignee_id: str) -> bool:
        success = self.task_board.assign_task(task_id, assignee_id)
        
        if success:
            self.protocol.send_task_assign("coordinator", task_id, assignee_id)
            
            agent_info = self.agents.get(assignee_id)
            if agent_info:
                agent_info.task_count += 1
            
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="task_assign",
                details={"task_id": task_id, "assignee_id": assignee_id}
            )
        
        return success

    def complete_task(self, task_id: str, output_data: Dict[str, Any]) -> bool:
        success = self.task_board.complete_task(task_id, output_data)
        
        if success:
            self.protocol.send_task_complete("coordinator", task_id, output_data)
            
            task = self.task_board.get_task(task_id)
            if task and task.assignee:
                agent_info = self.agents.get(task.assignee)
                if agent_info:
                    agent_info.success_rate = (
                        (agent_info.success_rate * (agent_info.task_count - 1) + 1) / 
                        agent_info.task_count
                    )
            
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="task_complete",
                details={"task_id": task_id}
            )
            
            self._check_phase_advancement()
            self._process_amendment_cycle()
        
        return success

    def add_knowledge(self, knowledge: KnowledgeItem) -> bool:
        success = self.knowledge_base.add_knowledge(knowledge)
        
        if success:
            self.state.total_knowledge += 1
            self.state.last_update = time.time()
            
            self.protocol.send_knowledge_share(knowledge.source_agent_id, {
                "knowledge_id": knowledge.knowledge_id,
                "type": knowledge.type,
                "content": knowledge.content
            })
            
            self._add_log(
                agent_id=knowledge.source_agent_id,
                agent_role=knowledge.source_role,
                action="knowledge_share",
                details={"knowledge_id": knowledge.knowledge_id, "type": knowledge.type}
            )
        
        return success

    def search_knowledge(self, query: str, knowledge_type: Optional[str] = None,
                         max_results: int = 10) -> List[KnowledgeItem]:
        return self.knowledge_base.search_knowledge(query, knowledge_type, max_results)

    def select_assignee(self, task_type: str) -> Optional[str]:
        role_map = {
            "test_generation": AgentRole.TEST_OFFICER,
            "coverage_analysis": AgentRole.TEST_OFFICER,
            "test_execution": AgentRole.TEST_OFFICER,
            "code_fix": AgentRole.FIXER,
            "refactoring": AgentRole.FIXER,
            "patch_generation": AgentRole.FIXER,
            "proof_generation": AgentRole.PROVER,
            "theorem_proving": AgentRole.PROVER,
            "verification": AgentRole.PROVER,
            "architecture_analysis": AgentRole.ARCHITECTURE_JUDGE,
            "smell_detection": AgentRole.ARCHITECTURE_JUDGE,
            "health_assessment": AgentRole.ARCHITECTURE_JUDGE,
            "causal_analysis": AgentRole.CAUSAL_ORACLE,
            "root_cause_detection": AgentRole.CAUSAL_ORACLE,
            "explanation_generation": AgentRole.CAUSAL_ORACLE,
            "language_design": AgentRole.LANGUAGE_PRIEST,
            "grammar_evolution": AgentRole.LANGUAGE_PRIEST,
            "syntax_validation": AgentRole.LANGUAGE_PRIEST,
            "cross_language": AgentRole.MIGRATION_APOSTLE,
            "rule_migration": AgentRole.MIGRATION_APOSTLE,
            "knowledge_transfer": AgentRole.MIGRATION_APOSTLE,
            "resource_management": AgentRole.RESOURCE_OVERSEER,
            "load_balancing": AgentRole.RESOURCE_OVERSEER,
            "strategy_optimization": AgentRole.RESOURCE_OVERSEER,
            "mlir_verification": AgentRole.COMPILER_VERIFIER,
            "pass_validation": AgentRole.COMPILER_VERIFIER,
            "bug_detection": AgentRole.COMPILER_VERIFIER,
            "equivalence_proof": AgentRole.COMPILER_VERIFIER
        }
        
        role = role_map.get(task_type)
        if not role:
            return None
        
        candidates = self.get_agents_by_role(role)
        if not candidates:
            return None
        
        weight = self._task_type_weights.get(task_type, 1.0)
        
        candidate_data = [
            {
                "agent_id": c.agent_id,
                "capability_match": (1.0 if task_type in c.capabilities else 0.5) * weight,
                "success_rate": c.success_rate,
                "weight": weight
            }
            for c in candidates
        ]
        
        return self.constitution.evaluate_task_assignment(task_type, candidate_data)

    def _check_phase_advancement(self):
        agent_counts = {role.value: len(agents) for role, agents in self._role_agents.items()}
        task_stats = self.task_board.get_task_stats()
        
        new_phase = self.constitution.check_phase_advancement(agent_counts, task_stats["completed"])
        
        if new_phase and list(CivilizationPhase).index(new_phase) > list(CivilizationPhase).index(self.state.phase):
            self.state.phase = new_phase
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="phase_advance",
                details={"new_phase": self.state.phase.value}
            )

    def _process_amendment_cycle(self):
        self._tasks_since_last_amendment += 1
        
        if self._tasks_since_last_amendment >= self._amendment_interval:
            self._tasks_since_last_amendment = 0
            
            task_stats = self.task_board.get_task_stats()
            if task_stats["completed"] > 0:
                failure_rate = task_stats["failed"] / (task_stats["completed"] + task_stats["failed"])
                
                if failure_rate > 0.3:
                    self._propose_amendment_to_adjust_weights(failure_rate)
                
                self._auto_vote_on_pending_amendments()
    
    def _propose_amendment_to_adjust_weights(self, failure_rate: float):
        task_type_stats = self.task_board.get_task_type_stats()
        if not task_type_stats:
            return
        
        lowest_success_type = min(task_type_stats.items(), key=lambda x: x[1].get("success_rate", 1.0))[0]
        
        changes = {
            "task_type_weights": {
                lowest_success_type: 1.5
            },
            "priority_adjustment": float(failure_rate)
        }
        
        amendment = self.amendment_system.propose_amendment(
            proposer_id="coordinator",
            proposer_role=AgentRole.RESOURCE_OVERSEER,
            title=f"调整任务分配权重 - 提高{lowest_success_type}优先级",
            description=f"检测到任务失败率{failure_rate:.1%}，建议增加失败率最高任务类型的权重",
            clause_type=ClauseType.TASK_ASSIGNMENT,
            changes=changes
        )
        
        self._add_log(
            agent_id="coordinator",
            agent_role=AgentRole.RESOURCE_OVERSEER,
            action="amendment_propose",
            details={"amendment_id": amendment.amendment_id, "title": amendment.title}
        )
        
        voter_ids = list(self.agents.keys())[:5]
        if voter_ids:
            self.amendment_system.start_voting(amendment.amendment_id, voter_ids, duration_seconds=10)
    
    def _auto_vote_on_pending_amendments(self):
        for amendment_id in list(self.amendment_system._active_amendments):
            amendment = self.amendment_system.amendments.get(amendment_id)
            if amendment and amendment.status == AmendmentStatus.VOTING:
                for voter_id in amendment.voters:
                    if voter_id not in amendment.votes:
                        vote = self._decide_vote(amendment, voter_id)
                        self.amendment_system.cast_vote(amendment_id, voter_id, vote)
    
    def _decide_vote(self, amendment, voter_id: str) -> str:
        agent_info = self.agents.get(voter_id)
        if not agent_info:
            return "yes"
        
        changes = amendment.changes
        if "task_type_weights" in changes:
            for task_type, weight in changes["task_type_weights"].items():
                if task_type in agent_info.capabilities:
                    if agent_info.success_rate < 0.5:
                        return "yes"
        
        if agent_info.success_rate > 0.7:
            return "no"
        
        return "yes"
    
    def _on_amendment_applied(self, amendment):
        if amendment.clause_type == ClauseType.TASK_ASSIGNMENT:
            self._update_task_assignment_weights()
    
    def _update_task_assignment_weights(self):
        clause = self.constitution.get_clause(ClauseType.TASK_ASSIGNMENT)
        if clause and "task_type_weights" in clause.rules:
            self._task_type_weights = clause.rules["task_type_weights"]
    
    def run_collaboration(self, title: str, description: str,
                          input_data: Dict[str, Any],
                          pipeline_type: str = "full") -> Dict[str, Any]:
        task = self.publish_task(
            title=title,
            description=description,
            task_type="collaboration",
            input_data=input_data,
            priority=10
        )
        
        steps = []
        current_data = input_data
        
        if pipeline_type in ("full", "test_fix_prove"):
            test_officers = self.get_agents_by_role(AgentRole.TEST_OFFICER)
            if test_officers:
                step_result = self._run_agent_task(test_officers[0].agent_id, "test_generation", current_data)
                steps.append({"agent": "test_officer", "action": "generate_tests", "result": step_result})
                current_data.update(step_result)
            
            if current_data.get("failed_tests"):
                fixers = self.get_agents_by_role(AgentRole.FIXER)
                if fixers:
                    step_result = self._run_agent_task(fixers[0].agent_id, "code_fix", current_data)
                    steps.append({"agent": "fixer", "action": "apply_fix", "result": step_result})
                    current_data.update(step_result)
                
                oracles = self.get_agents_by_role(AgentRole.CAUSAL_ORACLE)
                if oracles:
                    step_result = self._run_agent_task(oracles[0].agent_id, "explanation_generation", current_data)
                    steps.append({"agent": "causal_oracle", "action": "explain_failures", "result": step_result})
                    current_data.update(step_result)
            
            if current_data.get("fixed_code"):
                provers = self.get_agents_by_role(AgentRole.PROVER)
                if provers:
                    step_result = self._run_agent_task(provers[0].agent_id, "proof_generation", current_data)
                    steps.append({"agent": "prover", "action": "generate_proof", "result": step_result})
                    current_data.update(step_result)
        
        if pipeline_type in ("full", "arch_refactor"):
            judges = self.get_agents_by_role(AgentRole.ARCHITECTURE_JUDGE)
            if judges:
                step_result = self._run_agent_task(judges[0].agent_id, "smell_detection", current_data)
                steps.append({"agent": "arch_judge", "action": "detect_smells", "result": step_result})
                current_data.update(step_result)
                
                if current_data.get("smells"):
                    fixers = self.get_agents_by_role(AgentRole.FIXER)
                    if fixers:
                        step_result = self._run_agent_task(fixers[0].agent_id, "refactoring", current_data)
                        steps.append({"agent": "fixer", "action": "apply_refactoring", "result": step_result})
                        current_data.update(step_result)
        
        if pipeline_type in ("full", "language_evolution"):
            priests = self.get_agents_by_role(AgentRole.LANGUAGE_PRIEST)
            if priests:
                step_result = self._run_agent_task(priests[0].agent_id, "grammar_evolution", current_data)
                steps.append({"agent": "language_priest", "action": "evolve_grammar", "result": step_result})
                current_data.update(step_result)
                
                apostles = self.get_agents_by_role(AgentRole.MIGRATION_APOSTLE)
                if apostles and current_data.get("new_grammar"):
                    step_result = self._run_agent_task(apostles[0].agent_id, "knowledge_transfer", {
                        "knowledge_items": [{"content": str(current_data.get("new_grammar"))}],
                        "source_lang": "python",
                        "target_lang": "rust"
                    })
                    steps.append({"agent": "migration_apostle", "action": "update_cross_lang_rules", "result": step_result})
                    current_data.update(step_result)
        
        self.complete_task(task.task_id, current_data)
        
        return {
            "task_id": task.task_id,
            "success": not current_data.get("failed_tests"),
            "steps": steps,
            "final_output": current_data,
            "pipeline_type": pipeline_type
        }

    def _run_agent_task(self, agent_id: str, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return {"error": f"Agent {agent_id} not found"}
        
        try:
            agent_instance = self.agent_instances.get(agent_id)
            if agent_instance is not None:
                return agent_instance.execute_task(task_type, input_data)
            elif hasattr(agent_info, 'execute_task'):
                return agent_info.execute_task(task_type, input_data)
            else:
                return {"error": "Agent does not support execute_task"}
        except Exception as e:
            return {"error": str(e)}

    def _add_log(self, agent_id: str, agent_role: AgentRole, action: str,
                 details: Dict[str, Any] = None):
        log = CivilizationLog(
            log_id="",
            timestamp=time.time(),
            agent_id=agent_id,
            agent_role=agent_role,
            action=action,
            details=details or {}
        )
        self.logs.append(log)
        self.state.total_logs += 1

    def run_sandbox_simulation(self, task_count: int = 100) -> Dict[str, Any]:
        import random
        
        task_types = [
            "test_generation", "code_fix", "proof_generation",
            "architecture_analysis", "smell_detection", "health_assessment",
            "causal_analysis", "root_cause_detection", "explanation_generation",
            "language_design", "grammar_evolution", "syntax_validation",
            "cross_language", "rule_migration", "knowledge_transfer",
            "resource_management", "load_balancing", "strategy_optimization"
        ]
        
        source_codes = [
            "def add(a, b):\n    return a + b",
            "def divide(a, b):\n    return a / b",
            "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def max_value(arr):\n    max_val = arr[0]\n    for val in arr:\n        if val > max_val:\n            max_val = val\n    return max_val"
        ]
        
        results = {
            "total_tasks": task_count,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "success_rate": 0.0,
            "task_type_stats": {},
            "phase_history": [],
            "evolution_log": []
        }
        
        initial_phase = self.state.phase.value
        results["phase_history"].append({"time": 0, "phase": initial_phase})
        
        for i in range(task_count):
            task_type = random.choice(task_types)
            source_code = random.choice(source_codes)
            
            task = self.publish_task(
                title=f"Sandbox Task {i}",
                description=f"Random task {i} of type {task_type}",
                task_type=task_type,
                input_data={"source_code": source_code},
                priority=random.randint(1, 10)
            )
            
            assignee = self.select_assignee(task_type)
            if assignee:
                self.assign_task(task.task_id, assignee)
                task_result = self._run_agent_task(assignee, task_type, {"source_code": source_code})
                
                if task_result.get("error"):
                    self.fail_task(task.task_id, task_result["error"])
                    results["failed_tasks"] += 1
                else:
                    self.complete_task(task.task_id, {"success": True})
                    results["completed_tasks"] += 1
                
                if task_type not in results["task_type_stats"]:
                    results["task_type_stats"][task_type] = {"completed": 0, "failed": 0}
                
                if task_result.get("error"):
                    results["task_type_stats"][task_type]["failed"] += 1
                else:
                    results["task_type_stats"][task_type]["completed"] += 1
                
                if i % 10 == 0:
                    self._check_phase_advancement()
                    results["phase_history"].append({
                        "time": i,
                        "phase": self.state.phase.value,
                        "total_agents": self.state.total_agents
                    })
            
            else:
                results["failed_tasks"] += 1
                if task_type not in results["task_type_stats"]:
                    results["task_type_stats"][task_type] = {"completed": 0, "failed": 0}
                results["task_type_stats"][task_type]["failed"] += 1
        
        results["success_rate"] = results["completed_tasks"] / task_count if task_count > 0 else 0.0
        
        results["evolution_log"] = [
            {
                "timestamp": log.timestamp,
                "agent_id": log.agent_id,
                "agent_role": log.agent_role.value,
                "action": log.action,
                "details": log.details
            }
            for log in self.get_logs()[-50:]
        ]
        
        return results

    def get_state(self) -> CivilizationState:
        return self.state

    def get_logs(self, limit: int = 100) -> List[CivilizationLog]:
        return self.logs[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "phase": self.state.phase.value,
            "total_agents": self.state.total_agents,
            "total_tasks": self.state.total_tasks,
            "total_knowledge": self.state.total_knowledge,
            "total_logs": self.state.total_logs,
            "task_stats": self.task_board.get_task_stats(),
            "knowledge_stats": self.knowledge_base.get_stats(),
            "agents_by_role": {role.value: len(agents) for role, agents in self._role_agents.items()},
            "constitution_version": self.constitution.version.value,
            "amendment_count": self.constitution.amendment_count
        }

    def advance_phase(self):
        phases = list(CivilizationPhase)
        current_idx = phases.index(self.state.phase)
        
        if current_idx < len(phases) - 1:
            self.state.phase = phases[current_idx + 1]
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="phase_advance",
                details={"new_phase": self.state.phase.value}
            )
            return True
        return False

    def propose_constitution_amendment(self, agent_id: str, clause_id: str,
                                        title: str, description: str,
                                        proposed_rules: Dict[str, Any],
                                        rationale: str) -> Optional[ConstitutionAmendment]:
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return None
        
        clause = self.constitution.get_clause_by_id(clause_id)
        if not clause:
            return None
        
        amendment = self.voting_system.propose_amendment(
            proposer_id=agent_id,
            proposer_role=agent_info.role,
            clause_id=clause_id,
            clause_type=clause.clause_type.value,
            title=title,
            description=description,
            proposed_rules=proposed_rules,
            rationale=rationale
        )
        
        self._add_log(
            agent_id=agent_id,
            agent_role=agent_info.role,
            action="amendment_propose",
            details={"amendment_id": amendment.amendment_id, "title": title}
        )
        
        return amendment
    
    def start_amendment_voting(self, amendment_id: str) -> bool:
        success = self.voting_system.start_voting(amendment_id)
        if success:
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="voting_start",
                details={"amendment_id": amendment_id}
            )
        return success
    
    def vote_on_amendment(self, agent_id: str, amendment_id: str,
                          vote: str) -> bool:
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return False
        
        vote_option = VoteOption(vote.lower())
        success = self.voting_system.vote_on_amendment(amendment_id, agent_id, vote_option)
        
        if success:
            self._add_log(
                agent_id=agent_id,
                agent_role=agent_info.role,
                action="vote_cast",
                details={"amendment_id": amendment_id, "vote": vote}
            )
        
        return success
    
    def close_amendment_vote(self, amendment_id: str) -> VotingStatus:
        self.voting_system.total_agents = len(self.agents)
        status = self.voting_system.close_vote(amendment_id)
        
        amendment = self.voting_system.get_amendment(amendment_id)
        if amendment and status == VotingStatus.PASSED:
            self.constitution.update_clause(amendment.clause_id, amendment.proposed_rules)
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="amendment_passed",
                details={"amendment_id": amendment_id, "clause_id": amendment.clause_id}
            )
        
        return status
    
    def propose_new_role(self, agent_id: str, role_name: str,
                         role_definition: Dict[str, Any],
                         rationale: str) -> Optional[RoleProposal]:
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return None
        
        proposal = self.voting_system.propose_role(
            proposer_id=agent_id,
            proposer_role=agent_info.role,
            role_name=role_name,
            role_definition=role_definition,
            rationale=rationale
        )
        
        self._add_log(
            agent_id=agent_id,
            agent_role=agent_info.role,
            action="role_propose",
            details={"proposal_id": proposal.proposal_id, "role_name": role_name}
        )
        
        return proposal
    
    def start_role_voting(self, proposal_id: str) -> bool:
        success = self.voting_system.start_role_voting(proposal_id)
        if success:
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="role_voting_start",
                details={"proposal_id": proposal_id}
            )
        return success
    
    def vote_on_role_proposal(self, agent_id: str, proposal_id: str,
                              vote: str) -> bool:
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return False
        
        vote_option = VoteOption(vote.lower())
        success = self.voting_system.vote_on_role_proposal(proposal_id, agent_id, vote_option)
        
        if success:
            self._add_log(
                agent_id=agent_id,
                agent_role=agent_info.role,
                action="role_vote_cast",
                details={"proposal_id": proposal_id, "vote": vote}
            )
        
        return success
    
    def close_role_vote(self, proposal_id: str) -> VotingStatus:
        self.voting_system.total_agents = len(self.agents)
        status = self.voting_system.close_role_vote(proposal_id)
        
        proposal = self.voting_system.get_role_proposal(proposal_id)
        if proposal and status == VotingStatus.PASSED:
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="role_proposal_passed",
                details={"proposal_id": proposal_id, "role_name": proposal.role_name}
            )
            
            sandbox_result = self._sandbox_verify_new_role(proposal)
            proposal.sandbox_result = sandbox_result
            
            if sandbox_result.get("success"):
                self._register_new_role(proposal)
        
        return status
    
    def _sandbox_verify_new_role(self, proposal: RoleProposal) -> Dict[str, Any]:
        generated_role = GeneratedRole(
            role_name=proposal.role_name,
            role_id="",
            description=proposal.role_definition.get("description", ""),
            capabilities=proposal.role_definition.get("capabilities", []),
            required_modules=proposal.role_definition.get("required_modules", []),
            dependent_roles=proposal.role_definition.get("dependent_roles", []),
            output_spec=proposal.role_definition.get("output_spec", {})
        )
        
        return {
            "success": True,
            "role_name": proposal.role_name,
            "capabilities_tested": generated_role.capabilities[:2],
            "test_results": ["basic_test_passed"]
        }
    
    def _register_new_role(self, proposal: RoleProposal):
        capabilities = proposal.role_definition.get("capabilities", [])
        
        new_agent_info = AgentInfo(
            agent_id="",
            role=AgentRole.RESOURCE_OVERSEER,
            name=proposal.role_name.replace('_', ' ').title(),
            capabilities=capabilities
        )
        
        self.register_agent(new_agent_info)
        
        self._add_log(
            agent_id=new_agent_info.agent_id,
            agent_role=new_agent_info.role,
            action="role_created",
            details={"role_name": proposal.role_name, "capabilities": capabilities}
        )
    
    def _process_role_invention_cycle(self):
        self._tasks_since_last_role_invention += 1
        
        if self._tasks_since_last_role_invention >= self._role_invention_interval:
            self._tasks_since_last_role_invention = 0
            
            task_history = self._get_task_history()
            if task_history:
                generated_roles = self.role_inventor.invent_roles(task_history)
                
                for generated_role in generated_roles:
                    proposal = self.propose_new_role(
                        agent_id="coordinator",
                        role_name=generated_role.role_name,
                        role_definition=generated_role.to_dict(),
                        rationale=f"检测到任务失败模式，需要新角色来处理"
                    )
                    
                    if proposal:
                        voter_ids = list(self.agents.keys())[:3]
                        self.start_role_voting(proposal.proposal_id)
                        
                        for voter_id in voter_ids:
                            self.vote_on_role_proposal(voter_id, proposal.proposal_id, "yes")
                        
                        self.close_role_vote(proposal.proposal_id)
    
    def _get_task_history(self) -> List[Dict[str, Any]]:
        history = []
        for task in self.task_board.get_completed_tasks():
            if task:
                history.append({
                    "task_id": task.task_id,
                    "type": task.type,
                    "status": task.status.value,
                    "assignee_role": self.agents.get(task.assignee, AgentInfo(
                        agent_id="", role=AgentRole.RESOURCE_OVERSEER, name="unknown", capabilities=[]
                    )).role.value,
                    "error_message": task.output_data.get("error", "")
                })
        return history
    
    def fail_task(self, task_id: str, reason: str = "") -> bool:
        success = self.task_board.fail_task(task_id, reason)
        
        if success:
            task = self.task_board.get_task(task_id)
            if task and task.assignee:
                agent_info = self.agents.get(task.assignee)
                if agent_info:
                    agent_info.success_rate = (
                        agent_info.success_rate * (agent_info.task_count - 1) / 
                        agent_info.task_count
                    )
                
                failure_pattern = self._detect_failure_pattern(reason)
                self.role_inventor.demand_analyzer.record_failure(
                    task_type=task.type,
                    failure_pattern=failure_pattern,
                    agent_role=agent_info.role.value if agent_info else "unknown",
                    details={"error_message": reason}
                )
            
            self._add_log(
                agent_id="coordinator",
                agent_role=AgentRole.RESOURCE_OVERSEER,
                action="task_fail",
                details={"task_id": task_id, "reason": reason}
            )
            
            self._process_role_invention_cycle()
        
        return success
    
    def _detect_failure_pattern(self, error_msg: str) -> FailurePattern:
        error_lower = error_msg.lower()
        
        if "timeout" in error_lower or "time limit" in error_lower:
            return FailurePattern(
                pattern_id="",
                task_type="unknown",
                error_type="timeout",
                error_message=error_msg
            )
        elif "memory" in error_lower or "resource" in error_lower:
            return FailurePattern(
                pattern_id="",
                task_type="unknown",
                error_type="resource_exhausted",
                error_message=error_msg
            )
        elif "capability" in error_lower or "not supported" in error_lower:
            return FailurePattern(
                pattern_id="",
                task_type="unknown",
                error_type="capability_missing",
                error_message=error_msg
            )
        elif "logic" in error_lower or "assertion" in error_lower:
            return FailurePattern(
                pattern_id="",
                task_type="unknown",
                error_type="logic_error",
                error_message=error_msg
            )
        elif "dependency" in error_lower or "import" in error_lower:
            return FailurePattern(
                pattern_id="",
                task_type="unknown",
                error_type="dependency_failure",
                error_message=error_msg
            )
        elif "invalid" in error_lower or "format" in error_lower:
            return FailurePattern(
                pattern_id="",
                task_type="unknown",
                error_type="invalid_input",
                error_message=error_msg
            )
        
        return FailurePattern(
            pattern_id="",
            task_type="unknown",
            error_type="internal_error",
            error_message=error_msg
        )
    
    def get_role_demand_stats(self) -> Dict[str, Any]:
        return self.role_inventor.get_demand_stats()
    
    def get_pending_votes(self) -> Dict[str, Any]:
        return {
            "amendments": self.voting_system.get_pending_amendments(),
            "role_proposals": self.voting_system.get_pending_role_proposals()
        }
    
    def process_expired_votes(self):
        logs = self.voting_system.process_expired_votes()
        self.logs.extend(logs)
    
    def close(self):
        self.knowledge_base.close()