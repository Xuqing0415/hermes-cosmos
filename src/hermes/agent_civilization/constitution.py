import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .types import AgentRole, CivilizationPhase, Vote


class ConstitutionVersion(Enum):
    V1 = "v1"
    V2 = "v2"


class ClauseType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    ROLE_CREATION = "role_creation"
    PHASE_ADVANCEMENT = "phase_advancement"
    CONFLICT_RESOLUTION = "conflict_resolution"
    VOTING_RULES = "voting_rules"
    KNOWLEDGE_SHARING = "knowledge_sharing"


@dataclass
class ConstitutionClause:
    clause_id: str
    clause_type: ClauseType
    title: str
    description: str
    rules: Dict[str, Any]
    version: str = "1.0"
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class Constitution:
    version: ConstitutionVersion = ConstitutionVersion.V1
    clauses: List[ConstitutionClause] = field(default_factory=list)
    created_at: float = 0.0
    last_amended_at: float = 0.0
    amendment_count: int = 0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
            self.last_amended_at = self.created_at
            
            if not self.clauses and self.version == ConstitutionVersion.V1:
                self._initialize_default_clauses()

    def _initialize_default_clauses(self):
        self.clauses = [
            ConstitutionClause(
                clause_id="task_assign_001",
                clause_type=ClauseType.TASK_ASSIGNMENT,
                title="任务竞标规则",
                description="任务分配基于能力匹配系数和历史成功率",
                rules={
                    "weight_capability_match": 0.6,
                    "weight_success_rate": 0.4,
                    "require_bid": False,
                    "max_bidders": 5
                }
            ),
            ConstitutionClause(
                clause_id="role_create_001",
                title="角色创建规则",
                clause_type=ClauseType.ROLE_CREATION,
                description="新角色创建需要联署",
                rules={
                    "required_sponsors": 3,
                    "required_roles": ["TEST_OFFICER", "FIXER", "PROVER"],
                    "sandbox_verification_required": True
                }
            ),
            ConstitutionClause(
                clause_id="phase_adv_001",
                title="阶段升级规则",
                clause_type=ClauseType.PHASE_ADVANCEMENT,
                description="文明阶段升级条件",
                rules={
                    "bronze_age": {
                        "min_agents": 2,
                        "required_roles": ["TEST_OFFICER", "FIXER"]
                    },
                    "iron_age": {
                        "min_agents": 5,
                        "required_roles": ["TEST_OFFICER", "FIXER", "PROVER"],
                        "min_role_counts": {"TEST_OFFICER": 2, "PROVER": 1}
                    },
                    "industrial_age": {
                        "min_agents": 8,
                        "completed_tasks": 100
                    }
                }
            ),
            ConstitutionClause(
                clause_id="conflict_res_001",
                title="冲突仲裁规则",
                clause_type=ClauseType.CONFLICT_RESOLUTION,
                description="仲裁委员会组成规则",
                rules={
                    "arbitrator_count": 3,
                    "exclude_conflicting_roles": True,
                    "majority_required": 2
                }
            ),
            ConstitutionClause(
                clause_id="voting_001",
                title="投票规则",
                clause_type=ClauseType.VOTING_RULES,
                description="投票权重和通过条件",
                rules={
                    "quorum_percentage": 0.5,
                    "majority_percentage": 0.6,
                    "weight_by_influence": True,
                    "influence_factors": ["success_rate", "task_count", "knowledge_contributions"]
                }
            ),
            ConstitutionClause(
                clause_id="knowledge_001",
                title="知识共享规则",
                clause_type=ClauseType.KNOWLEDGE_SHARING,
                description="知识贡献和奖励规则",
                rules={
                    "sharing_required": True,
                    "reward_for_usage": True,
                    "max_knowledge_per_agent": 100
                }
            )
        ]

    def get_clause(self, clause_type: ClauseType) -> Optional[ConstitutionClause]:
        for clause in self.clauses:
            if clause.clause_type == clause_type:
                return clause
        return None

    def get_clause_by_id(self, clause_id: str) -> Optional[ConstitutionClause]:
        for clause in self.clauses:
            if clause.clause_id == clause_id:
                return clause
        return None

    def add_clause(self, clause: ConstitutionClause):
        self.clauses.append(clause)
        self.amendment_count += 1
        self.last_amended_at = time.time()

    def update_clause(self, clause_id: str, rules: Dict[str, Any]):
        clause = self.get_clause_by_id(clause_id)
        if clause:
            clause.rules.update(rules)
            clause.version = str(float(clause.version) + 0.1)
            self.amendment_count += 1
            self.last_amended_at = time.time()

    def remove_clause(self, clause_id: str) -> bool:
        for i, clause in enumerate(self.clauses):
            if clause.clause_id == clause_id:
                self.clauses.pop(i)
                self.amendment_count += 1
                self.last_amended_at = time.time()
                return True
        return False

    def evaluate_task_assignment(self, task_type: str, candidates: List[Dict[str, Any]]) -> Optional[str]:
        clause = self.get_clause(ClauseType.TASK_ASSIGNMENT)
        if not clause or not candidates:
            return None
        
        rules = clause.rules
        weighted_scores = []
        
        for candidate in candidates:
            capability_match = candidate.get("capability_match", 0.5)
            success_rate = candidate.get("success_rate", 0.5)
            
            score = (
                capability_match * rules.get("weight_capability_match", 0.6) +
                success_rate * rules.get("weight_success_rate", 0.4)
            )
            weighted_scores.append((candidate["agent_id"], score))
        
        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        return weighted_scores[0][0] if weighted_scores else None

    def check_phase_advancement(self, agent_counts: Dict[str, int], total_tasks: int) -> Optional[CivilizationPhase]:
        clause = self.get_clause(ClauseType.PHASE_ADVANCEMENT)
        if not clause:
            return None
        
        rules = clause.rules
        
        if self._check_phase_condition("industrial_age", rules, agent_counts, total_tasks):
            return CivilizationPhase.INDUSTRIAL_AGE
        elif self._check_phase_condition("iron_age", rules, agent_counts, total_tasks):
            return CivilizationPhase.IRON_AGE
        elif self._check_phase_condition("bronze_age", rules, agent_counts, total_tasks):
            return CivilizationPhase.BRONZE_AGE
        
        return None

    def _check_phase_condition(self, phase_name: str, rules: Dict[str, Any], 
                               agent_counts: Dict[str, int], total_tasks: int) -> bool:
        conditions = rules.get(phase_name)
        if not conditions:
            return False
        
        if conditions.get("min_agents") and sum(agent_counts.values()) < conditions["min_agents"]:
            return False
        
        lower_counts = {k.lower(): v for k, v in agent_counts.items()}
        
        required_roles = conditions.get("required_roles", [])
        for role in required_roles:
            if lower_counts.get(role.lower(), 0) < 1:
                return False
        
        min_role_counts = conditions.get("min_role_counts", {})
        for role, count in min_role_counts.items():
            if lower_counts.get(role.lower(), 0) < count:
                return False
        
        if conditions.get("completed_tasks") and total_tasks < conditions["completed_tasks"]:
            return False
        
        return True

    def check_role_creation_requirements(self, sponsors: List[str], sponsor_roles: List[str]) -> bool:
        clause = self.get_clause(ClauseType.ROLE_CREATION)
        if not clause:
            return False
        
        rules = clause.rules
        
        if len(sponsors) < rules.get("required_sponsors", 3):
            return False
        
        required_roles = rules.get("required_roles", [])
        for role in required_roles:
            if role not in sponsor_roles:
                return False
        
        return True

    def check_voting_quorum(self, total_agents: int, voting_agents: int) -> bool:
        clause = self.get_clause(ClauseType.VOTING_RULES)
        if not clause:
            return True
        
        quorum_pct = clause.rules.get("quorum_percentage", 0.5)
        return voting_agents >= total_agents * quorum_pct

    def check_voting_majority(self, yes_votes: int, total_votes: int) -> bool:
        clause = self.get_clause(ClauseType.VOTING_RULES)
        if not clause:
            return yes_votes > total_votes / 2
        
        majority_pct = clause.rules.get("majority_percentage", 0.6)
        return yes_votes >= total_votes * majority_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version.value,
            "created_at": self.created_at,
            "last_amended_at": self.last_amended_at,
            "amendment_count": self.amendment_count,
            "clauses": [
                {
                    "clause_id": clause.clause_id,
                    "clause_type": clause.clause_type.value,
                    "title": clause.title,
                    "description": clause.description,
                    "rules": clause.rules,
                    "version": clause.version,
                    "created_at": clause.created_at
                }
                for clause in self.clauses
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constitution':
        constitution = cls(
            version=ConstitutionVersion(data.get("version", "v1")),
            created_at=data.get("created_at", time.time()),
            last_amended_at=data.get("last_amended_at", time.time()),
            amendment_count=data.get("amendment_count", 0),
            clauses=[]
        )
        
        for clause_data in data.get("clauses", []):
            constitution.clauses.append(ConstitutionClause(
                clause_id=clause_data["clause_id"],
                clause_type=ClauseType(clause_data["clause_type"]),
                title=clause_data["title"],
                description=clause_data["description"],
                rules=clause_data["rules"],
                version=clause_data.get("version", "1.0"),
                created_at=clause_data.get("created_at", time.time())
            ))
        
        return constitution

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'Constitution':
        return cls.from_dict(json.loads(json_str))