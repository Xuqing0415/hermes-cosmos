import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .types import AgentRole, CivilizationLog


class VoteOption(Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class AmendmentStatus(Enum):
    PROPOSED = "proposed"
    VOTING = "voting"
    PASSED = "passed"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


@dataclass
class ConstitutionAmendment:
    amendment_id: str
    proposer_id: str
    proposer_role: AgentRole
    clause_id: str
    clause_type: str
    title: str
    description: str
    proposed_rules: Dict[str, Any]
    rationale: str
    status: AmendmentStatus = AmendmentStatus.PROPOSED
    votes: Dict[str, VoteOption] = field(default_factory=dict)
    created_at: float = 0.0
    voting_deadline: float = 0.0
    passed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.amendment_id == "":
            self.amendment_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def add_vote(self, agent_id: str, vote: VoteOption):
        self.votes[agent_id] = vote
    
    def get_vote_counts(self) -> Dict[str, int]:
        counts = {"yes": 0, "no": 0, "abstain": 0}
        for vote in self.votes.values():
            counts[vote.value] += 1
        return counts
    
    def is_voting_closed(self) -> bool:
        return time.time() >= self.voting_deadline
    
    def has_quorum(self, total_agents: int, quorum_pct: float = 0.5) -> bool:
        return len(self.votes) >= total_agents * quorum_pct
    
    def is_passed(self, majority_pct: float = 0.6) -> bool:
        counts = self.get_vote_counts()
        total_votes = counts["yes"] + counts["no"]
        if total_votes == 0:
            return False
        return counts["yes"] >= total_votes * majority_pct


@dataclass
class RoleProposal:
    proposal_id: str
    proposer_id: str
    proposer_role: AgentRole
    role_name: str
    role_definition: Dict[str, Any]
    rationale: str
    status: AmendmentStatus = AmendmentStatus.PROPOSED
    votes: Dict[str, VoteOption] = field(default_factory=dict)
    sponsors: List[str] = field(default_factory=list)
    created_at: float = 0.0
    voting_deadline: float = 0.0
    sandbox_result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.proposal_id == "":
            self.proposal_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def add_vote(self, agent_id: str, vote: VoteOption):
        self.votes[agent_id] = vote
    
    def add_sponsor(self, agent_id: str):
        if agent_id not in self.sponsors:
            self.sponsors.append(agent_id)
    
    def get_vote_counts(self) -> Dict[str, int]:
        counts = {"yes": 0, "no": 0, "abstain": 0}
        for vote in self.votes.values():
            counts[vote.value] += 1
        return counts
    
    def is_voting_closed(self) -> bool:
        return time.time() >= self.voting_deadline
    
    def has_quorum(self, total_agents: int, quorum_pct: float = 0.5) -> bool:
        return len(self.votes) >= total_agents * quorum_pct
    
    def is_passed(self, majority_pct: float = 0.6) -> bool:
        counts = self.get_vote_counts()
        total_votes = counts["yes"] + counts["no"]
        if total_votes == 0:
            return False
        return counts["yes"] >= total_votes * majority_pct


class ConstitutionVoting:
    def __init__(self):
        self.amendments: Dict[str, ConstitutionAmendment] = {}
        self.role_proposals: Dict[str, RoleProposal] = {}
        self.total_agents: int = 0
        self.voting_timeout: float = 30.0
        self.quorum_percentage: float = 0.5
        self.majority_percentage: float = 0.6
    
    def propose_amendment(self, proposer_id: str, proposer_role: AgentRole,
                          clause_id: str, clause_type: str, title: str,
                          description: str, proposed_rules: Dict[str, Any],
                          rationale: str) -> ConstitutionAmendment:
        amendment = ConstitutionAmendment(
            amendment_id="",
            proposer_id=proposer_id,
            proposer_role=proposer_role,
            clause_id=clause_id,
            clause_type=clause_type,
            title=title,
            description=description,
            proposed_rules=proposed_rules,
            rationale=rationale,
            voting_deadline=time.time() + self.voting_timeout
        )
        self.amendments[amendment.amendment_id] = amendment
        return amendment
    
    def propose_role(self, proposer_id: str, proposer_role: AgentRole,
                     role_name: str, role_definition: Dict[str, Any],
                     rationale: str) -> RoleProposal:
        proposal = RoleProposal(
            proposal_id="",
            proposer_id=proposer_id,
            proposer_role=proposer_role,
            role_name=role_name,
            role_definition=role_definition,
            rationale=rationale,
            voting_deadline=time.time() + self.voting_timeout
        )
        self.role_proposals[proposal.proposal_id] = proposal
        return proposal
    
    def vote_on_amendment(self, amendment_id: str, agent_id: str,
                          vote: VoteOption) -> bool:
        amendment = self.amendments.get(amendment_id)
        if not amendment or amendment.status != AmendmentStatus.VOTING:
            return False
        
        amendment.add_vote(agent_id, vote)
        return True
    
    def vote_on_role_proposal(self, proposal_id: str, agent_id: str,
                              vote: VoteOption) -> bool:
        proposal = self.role_proposals.get(proposal_id)
        if not proposal or proposal.status != AmendmentStatus.VOTING:
            return False
        
        proposal.add_vote(agent_id, vote)
        return True
    
    def start_voting(self, amendment_id: str) -> bool:
        amendment = self.amendments.get(amendment_id)
        if not amendment:
            return False
        
        amendment.status = AmendmentStatus.VOTING
        amendment.voting_deadline = time.time() + self.voting_timeout
        return True
    
    def start_role_voting(self, proposal_id: str) -> bool:
        proposal = self.role_proposals.get(proposal_id)
        if not proposal:
            return False
        
        proposal.status = AmendmentStatus.VOTING
        proposal.voting_deadline = time.time() + self.voting_timeout
        return True
    
    def close_vote(self, amendment_id: str) -> AmendmentStatus:
        amendment = self.amendments.get(amendment_id)
        if not amendment:
            return AmendmentStatus.REJECTED
        
        effective_total = max(self.total_agents, 1)
        if amendment.has_quorum(effective_total) and amendment.is_passed(self.majority_percentage):
            amendment.status = AmendmentStatus.PASSED
            amendment.passed_at = time.time()
        else:
            amendment.status = AmendmentStatus.REJECTED
        
        return amendment.status
    
    def close_role_vote(self, proposal_id: str) -> AmendmentStatus:
        proposal = self.role_proposals.get(proposal_id)
        if not proposal:
            return AmendmentStatus.REJECTED
        
        effective_total = max(self.total_agents, 1)
        if proposal.has_quorum(effective_total) and proposal.is_passed(self.majority_percentage):
            proposal.status = AmendmentStatus.PASSED
            proposal.passed_at = time.time()
        else:
            proposal.status = AmendmentStatus.REJECTED
        
        return proposal.status
    
    def get_pending_amendments(self) -> List[ConstitutionAmendment]:
        return [a for a in self.amendments.values() 
                if a.status in [AmendmentStatus.PROPOSED, AmendmentStatus.VOTING]]
    
    def get_pending_role_proposals(self) -> List[RoleProposal]:
        return [p for p in self.role_proposals.values() 
                if p.status in [AmendmentStatus.PROPOSED, AmendmentStatus.VOTING]]
    
    def get_amendment(self, amendment_id: str) -> Optional[ConstitutionAmendment]:
        return self.amendments.get(amendment_id)
    
    def get_role_proposal(self, proposal_id: str) -> Optional[RoleProposal]:
        return self.role_proposals.get(proposal_id)
    
    def process_expired_votes(self) -> List[CivilizationLog]:
        logs = []
        
        for amendment in self.amendments.values():
            if amendment.status == AmendmentStatus.VOTING and amendment.is_voting_closed():
                self.close_vote(amendment.amendment_id)
                logs.append(CivilizationLog(
                    log_id="",
                    agent_id="system",
                    agent_role=AgentRole.RESOURCE_OVERSEER,
                    action="vote_closed",
                    details={
                        "amendment_id": amendment.amendment_id,
                        "title": amendment.title,
                        "status": amendment.status.value,
                        "votes": amendment.get_vote_counts()
                    }
                ))
        
        for proposal in self.role_proposals.values():
            if proposal.status == AmendmentStatus.VOTING and proposal.is_voting_closed():
                self.close_role_vote(proposal.proposal_id)
                logs.append(CivilizationLog(
                    log_id="",
                    agent_id="system",
                    agent_role=AgentRole.RESOURCE_OVERSEER,
                    action="role_vote_closed",
                    details={
                        "proposal_id": proposal.proposal_id,
                        "role_name": proposal.role_name,
                        "status": proposal.status.value,
                        "votes": proposal.get_vote_counts()
                    }
                ))
        
        return logs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "amendments": {
                aid: {
                    "amendment_id": a.amendment_id,
                    "proposer_id": a.proposer_id,
                    "proposer_role": a.proposer_role.value,
                    "clause_id": a.clause_id,
                    "clause_type": a.clause_type,
                    "title": a.title,
                    "description": a.description,
                    "proposed_rules": a.proposed_rules,
                    "rationale": a.rationale,
                    "status": a.status.value,
                    "votes": {k: v.value for k, v in a.votes.items()},
                    "created_at": a.created_at,
                    "voting_deadline": a.voting_deadline,
                    "passed_at": a.passed_at
                }
                for aid, a in self.amendments.items()
            },
            "role_proposals": {
                pid: {
                    "proposal_id": p.proposal_id,
                    "proposer_id": p.proposer_id,
                    "proposer_role": p.proposer_role.value,
                    "role_name": p.role_name,
                    "role_definition": p.role_definition,
                    "rationale": p.rationale,
                    "status": p.status.value,
                    "votes": {k: v.value for k, v in p.votes.items()},
                    "sponsors": p.sponsors,
                    "created_at": p.created_at,
                    "voting_deadline": p.voting_deadline,
                    "sandbox_result": p.sandbox_result
                }
                for pid, p in self.role_proposals.items()
            },
            "voting_timeout": self.voting_timeout,
            "quorum_percentage": self.quorum_percentage,
            "majority_percentage": self.majority_percentage
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)