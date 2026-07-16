import time
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .types import AgentRole, Vote
from .constitution import Constitution, ClauseType, ConstitutionClause


class AmendmentStatus(Enum):
    PROPOSED = "proposed"
    VOTING = "voting"
    PASSED = "passed"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


@dataclass
class Amendment:
    amendment_id: str
    proposer_id: str
    proposer_role: AgentRole
    title: str
    description: str
    clause_type: ClauseType
    changes: Dict[str, Any]
    status: AmendmentStatus = AmendmentStatus.PROPOSED
    votes: Dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    voting_end_time: float = 0.0
    voters: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.amendment_id == "":
            self.amendment_id = str(uuid.uuid4())[:8]
        if self.created_at == 0.0:
            self.created_at = time.time()


class ConstitutionAmendmentSystem:
    def __init__(self, constitution: Constitution):
        self.constitution = constitution
        self.amendments: Dict[str, Amendment] = {}
        self._active_amendments: List[str] = []
        self._on_amendment_applied = None
    
    def set_on_amendment_applied_callback(self, callback):
        self._on_amendment_applied = callback
    
    def propose_amendment(self, proposer_id: str, proposer_role: AgentRole,
                          title: str, description: str, clause_type: ClauseType,
                          changes: Dict[str, Any]) -> Amendment:
        amendment = Amendment(
            amendment_id="",
            proposer_id=proposer_id,
            proposer_role=proposer_role,
            title=title,
            description=description,
            clause_type=clause_type,
            changes=changes
        )
        
        self.amendments[amendment.amendment_id] = amendment
        self._active_amendments.append(amendment.amendment_id)
        
        return amendment
    
    def start_voting(self, amendment_id: str, voters: List[str], duration_seconds: int = 300) -> bool:
        amendment = self.amendments.get(amendment_id)
        if not amendment or amendment.status != AmendmentStatus.PROPOSED:
            return False
        
        amendment.status = AmendmentStatus.VOTING
        amendment.voters = voters
        amendment.voting_end_time = time.time() + duration_seconds
        
        return True
    
    def cast_vote(self, amendment_id: str, voter_id: str, vote: str) -> bool:
        amendment = self.amendments.get(amendment_id)
        if not amendment:
            return False
        
        if amendment.status != AmendmentStatus.VOTING:
            return False
        
        if voter_id not in amendment.voters:
            return False
        
        if time.time() > amendment.voting_end_time:
            self._close_voting(amendment_id)
            return False
        
        amendment.votes[voter_id] = vote
        
        if len(amendment.votes) == len(amendment.voters):
            self._close_voting(amendment_id)
        
        return True
    
    def _close_voting(self, amendment_id: str):
        amendment = self.amendments.get(amendment_id)
        if not amendment:
            return
        
        total_voters = len(amendment.voters)
        yes_votes = sum(1 for v in amendment.votes.values() if v.lower() == "yes")
        
        quorum_ok = self.constitution.check_voting_quorum(total_voters, len(amendment.votes))
        majority_ok = self.constitution.check_voting_majority(yes_votes, len(amendment.votes))
        
        if quorum_ok and majority_ok:
            amendment.status = AmendmentStatus.PASSED
            self._apply_amendment(amendment)
        else:
            amendment.status = AmendmentStatus.REJECTED
        
        if amendment_id in self._active_amendments:
            self._active_amendments.remove(amendment_id)
    
    def _apply_amendment(self, amendment: Amendment):
        clause = self.constitution.get_clause(amendment.clause_type)
        
        if clause:
            clause.rules.update(amendment.changes)
            clause.version = str(float(clause.version) + 0.1)
            self.constitution.amendment_count += 1
            self.constitution.last_amended_at = time.time()
        else:
            new_clause = ConstitutionClause(
                clause_id=f"amendment_{amendment.amendment_id}",
                clause_type=amendment.clause_type,
                title=amendment.title,
                description=amendment.description,
                rules=amendment.changes
            )
            self.constitution.add_clause(new_clause)
        
        if self._on_amendment_applied:
            self._on_amendment_applied(amendment)
    
    def withdraw_amendment(self, amendment_id: str, proposer_id: str) -> bool:
        amendment = self.amendments.get(amendment_id)
        if not amendment:
            return False
        
        if amendment.proposer_id != proposer_id:
            return False
        
        if amendment.status == AmendmentStatus.VOTING:
            return False
        
        amendment.status = AmendmentStatus.WITHDRAWN
        
        if amendment_id in self._active_amendments:
            self._active_amendments.remove(amendment_id)
        
        return True
    
    def get_amendment(self, amendment_id: str) -> Optional[Amendment]:
        return self.amendments.get(amendment_id)
    
    def get_active_amendments(self) -> List[Amendment]:
        return [self.amendments[aid] for aid in self._active_amendments]
    
    def get_amendments_by_status(self, status: AmendmentStatus) -> List[Amendment]:
        return [a for a in self.amendments.values() if a.status == status]
    
    def get_voting_status(self, amendment_id: str) -> Dict[str, Any]:
        amendment = self.amendments.get(amendment_id)
        if not amendment:
            return {"error": "Amendment not found"}
        
        yes_votes = sum(1 for v in amendment.votes.values() if v.lower() == "yes")
        no_votes = sum(1 for v in amendment.votes.values() if v.lower() == "no")
        
        return {
            "amendment_id": amendment.amendment_id,
            "title": amendment.title,
            "status": amendment.status.value,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "total_voters": len(amendment.voters),
            "voted_count": len(amendment.votes),
            "voting_ends_at": amendment.voting_end_time,
            "time_remaining": max(0, amendment.voting_end_time - time.time())
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_amendments": len(self.amendments),
            "active_amendments": len(self._active_amendments),
            "proposed": len(self.get_amendments_by_status(AmendmentStatus.PROPOSED)),
            "voting": len(self.get_amendments_by_status(AmendmentStatus.VOTING)),
            "passed": len(self.get_amendments_by_status(AmendmentStatus.PASSED)),
            "rejected": len(self.get_amendments_by_status(AmendmentStatus.REJECTED)),
            "constitution_amendments": self.constitution.amendment_count
        }


AmendmentStatus.__module__ = __name__