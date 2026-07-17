from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DomainContext:
    domain: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "path": self.path,
            "metadata": self.metadata,
            "issues": self.issues,
            "artifacts": self.artifacts
        }


@dataclass
class PainPoint:
    id: str
    type: str
    severity: str
    message: str
    location: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "context": self.context
        }


@dataclass
class PatchPlan:
    id: str
    pain_point_id: str
    description: str
    operations: List[Dict[str, Any]]
    estimated_effort: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pain_point_id": self.pain_point_id,
            "description": self.description,
            "operations": self.operations,
            "estimated_effort": self.estimated_effort
        }


@dataclass
class ExecutionResult:
    success: bool
    patch_plan_id: str
    message: str
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "patch_plan_id": self.patch_plan_id,
            "message": self.message,
            "verification_results": self.verification_results,
            "artifacts": self.artifacts
        }


class Perceiver(ABC):
    @abstractmethod
    def detect(self, context: DomainContext) -> List[PainPoint]:
        pass


class Sage(ABC):
    @abstractmethod
    def generate_patch(self, context: DomainContext, pain_point: PainPoint) -> PatchPlan:
        pass


class Knight(ABC):
    @abstractmethod
    def execute(self, context: DomainContext, patch_plan: PatchPlan) -> ExecutionResult:
        pass
    
    @abstractmethod
    def verify(self, context: DomainContext, patch_plan: PatchPlan) -> ExecutionResult:
        pass


class DomainPlugin(ABC):
    @property
    @abstractmethod
    def domain_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def create_context(self, path: str, **kwargs) -> DomainContext:
        pass
    
    @abstractmethod
    def get_perceiver(self) -> Perceiver:
        pass
    
    @abstractmethod
    def get_sage(self) -> Sage:
        pass
    
    @abstractmethod
    def get_knight(self) -> Knight:
        pass