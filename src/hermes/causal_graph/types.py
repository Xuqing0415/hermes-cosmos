from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time

class NodeType(Enum):
    ROOT_CAUSE = "root_cause"
    INTERMEDIATE = "intermediate"
    SYMPTOM = "symptom"
    OBSERVATION = "observation"

class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class CausalNode:
    node_id: str
    label: str
    node_type: NodeType
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    relation: str
    confidence: float = 0.0
    description: str = ""
    
@dataclass
class CausalGraph:
    nodes: List[CausalNode] = field(default_factory=list)
    edges: List[CausalEdge] = field(default_factory=list)
    
    def add_node(self, node: CausalNode):
        self.nodes.append(node)
    
    def add_edge(self, edge: CausalEdge):
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[CausalNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def get_root_causes(self) -> List[CausalNode]:
        return [n for n in self.nodes if n.node_type == NodeType.ROOT_CAUSE]
    
    def get_symptoms(self) -> List[CausalNode]:
        return [n for n in self.nodes if n.node_type == NodeType.SYMPTOM]
    
    def get_children(self, node_id: str) -> List[CausalNode]:
        child_ids = [e.target_id for e in self.edges if e.source_id == node_id]
        return [self.get_node(cid) for cid in child_ids if self.get_node(cid)]
    
    def get_parents(self, node_id: str) -> List[CausalNode]:
        parent_ids = [e.source_id for e in self.edges if e.target_id == node_id]
        return [self.get_node(pid) for pid in parent_ids if self.get_node(pid)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [{"node_id": n.node_id, "label": n.label, "type": n.node_type.value, 
                       "description": n.description, "confidence": n.confidence} for n in self.nodes],
            "edges": [{"source": e.source_id, "target": e.target_id, 
                       "relation": e.relation, "confidence": e.confidence} for e in self.edges]
        }

@dataclass
class CausalExplanation:
    graph: CausalGraph
    root_causes: List[CausalNode]
    explanation: str
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
