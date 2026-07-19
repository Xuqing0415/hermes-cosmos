from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from .abstract_pattern_extractor import AbstractPattern, AbstractPatternType
from .pattern_similarity_engine import SimilarityMatch
from .cross_domain_translator import UniversalFixTemplate


@dataclass
class KnowledgeNode:
    id: str
    pattern_type: AbstractPatternType
    domains: List[str] = field(default_factory=list)
    representative_keywords: List[str] = field(default_factory=list)
    occurrences: int = 0
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "domains": self.domains,
            "representative_keywords": self.representative_keywords,
            "occurrences": self.occurrences,
            "confidence": round(self.confidence, 2),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class KnowledgeEdge:
    source_id: str
    target_id: str
    similarity: float
    relationship_type: str = "similar"
    domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "similarity": round(self.similarity, 2),
            "relationship_type": self.relationship_type,
            "domains": self.domains
        }


@dataclass
class KnowledgeGraph:
    nodes: List[KnowledgeNode] = field(default_factory=list)
    edges: List[KnowledgeEdge] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "last_updated": self.last_updated.isoformat(),
            "total_patterns": len(self.nodes),
            "total_relationships": len(self.edges)
        }


class KnowledgeAmalgamator:
    def __init__(self, storage_path: str = "knowledge_graph.json"):
        self._storage_path = storage_path
        self._graph = KnowledgeGraph()
        self._node_id_counter = 0

    def amalgamate_patterns(self, patterns: List[AbstractPattern]) -> int:
        new_patterns_count = 0

        for pattern in patterns:
            existing_node = self._find_or_create_node(pattern)
            if existing_node:
                self._update_node(existing_node, pattern)
                new_patterns_count += 1

        self._graph.last_updated = datetime.utcnow()
        return new_patterns_count

    def amalgamate_similarities(self, similarities: List[SimilarityMatch]) -> int:
        new_edges_count = 0

        for similarity in similarities:
            source_node = self._find_node_by_pattern(similarity.pattern_a)
            target_node = self._find_node_by_pattern(similarity.pattern_b)

            if source_node and target_node:
                if not self._edge_exists(source_node.id, target_node.id):
                    edge = KnowledgeEdge(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        similarity=similarity.similarity,
                        relationship_type="similar",
                        domains=[similarity.pattern_a.domain, similarity.pattern_b.domain]
                    )
                    self._graph.edges.append(edge)
                    new_edges_count += 1

        self._graph.last_updated = datetime.utcnow()
        return new_edges_count

    def get_graph(self) -> KnowledgeGraph:
        return self._graph

    def get_pattern_count(self) -> int:
        return len(self._graph.nodes)

    def get_relationship_count(self) -> int:
        return len(self._graph.edges)

    def save(self):
        data = self._graph.to_dict()
        directory = os.path.dirname(self._storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(self._storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self._storage_path):
            with open(self._storage_path, 'r') as f:
                data = json.load(f)

            self._graph.nodes = []
            for node_data in data.get("nodes", []):
                node = KnowledgeNode(
                    id=node_data["id"],
                    pattern_type=AbstractPatternType(node_data["pattern_type"]),
                    domains=node_data.get("domains", []),
                    representative_keywords=node_data.get("representative_keywords", []),
                    occurrences=node_data.get("occurrences", 0),
                    confidence=node_data.get("confidence", 0.0),
                    created_at=datetime.fromisoformat(node_data.get("created_at", datetime.utcnow().isoformat())),
                    updated_at=datetime.fromisoformat(node_data.get("updated_at", datetime.utcnow().isoformat()))
                )
                self._graph.nodes.append(node)
                node_id = int(node.id.split("-")[-1]) if "-" in node.id else 0
                self._node_id_counter = max(self._node_id_counter, node_id + 1)

            self._graph.edges = []
            for edge_data in data.get("edges", []):
                edge = KnowledgeEdge(
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    similarity=edge_data.get("similarity", 0.0),
                    relationship_type=edge_data.get("relationship_type", "similar"),
                    domains=edge_data.get("domains", [])
                )
                self._graph.edges.append(edge)

            if "last_updated" in data:
                self._graph.last_updated = datetime.fromisoformat(data["last_updated"])

    def get_universal_fixes(self) -> List[Dict[str, Any]]:
        fixes = []
        for node in self._graph.nodes:
            fix_entry = {
                "pattern_type": node.pattern_type.value,
                "domains": node.domains,
                "confidence": node.confidence,
                "occurrences": node.occurrences
            }
            fixes.append(fix_entry)
        return sorted(fixes, key=lambda x: x["occurrences"], reverse=True)

    def query_similar(self, pattern_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the knowledge graph for patterns similar to the given type."""
        matches = []
        target_node = None
        for node in self._graph.nodes:
            if node.pattern_type.value == pattern_type:
                target_node = node
                break

        if target_node is None:
            return []

        for edge in self._graph.edges:
            if edge.source_id == target_node.id:
                neighbor_id = edge.target_id
            elif edge.target_id == target_node.id:
                neighbor_id = edge.source_id
            else:
                continue

            neighbor = None
            for node in self._graph.nodes:
                if node.id == neighbor_id:
                    neighbor = node
                    break

            if neighbor:
                matches.append({
                    "pattern_type": neighbor.pattern_type.value,
                    "source_domain": edge.domains[0] if edge.domains else "unknown",
                    "target_domain": edge.domains[-1] if len(edge.domains) > 1 else edge.domains[0] if edge.domains else "unknown",
                    "similarity": edge.similarity,
                    "confidence": neighbor.confidence,
                    "occurrences": neighbor.occurrences,
                })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

    def add_cross_domain_link(self, source_domain: str, target_domain: str,
                               pattern_type: str, similarity: float, success: bool):
        """Record a cross-domain link in the knowledge graph."""
        source_node = None
        target_node = None
        for node in self._graph.nodes:
            if node.pattern_type.value == pattern_type:
                if source_domain in node.domains:
                    source_node = node
                if target_domain not in node.domains:
                    node.domains.append(target_domain)
                target_node = node

        if source_node is None:
            self._node_id_counter += 1
            source_node = KnowledgeNode(
                id=f"knode-{self._node_id_counter:04d}",
                pattern_type=AbstractPatternType(pattern_type) if any(
                    pt.value == pattern_type for pt in AbstractPatternType
                ) else AbstractPatternType.BOUNDARY_CHECK_MISSING,
                domains=[source_domain],
                occurrences=1,
                confidence=0.5
            )
            self._graph.nodes.append(source_node)

        if target_node is None or target_node.id == source_node.id:
            self._node_id_counter += 1
            target_node = KnowledgeNode(
                id=f"knode-{self._node_id_counter:04d}",
                pattern_type=source_node.pattern_type,
                domains=[target_domain],
                occurrences=1,
                confidence=0.5
            )
            self._graph.nodes.append(target_node)

        if not self._edge_exists(source_node.id, target_node.id):
            edge = KnowledgeEdge(
                source_id=source_node.id,
                target_id=target_node.id,
                similarity=similarity,
                relationship_type="cross_domain_fix" if success else "cross_domain_attempt",
                domains=[source_domain, target_domain]
            )
            self._graph.edges.append(edge)

        self._graph.last_updated = datetime.utcnow()

    def boost_similarity(self, pattern_type: str, increment: float = 0.05):
        """Boost similarity weight for edges connected to a pattern node."""
        for node in self._graph.nodes:
            if node.pattern_type.value == pattern_type:
                node.confidence = min(1.0, node.confidence + increment)
                for edge in self._graph.edges:
                    if edge.source_id == node.id or edge.target_id == node.id:
                        edge.similarity = min(1.0, edge.similarity + increment)
                break

    def _find_or_create_node(self, pattern: AbstractPattern) -> Optional[KnowledgeNode]:
        for node in self._graph.nodes:
            if node.pattern_type == pattern.pattern_type:
                return node

        self._node_id_counter += 1
        new_node = KnowledgeNode(
            id=f"knode-{self._node_id_counter:04d}",
            pattern_type=pattern.pattern_type,
            domains=[pattern.domain],
            representative_keywords=pattern.keywords[:5],
            occurrences=1,
            confidence=pattern.confidence
        )
        self._graph.nodes.append(new_node)
        return new_node

    def _find_node_by_pattern(self, pattern: AbstractPattern) -> Optional[KnowledgeNode]:
        for node in self._graph.nodes:
            if node.pattern_type == pattern.pattern_type:
                return node
        return None

    def _update_node(self, node: KnowledgeNode, pattern: AbstractPattern):
        node.occurrences += 1

        if pattern.domain not in node.domains:
            node.domains.append(pattern.domain)

        for kw in pattern.keywords[:5]:
            if kw not in node.representative_keywords:
                node.representative_keywords.append(kw)

        node.representative_keywords = node.representative_keywords[:5]

        node.confidence = (node.confidence * (node.occurrences - 1) + pattern.confidence) / node.occurrences

        node.updated_at = datetime.utcnow()

    def _edge_exists(self, source_id: str, target_id: str) -> bool:
        for edge in self._graph.edges:
            if (edge.source_id == source_id and edge.target_id == target_id) or \
               (edge.source_id == target_id and edge.target_id == source_id):
                return True
        return False