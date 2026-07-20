"""
Pattern Discovery Module
Analyzes the knowledge graph's fix cases and uses clustering algorithms to
automatically discover new pattern types from cross-domain fix data.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import math
from collections import Counter

from .abstract_pattern_extractor import AbstractPattern, AbstractPatternType
from .knowledge_amalgamator import KnowledgeAmalgamator, KnowledgeNode, KnowledgeEdge


@dataclass
class DiscoveredPattern:
    """A new pattern type discovered by clustering."""
    name: str
    description: str
    source_nodes: List[str]
    domains: List[str]
    representative_keywords: List[str]
    confidence: float
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source_nodes": self.source_nodes,
            "domains": self.domains,
            "representative_keywords": self.representative_keywords,
            "confidence": round(self.confidence, 2),
            "similarity_score": round(self.similarity_score, 2),
        }


class PatternClusterNode:
    """Internal node for clustering."""
    def __init__(self, node: KnowledgeNode, keywords: List[str], domain: str):
        self.node_id = node.id
        self.pattern_type = node.pattern_type.value
        self.keywords = keywords
        self.domain = domain
        self.occurrences = node.occurrences
        self.confidence = node.confidence


class PatternDiscoveryEngine:
    """
    Analyzes knowledge graph nodes and cross-domain edges to discover
    new pattern types via keyword clustering.
    """

    def __init__(self, amalgamator: KnowledgeAmalgamator):
        self._amalgamator = amalgamator
        self._known_types: set = set()

    def discover_new_patterns(self, min_cluster_size: int = 2,
                              similarity_threshold: float = 0.35) -> List[DiscoveredPattern]:
        """
        Run pattern discovery on the knowledge graph.
        Returns newly discovered patterns not yet in AbstractPatternType.
        """
        self._known_types = set(pt.value for pt in AbstractPatternType)
        graph = self._amalgamator.get_graph()

        if len(graph.nodes) < min_cluster_size:
            return []

        # Build TF-like vectors from node keywords
        clusters = self._cluster_nodes(graph.nodes, graph.edges, similarity_threshold)

        # Filter out clusters that match existing AbstractPatternType
        discovered = []
        for cluster_nodes, keywords, domains in clusters:
            if len(cluster_nodes) < min_cluster_size:
                continue

            # Check if this cluster already maps to a known type
            known_match = self._match_known_type(keywords, cluster_nodes)
            if known_match:
                continue

            # Generate a name from the top keywords
            name = self._generate_pattern_name(keywords, domains)
            description = self._generate_description(name, keywords, domains)

            discovered.append(DiscoveredPattern(
                name=name,
                description=description,
                source_nodes=[n.node_id for n in cluster_nodes],
                domains=list(domains),
                representative_keywords=keywords[:5],
                confidence=sum(n.confidence for n in cluster_nodes) / len(cluster_nodes),
                similarity_score=self._calculate_cluster_cohesion(cluster_nodes),
            ))

        return sorted(discovered, key=lambda p: p.confidence, reverse=True)

    def _cluster_nodes(self, nodes: List[KnowledgeNode], edges: List[KnowledgeEdge],
                       threshold: float) -> List[Tuple[List[PatternClusterNode], List[str], set]]:
        """Cluster knowledge graph nodes using keyword Jaccard similarity."""
        cluster_nodes = [
            PatternClusterNode(n, n.representative_keywords, n.domains[0] if n.domains else "unknown")
            for n in nodes
        ]

        if not cluster_nodes:
            return []

        # Adjacency list based on keyword Jaccard similarity
        adj = {i: set() for i in range(len(cluster_nodes))}
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                sim = self._keyword_jaccard(cluster_nodes[i].keywords, cluster_nodes[j].keywords)
                # Also check if there's an edge between the original nodes
                edge_sim = self._get_edge_similarity(
                    cluster_nodes[i].node_id, cluster_nodes[j].node_id, edges
                )
                combined = max(sim, edge_sim)
                if combined >= threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        # Label propagation (simple connected components)
        visited = set()
        clusters = []
        for i in range(len(cluster_nodes)):
            if i in visited:
                continue
            component = set()
            stack = [i]
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                visited.add(idx)
                component.add(idx)
                for neighbor in adj[idx]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if component:
                component_nodes = [cluster_nodes[idx] for idx in component]
                all_keywords = []
                all_domains = set()
                for cn in component_nodes:
                    all_keywords.extend(cn.keywords)
                    all_domains.add(cn.domain)
                # Count keyword frequency
                kw_counter = Counter(all_keywords)
                top_keywords = [kw for kw, _ in kw_counter.most_common(10)]
                clusters.append((component_nodes, top_keywords, all_domains))

        return clusters

    def _keyword_jaccard(self, kw_a: List[str], kw_b: List[str]) -> float:
        if not kw_a or not kw_b:
            return 0.0
        set_a = set(kw_a)
        set_b = set(kw_b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    def _get_edge_similarity(self, node_a_id: str, node_b_id: str,
                             edges: List[KnowledgeEdge]) -> float:
        for e in edges:
            if {e.source_id, e.target_id} == {node_a_id, node_b_id}:
                return e.similarity
        return 0.0

    def _match_known_type(self, keywords: List[str],
                          cluster_nodes: List[PatternClusterNode]) -> Optional[str]:
        """Check if a cluster's keywords match an existing AbstractPatternType."""
        kw_set = set(k.lower() for k in keywords)
        for pt in AbstractPatternType:
            # Extract characteristic words from the enum value
            type_words = set(pt.value.replace("_", " ").split())
            overlap = kw_set & type_words
            if len(overlap) >= 2 or (len(overlap) >= 1 and len(kw_set) <= 3):
                return pt.value
        return None

    def _generate_pattern_name(self, keywords: List[str], domains: set) -> str:
        """Generate a descriptive pattern name from keywords and domains."""
        # Use the most distinctive keywords to form a name
        stop_words = {"missing", "check", "error", "invalid", "failed", "not"}
        meaningful = [kw for kw in keywords if kw.lower() not in stop_words]

        if not meaningful:
            meaningful = keywords[:3]

        if len(meaningful) >= 2:
            name = "_".join(meaningful[:2]).lower()
        else:
            name = meaningful[0].lower()

        # Ensure name is a valid identifier suffix
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        if not name:
            name = "unknown_pattern"

        return name

    def _generate_description(self, name: str, keywords: List[str], domains: set) -> str:
        """Generate a human-readable description of the discovered pattern."""
        domain_list = ", ".join(sorted(domains))
        kw_list = ", ".join(keywords[:5])
        readable_name = name.replace("_", " ").title()
        return (f"{readable_name}: detected across [{domain_list}] "
                f"with keywords [{kw_list}]")

    def _calculate_cluster_cohesion(self, cluster_nodes: List[PatternClusterNode]) -> float:
        """Calculate the average pairwise similarity within a cluster."""
        if len(cluster_nodes) < 2:
            return 1.0
        total = 0.0
        count = 0
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                total += self._keyword_jaccard(
                    cluster_nodes[i].keywords, cluster_nodes[j].keywords
                )
                count += 1
        return total / count if count > 0 else 0.0