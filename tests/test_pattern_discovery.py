"""模式发现引擎（``PatternDiscoveryEngine``）的单元测试。

覆盖三件事：空图 / 数据不足时不崩且返回空、已知输入下的聚类数与名称、以及名字与置信度
确实是算出来的（不是硬编码常量）。
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hermes.cross_domain.abstract_pattern_extractor import AbstractPatternType  # noqa: E402
from hermes.cross_domain.knowledge_amalgamator import (  # noqa: E402
    KnowledgeAmalgamator,
    KnowledgeEdge,
    KnowledgeNode,
)
from hermes.cross_domain.pattern_discovery import PatternDiscoveryEngine  # noqa: E402


def _node(node_id, keywords, domain="python", confidence=0.5, occurrences=1, pattern_type=None):
    return KnowledgeNode(
        id=node_id,
        pattern_type=pattern_type or AbstractPatternType.DIVISION_BY_ZERO,
        domains=[domain],
        representative_keywords=list(keywords),
        occurrences=occurrences,
        confidence=confidence,
    )


def _engine(nodes, edges=()):
    amalgamator = KnowledgeAmalgamator()
    graph = amalgamator.get_graph()
    graph.nodes.extend(nodes)
    graph.edges.extend(edges)
    return PatternDiscoveryEngine(amalgamator)


class TestEmptyAndInsufficientInput:
    def test_empty_graph_discovers_nothing(self):
        assert _engine([]).discover_new_patterns() == []

    def test_single_node_is_below_the_minimum_cluster_size(self):
        engine = _engine([_node("a", ["timeout", "retry"])])

        assert engine.discover_new_patterns() == []

    def test_three_node_minimum_filters_out_a_two_node_cluster(self):
        nodes = [_node("a", ["timeout", "retry", "backoff"]), _node("b", ["timeout", "retry", "circuit"])]

        assert _engine(nodes).discover_new_patterns(min_cluster_size=3) == []

    def test_dissimilar_nodes_do_not_get_paired_up(self):
        nodes = [_node("a", ["timeout", "retry"]), _node("b", ["kubernetes", "pod"])]

        assert _engine(nodes).discover_new_patterns() == []


class TestKnownInputsProduceExpectedClusters:
    def test_two_similar_nodes_form_one_cluster_of_both(self):
        nodes = [_node("a", ["timeout", "retry", "backoff"]), _node("b", ["timeout", "retry", "circuit"])]

        discovered = _engine(nodes).discover_new_patterns()

        assert len(discovered) == 1
        assert sorted(discovered[0].source_nodes) == ["a", "b"]

    def test_cluster_spanning_two_domains_lists_both(self):
        nodes = [
            _node("a", ["timeout", "retry", "backoff"], domain="python"),
            _node("b", ["timeout", "retry", "circuit"], domain="k8s"),
        ]

        discovered = _engine(nodes).discover_new_patterns()

        assert discovered[0].domains == ["k8s", "python"]

    def test_edges_alone_can_join_two_nodes_with_different_keywords(self):
        nodes = [
            _node("a", ["timeout", "retry", "backoff"]),
            _node("b", ["kubernetes", "pod", "scheduling"]),
        ]
        edges = [KnowledgeEdge(source_id="a", target_id="b", similarity=0.9, domains=["python", "k8s"])]

        discovered = _engine(nodes, edges).discover_new_patterns()

        assert len(discovered) == 1
        assert sorted(discovered[0].source_nodes) == ["a", "b"]

    def test_cluster_matching_a_known_pattern_type_is_not_reported(self):
        nodes = [
            _node("a", ["boundary", "check", "missing"]),
            _node("b", ["boundary", "check", "missing"]),
        ]

        assert _engine(nodes).discover_new_patterns() == []


class TestFieldsAreDerivedNotHardcoded:
    def test_name_is_built_from_the_cluster_keywords(self):
        nodes = [
            _node("a", ["timeout", "backoff", "jitter"]),
            _node("b", ["timeout", "backoff", "ceiling"]),
        ]

        discovered = _engine(nodes).discover_new_patterns()

        assert discovered[0].name == "timeout_backoff"

    def test_two_different_clusters_get_two_different_names(self):
        nodes = [
            _node("a", ["timeout", "backoff", "jitter"]),
            _node("b", ["timeout", "backoff", "ceiling"]),
            _node("c", ["quota", "exceeded", "throttle"]),
            _node("d", ["quota", "exceeded", "burst"]),
        ]

        names = [pattern.name for pattern in _engine(nodes).discover_new_patterns()]

        assert len(names) == 2
        assert len(set(names)) == 2

    def test_confidence_is_the_mean_of_the_member_nodes(self):
        nodes = [
            _node("a", ["timeout", "retry", "backoff"], confidence=0.4),
            _node("b", ["timeout", "retry", "circuit"], confidence=0.8),
        ]

        discovered = _engine(nodes).discover_new_patterns()

        assert discovered[0].confidence == pytest.approx(0.6)
        assert discovered[0].to_dict()["confidence"] == pytest.approx(0.6)

    def test_patterns_are_sorted_by_confidence_descending(self):
        nodes = [
            _node("low_a", ["timeout", "retry", "backoff"], confidence=0.2),
            _node("low_b", ["timeout", "retry", "circuit"], confidence=0.3),
            _node("high_a", ["quota", "exceeded", "throttle"], confidence=0.9),
            _node("high_b", ["quota", "exceeded", "burst"], confidence=0.95),
        ]

        confidences = [pattern.confidence for pattern in _engine(nodes).discover_new_patterns()]

        assert confidences == sorted(confidences, reverse=True)
        assert confidences[0] > confidences[-1]
