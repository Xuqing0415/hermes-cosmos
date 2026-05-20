"""
Topology-Aware Federated Learning Module

Implements federated learning that adapts to network topology, including:
- Topology discovery and modeling
- Hierarchical aggregation
- Topology-aware client selection
- Adaptive communication compression
"""

from .topology_discovery import (
    NetworkTopology,
    TopologyDiscovery,
    LinkQualityMonitor
)

from .hierarchical_aggregator import (
    HierarchicalAggregator,
    TreeAggregationNode,
    AggregationTree
)

from .topology_scheduler import (
    TopologyScheduler,
    GraphBasedSelector,
    SteinerTreeOptimizer
)

from .topology_aware_fl import (
    TopologyAwareFL,
    run_topology_demo
)

__all__ = [
    'NetworkTopology',
    'TopologyDiscovery',
    'LinkQualityMonitor',
    'HierarchicalAggregator',
    'TreeAggregationNode',
    'AggregationTree',
    'TopologyScheduler',
    'GraphBasedSelector',
    'SteinerTreeOptimizer',
    'TopologyAwareFL',
    'run_topology_demo'
]