"""
Federated Graph Learning (FedGNN)

This module implements federated learning for graph neural networks,
supporting both horizontal (inter-graph) and vertical (intra-graph) federated settings.

Key Components:
- graph_data_utils: Graph data generation and partitioning
- graph_models: GNN models (GCN, GraphSAGE, GIN)
- privacy_preserving_exchange: Privacy-preserving embedding exchange
- graph_client: Federated graph learning client
- graph_server: Federated graph learning server
- fed_graph_coordinator: Main coordinator for FedGNN
"""

from .graph_data_utils import (
    GraphDataGenerator,
    HorizontalGraphPartitioner,
    VerticalGraphPartitioner,
    GraphDataset
)

from .graph_models import (
    GCNLayer,
    GCNModel,
    GraphSAGELayer,
    GraphSAGEModel,
    GINLayer,
    GINModel
)

from .privacy_preserving_exchange import (
    DifferentialPrivacyMechanism,
    SecureEmbeddingExchange,
    TopKNeighborSelector
)

from .graph_client import (
    HorizontalFedGraphClient,
    VerticalFedGraphClient
)

from .graph_server import (
    FedGraphServer,
    CrossEdgeRouter
)

from .fed_graph_coordinator import (
    FedGraphCoordinator,
    FedGraphConfig
)

__all__ = [
    'GraphDataGenerator',
    'HorizontalGraphPartitioner',
    'VerticalGraphPartitioner',
    'GraphDataset',
    'GCNLayer',
    'GCNModel',
    'GraphSAGELayer',
    'GraphSAGEModel',
    'GINLayer',
    'GINModel',
    'DifferentialPrivacyMechanism',
    'SecureEmbeddingExchange',
    'TopKNeighborSelector',
    'HorizontalFedGraphClient',
    'VerticalFedGraphClient',
    'FedGraphServer',
    'CrossEdgeRouter',
    'FedGraphCoordinator',
    'FedGraphConfig'
]
