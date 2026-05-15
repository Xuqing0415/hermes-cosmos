"""
Test script for Federated Graph Learning module.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Testing Federated Graph Learning Module")
print("=" * 60)

print("\n1. Testing imports...")
try:
    from hermes_unified.fed_graph_learning.graph_data_utils import (
        GraphDataGenerator,
        HorizontalGraphPartitioner,
        VerticalGraphPartitioner,
        GraphDataset
    )
    print("   graph_data_utils: OK")
except Exception as e:
    print(f"   graph_data_utils: FAILED - {e}")
    sys.exit(1)

try:
    from hermes_unified.fed_graph_learning.graph_models import (
        GCNModel,
        GraphSAGEModel,
        GINModel,
        create_gnn_model
    )
    print("   graph_models: OK")
except Exception as e:
    print(f"   graph_models: FAILED - {e}")
    sys.exit(1)

try:
    from hermes_unified.fed_graph_learning.privacy_preserving_exchange import (
        DifferentialPrivacyMechanism,
        SecureEmbeddingExchange,
        TopKNeighborSelector
    )
    print("   privacy_preserving_exchange: OK")
except Exception as e:
    print(f"   privacy_preserving_exchange: FAILED - {e}")
    sys.exit(1)

try:
    from hermes_unified.fed_graph_learning.graph_client import (
        HorizontalFedGraphClient,
        VerticalFedGraphClient,
        ClientConfig
    )
    print("   graph_client: OK")
except Exception as e:
    print(f"   graph_client: FAILED - {e}")
    sys.exit(1)

try:
    from hermes_unified.fed_graph_learning.graph_server import (
        FedGraphServer,
        ServerConfig,
        CrossEdgeRouter
    )
    print("   graph_server: OK")
except Exception as e:
    print(f"   graph_server: FAILED - {e}")
    sys.exit(1)

try:
    from hermes_unified.fed_graph_learning.fed_graph_coordinator import (
        FedGraphCoordinator,
        FedGraphConfig
    )
    print("   fed_graph_coordinator: OK")
except Exception as e:
    print(f"   fed_graph_coordinator: FAILED - {e}")
    sys.exit(1)

print("\n2. Testing graph data generation...")
import numpy as np

generator = GraphDataGenerator(seed=42)
graph = generator.generate_erdos_renyi(
    num_nodes=50,
    edge_prob=0.1,
    feature_dim=16,
    num_classes=3
)
print(f"   Generated graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

print("\n3. Testing GNN model...")
model = create_gnn_model(
    model_type='gcn',
    input_dim=16,
    hidden_dims=[32],
    output_dim=3,
    task='node_classification'
)
output = model.forward(graph.node_features, graph.adj_matrix, training=False)
print(f"   Model output shape: {output.shape}")

print("\n4. Testing horizontal federated learning...")
from hermes_unified.fed_graph_learning.fed_graph_coordinator import FedGraphConfig, FedGraphCoordinator

config = FedGraphConfig(
    mode='horizontal',
    num_clients=3,
    num_rounds=3,
    model_type='gin',
    hidden_dims=[32, 16],
    input_dim=16,
    output_dim=3,
    local_epochs=1,
    seed=42
)

coordinator = FedGraphCoordinator(config)
coordinator.setup_horizontal(num_graphs=9, min_nodes=15, max_nodes=25)
results = coordinator.train(verbose=False)

print(f"   Final accuracy: {results['final_accuracy']:.4f}")
print(f"   Final loss: {results['final_loss']:.4f}")

print("\n5. Testing vertical federated learning...")
config_v = FedGraphConfig(
    mode='vertical',
    num_clients=3,
    num_rounds=3,
    model_type='gcn',
    hidden_dims=[32],
    input_dim=16,
    output_dim=3,
    local_epochs=1,
    use_dp=True,
    epsilon=1.0,
    seed=42
)

coordinator_v = FedGraphCoordinator(config_v)
coordinator_v.setup_vertical(num_nodes=80)
results_v = coordinator_v.train(verbose=False)

print(f"   Final accuracy: {results_v['final_accuracy']:.4f}")
print(f"   Final loss: {results_v['final_loss']:.4f}")
print(f"   Cross-edges: {coordinator_v.cross_edge_info.get('total_cross_edges', 0)}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
