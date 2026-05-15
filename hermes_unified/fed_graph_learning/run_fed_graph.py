"""
Run Federated Graph Learning

Main entry point for running federated graph learning experiments.
Supports both horizontal (inter-graph) and vertical (intra-graph) modes.
"""

import argparse
import logging
import json
import time
from typing import Dict, Any, Optional

import numpy as np

from fed_graph_coordinator import (
    FedGraphCoordinator,
    FedGraphConfig,
    run_fed_graph_demo
)
from graph_data_utils import (
    GraphDataGenerator,
    HorizontalGraphPartitioner,
    VerticalGraphPartitioner,
    GraphDataset,
    create_synthetic_graph_dataset
)
from graph_models import create_gnn_model, GCNModel, GraphSAGEModel, GINModel
from privacy_preserving_exchange import (
    DifferentialPrivacyMechanism,
    SecureEmbeddingExchange,
    TopKNeighborSelector,
    PrivacyPreservingExchange,
    ExchangeConfig
)
from graph_client import HorizontalFedGraphClient, VerticalFedGraphClient, ClientConfig
from graph_server import FedGraphServer, ServerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_horizontal_experiment(
    num_clients: int = 4,
    num_rounds: int = 20,
    num_graphs: int = 40,
    model_type: str = 'gin',
    use_dp: bool = False,
    epsilon: float = 1.0,
    non_iid: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run horizontal federated graph learning experiment.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        num_graphs: Total number of graphs
        model_type: GNN model type
        use_dp: Use differential privacy
        epsilon: DP epsilon
        non_iid: Create non-IID data distribution
        seed: Random seed
    
    Returns:
        Experiment results
    """
    print("\n" + "=" * 70)
    print("  HORIZONTAL FEDERATED GRAPH LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Graphs: {num_graphs}")
    print(f"  Model: {model_type.upper()}")
    print(f"  DP: {'Enabled (ε={})'.format(epsilon) if use_dp else 'Disabled'}")
    print(f"  Non-IID: {non_iid}")
    print("=" * 70)
    
    config = FedGraphConfig(
        mode='horizontal',
        num_clients=num_clients,
        num_rounds=num_rounds,
        model_type=model_type,
        hidden_dims=[64, 32],
        input_dim=16,
        output_dim=3,
        local_epochs=3,
        use_dp=use_dp,
        epsilon=epsilon,
        seed=seed
    )
    
    coordinator = FedGraphCoordinator(config)
    
    print("\n[1/3] Setting up horizontal partitioning...")
    client_data = coordinator.setup_horizontal(
        num_graphs=num_graphs,
        min_nodes=20,
        max_nodes=50,
        non_iid=non_iid
    )
    
    print("\n[2/3] Training...")
    start_time = time.time()
    results = coordinator.train(verbose=True)
    training_time = time.time() - start_time
    
    print("\n[3/3] Evaluating...")
    test_graphs = create_synthetic_graph_dataset(
        dataset_type='random',
        num_nodes=30,
        num_graphs=10,
        feature_dim=16,
        num_classes=3,
        seed=seed + 1000
    )
    eval_results = coordinator.evaluate(test_graphs)
    
    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"  Training Loss: {results['final_loss']:.4f}")
    print(f"  Training Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Test Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Avg Round Time: {results['avg_round_time']:.2f}s")
    
    comm = coordinator.get_communication_cost()
    print(f"  Communication: {comm['total_bytes']:,} bytes")
    print("-" * 70)
    
    return {
        'mode': 'horizontal',
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'num_graphs': num_graphs,
            'model_type': model_type,
            'use_dp': use_dp,
            'epsilon': epsilon
        },
        'results': results,
        'evaluation': eval_results,
        'communication_cost': comm,
        'training_time': training_time
    }


def run_vertical_experiment(
    num_clients: int = 3,
    num_rounds: int = 20,
    num_nodes: int = 200,
    model_type: str = 'gcn',
    use_dp: bool = True,
    epsilon: float = 1.0,
    use_top_k: bool = True,
    top_k: int = 10,
    partition_strategy: str = 'random',
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run vertical federated graph learning experiment.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        num_nodes: Number of nodes in the graph
        model_type: GNN model type
        use_dp: Use differential privacy
        epsilon: DP epsilon
        use_top_k: Use top-k neighbor selection
        top_k: Number of top neighbors
        partition_strategy: Graph partitioning strategy
        seed: Random seed
    
    Returns:
        Experiment results
    """
    print("\n" + "=" * 70)
    print("  VERTICAL FEDERATED GRAPH LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Model: {model_type.upper()}")
    print(f"  DP: {'Enabled (ε={})'.format(epsilon) if use_dp else 'Disabled'}")
    print(f"  Top-K: {top_k if use_top_k else 'Disabled'}")
    print(f"  Partition: {partition_strategy}")
    print("=" * 70)
    
    config = FedGraphConfig(
        mode='vertical',
        num_clients=num_clients,
        num_rounds=num_rounds,
        model_type=model_type,
        hidden_dims=[64, 32],
        input_dim=16,
        output_dim=3,
        local_epochs=3,
        use_dp=use_dp,
        epsilon=epsilon,
        use_top_k=use_top_k,
        top_k=top_k,
        seed=seed
    )
    
    coordinator = FedGraphCoordinator(config)
    
    print("\n[1/3] Setting up vertical partitioning...")
    client_graphs, partition_info = coordinator.setup_vertical(
        num_nodes=num_nodes,
        partition_strategy=partition_strategy
    )
    
    cross_edge_stats = coordinator.cross_edge_info
    print(f"  Cross-edges: {cross_edge_stats.get('total_cross_edges', 0)}")
    
    print("\n[2/3] Training...")
    start_time = time.time()
    results = coordinator.train(verbose=True)
    training_time = time.time() - start_time
    
    print("\n[3/3] Evaluating...")
    eval_results = coordinator.evaluate()
    
    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"  Training Loss: {results['final_loss']:.4f}")
    print(f"  Training Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Test Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Avg Round Time: {results['avg_round_time']:.2f}s")
    
    comm = coordinator.get_communication_cost()
    print(f"  Communication: {comm['total_bytes']:,} bytes")
    print(f"  Cross-edge exchanges: {cross_edge_stats.get('total_cross_edges', 0) * num_rounds}")
    print("-" * 70)
    
    return {
        'mode': 'vertical',
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'num_nodes': num_nodes,
            'model_type': model_type,
            'use_dp': use_dp,
            'epsilon': epsilon,
            'use_top_k': use_top_k,
            'top_k': top_k
        },
        'results': results,
        'evaluation': eval_results,
        'communication_cost': comm,
        'cross_edge_stats': cross_edge_stats,
        'training_time': training_time
    }


def run_comparison_experiment(
    num_clients: int = 4,
    num_rounds: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run comparison between horizontal and vertical modes.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        seed: Random seed
    
    Returns:
        Comparison results
    """
    print("\n" + "=" * 70)
    print("  FEDERATED GRAPH LEARNING COMPARISON")
    print("=" * 70)
    
    h_results = run_horizontal_experiment(
        num_clients=num_clients,
        num_rounds=num_rounds,
        num_graphs=30,
        model_type='gin',
        use_dp=False,
        seed=seed
    )
    
    v_results = run_vertical_experiment(
        num_clients=num_clients,
        num_rounds=num_rounds,
        num_nodes=150,
        model_type='gcn',
        use_dp=True,
        epsilon=1.0,
        seed=seed
    )
    
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Horizontal':<20} {'Vertical':<20}")
    print("-" * 70)
    print(f"  {'Training Accuracy':<25} {h_results['results']['final_accuracy']:<20.4f} {v_results['results']['final_accuracy']:<20.4f}")
    print(f"  {'Training Loss':<25} {h_results['results']['final_loss']:<20.4f} {v_results['results']['final_loss']:<20.4f}")
    print(f"  {'Training Time (s)':<25} {h_results['training_time']:<20.2f} {v_results['training_time']:<20.2f}")
    print(f"  {'Communication (KB)':<25} {h_results['communication_cost']['total_bytes']/1024:<20.2f} {v_results['communication_cost']['total_bytes']/1024:<20.2f}")
    print("=" * 70)
    
    return {
        'horizontal': h_results,
        'vertical': v_results
    }


def run_privacy_ablation(
    epsilon_values: list = [0.1, 0.5, 1.0, 2.0, 5.0],
    num_rounds: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run privacy budget ablation study.
    
    Args:
        epsilon_values: List of epsilon values to test
        num_rounds: Number of training rounds
        seed: Random seed
    
    Returns:
        Ablation results
    """
    print("\n" + "=" * 70)
    print("  PRIVACY BUDGET ABLATION STUDY")
    print("=" * 70)
    print(f"  Testing epsilon values: {epsilon_values}")
    print("=" * 70)
    
    results = {}
    
    for epsilon in epsilon_values:
        print(f"\n--- Testing ε = {epsilon} ---")
        
        v_results = run_vertical_experiment(
            num_clients=3,
            num_rounds=num_rounds,
            num_nodes=100,
            model_type='gcn',
            use_dp=True,
            epsilon=epsilon,
            seed=seed
        )
        
        results[epsilon] = {
            'accuracy': v_results['results']['final_accuracy'],
            'loss': v_results['results']['final_loss']
        }
    
    print("\n" + "=" * 70)
    print("  PRIVACY-UTILITY TRADEOFF")
    print("=" * 70)
    print(f"  {'Epsilon':<15} {'Accuracy':<15} {'Loss':<15}")
    print("-" * 70)
    for eps, res in results.items():
        print(f"  {eps:<15.1f} {res['accuracy']:<15.4f} {res['loss']:<15.4f}")
    print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Federated Graph Learning Experiments'
    )
    
    parser.add_argument(
        '--mode', type=str, default='horizontal',
        choices=['horizontal', 'vertical', 'comparison', 'ablation'],
        help='Experiment mode'
    )
    parser.add_argument(
        '--num-clients', type=int, default=4,
        help='Number of clients'
    )
    parser.add_argument(
        '--num-rounds', type=int, default=20,
        help='Number of training rounds'
    )
    parser.add_argument(
        '--model', type=str, default='gcn',
        choices=['gcn', 'graphsage', 'gin'],
        help='GNN model type'
    )
    parser.add_argument(
        '--use-dp', action='store_true',
        help='Use differential privacy'
    )
    parser.add_argument(
        '--epsilon', type=float, default=1.0,
        help='DP epsilon value'
    )
    parser.add_argument(
        '--num-graphs', type=int, default=40,
        help='Number of graphs (horizontal mode)'
    )
    parser.add_argument(
        '--num-nodes', type=int, default=200,
        help='Number of nodes (vertical mode)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    results = None
    
    if args.mode == 'horizontal':
        results = run_horizontal_experiment(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            num_graphs=args.num_graphs,
            model_type=args.model,
            use_dp=args.use_dp,
            epsilon=args.epsilon,
            seed=args.seed
        )
    
    elif args.mode == 'vertical':
        results = run_vertical_experiment(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            num_nodes=args.num_nodes,
            model_type=args.model,
            use_dp=args.use_dp,
            epsilon=args.epsilon,
            seed=args.seed
        )
    
    elif args.mode == 'comparison':
        results = run_comparison_experiment(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            seed=args.seed
        )
    
    elif args.mode == 'ablation':
        results = run_privacy_ablation(
            num_rounds=args.num_rounds,
            seed=args.seed
        )
    
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
