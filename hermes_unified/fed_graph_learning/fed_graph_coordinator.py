"""
Federated Graph Learning Coordinator

Main coordinator that integrates all components for federated graph learning,
supporting both horizontal (inter-graph) and vertical (intra-graph) modes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import time
from enum import Enum

from .graph_data_utils import (
    GraphDataGenerator,
    HorizontalGraphPartitioner,
    VerticalGraphPartitioner,
    GraphDataset
)
from .graph_models import create_gnn_model
from .privacy_preserving_exchange import ExchangeConfig
from .graph_client import (
    HorizontalFedGraphClient,
    VerticalFedGraphClient,
    ClientConfig,
    FedGraphClientFactory
)
from .graph_server import FedGraphServer, ServerConfig, CrossEdgeRouter

logger = logging.getLogger(__name__)


class FedGraphMode(Enum):
    """Federated graph learning modes."""
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


@dataclass
class FedGraphConfig:
    """
    Configuration for federated graph learning.
    
    Attributes:
        mode: 'horizontal' or 'vertical'
        num_clients: Number of clients
        num_rounds: Number of federated rounds
        model_type: 'gcn', 'graphsage', or 'gin'
        hidden_dims: Hidden layer dimensions
        input_dim: Input feature dimension
        output_dim: Number of output classes
        local_epochs: Local training epochs
        learning_rate: Learning rate
        aggregation_method: Aggregation method
        use_dp: Use differential privacy
        epsilon: DP epsilon
        use_top_k: Use top-k neighbor selection
        top_k: Number of top neighbors
        seed: Random seed
    """
    mode: str = 'horizontal'
    num_clients: int = 4
    num_rounds: int = 10
    model_type: str = 'gcn'
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    input_dim: int = 16
    output_dim: int = 3
    local_epochs: int = 5
    learning_rate: float = 0.01
    aggregation_method: str = 'fedavg'
    use_dp: bool = False
    epsilon: float = 1.0
    use_top_k: bool = True
    top_k: int = 10
    seed: int = 42
    
    def __post_init__(self):
        self.mode = FedGraphMode(self.mode.lower()).value


class FedGraphCoordinator:
    """
    Main coordinator for federated graph learning.
    
    Orchestrates:
    - Data partitioning (horizontal/vertical)
    - Client creation and management
    - Server coordination
    - Training rounds
    - Cross-edge embedding exchange (vertical)
    - Evaluation and metrics
    """
    
    def __init__(self, config: FedGraphConfig):
        """
        Initialize federated graph learning coordinator.
        
        Args:
            config: Configuration for FedGNN
        """
        self.config = config
        self.mode = config.mode
        
        self.rng = np.random.RandomState(config.seed)
        
        self.server: Optional[FedGraphServer] = None
        self.clients: Dict[int, Union[HorizontalFedGraphClient, VerticalFedGraphClient]] = {}
        
        self.graph_generator = GraphDataGenerator(seed=config.seed)
        self.h_partitioner = HorizontalGraphPartitioner(seed=config.seed)
        self.v_partitioner = VerticalGraphPartitioner(seed=config.seed)
        
        self.partition_info: Dict[str, Any] = {}
        self.cross_edge_info: Dict[str, Any] = {}
        
        self.training_history: List[Dict] = []
        self.metrics: Dict[str, List] = {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'round_time': [],
            'communication_cost': []
        }
        
        self.current_round = 0
    
    def setup_horizontal(
        self,
        num_graphs: int = 20,
        min_nodes: int = 20,
        max_nodes: int = 50,
        non_iid: bool = True
    ) -> Dict[int, List[GraphDataset]]:
        """
        Setup horizontal federated graph learning.
        
        Args:
            num_graphs: Total number of graphs
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            non_iid: Whether to create non-IID distribution
        
        Returns:
            Dictionary mapping client_id to list of graphs
        """
        logger.info("Setting up horizontal federated graph learning...")
        
        client_data = self.h_partitioner.partition(
            num_graphs=num_graphs,
            num_clients=self.config.num_clients,
            graph_generator=self.graph_generator,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            feature_dim=self.config.input_dim,
            num_classes=self.config.output_dim,
            non_iid=non_iid
        )
        
        server_config = ServerConfig(
            model_type=self.config.model_type,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            aggregation_method=self.config.aggregation_method,
            seed=self.config.seed
        )
        
        self.server = FedGraphServer(server_config, self.config.input_dim)
        
        for client_id in range(self.config.num_clients):
            client_config = ClientConfig(
                client_id=client_id,
                model_type=self.config.model_type,
                hidden_dims=self.config.hidden_dims,
                learning_rate=self.config.learning_rate,
                local_epochs=self.config.local_epochs,
                use_dp=self.config.use_dp,
                epsilon=self.config.epsilon,
                seed=self.config.seed + client_id
            )
            
            client = HorizontalFedGraphClient(
                config=client_config,
                input_dim=self.config.input_dim,
                output_dim=self.config.output_dim
            )
            
            if client_id in client_data:
                client.set_graphs(client_data[client_id])
            
            self.clients[client_id] = client
            self.server.register_client(client_id)
        
        self.partition_info = {
            'mode': 'horizontal',
            'num_graphs': num_graphs,
            'graphs_per_client': {cid: len(graphs) for cid, graphs in client_data.items()}
        }
        
        logger.info(f"Setup complete: {len(self.clients)} clients, "
                   f"{sum(len(g) for g in client_data.values())} total graphs")
        
        return client_data
    
    def setup_vertical(
        self,
        graph: Optional[GraphDataset] = None,
        num_nodes: int = 100,
        partition_strategy: str = 'random'
    ) -> Tuple[Dict[int, GraphDataset], Dict[str, Any]]:
        """
        Setup vertical federated graph learning.
        
        Args:
            graph: Optional pre-generated graph (will be generated if None)
            num_nodes: Number of nodes if generating graph
            partition_strategy: 'random' or 'degree'
        
        Returns:
            Tuple of (client_graphs, partition_info)
        """
        logger.info("Setting up vertical federated graph learning...")
        
        if graph is None:
            graph = self.graph_generator.generate_sbm(
                num_nodes=num_nodes,
                num_communities=self.config.output_dim,
                p_in=0.3,
                p_out=0.05,
                feature_dim=self.config.input_dim
            )
        
        client_graphs, partition_info = self.v_partitioner.partition(
            graph=graph,
            num_clients=self.config.num_clients,
            partition_strategy=partition_strategy
        )
        
        server_config = ServerConfig(
            model_type=self.config.model_type,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            aggregation_method=self.config.aggregation_method,
            seed=self.config.seed
        )
        
        self.server = FedGraphServer(server_config, self.config.input_dim)
        
        exchange_config = ExchangeConfig(
            use_dp=self.config.use_dp,
            epsilon=self.config.epsilon,
            use_top_k=self.config.use_top_k,
            k=self.config.top_k
        )
        
        for client_id, client_graph in client_graphs.items():
            client_config = ClientConfig(
                client_id=client_id,
                model_type=self.config.model_type,
                hidden_dims=self.config.hidden_dims,
                learning_rate=self.config.learning_rate,
                local_epochs=self.config.local_epochs,
                use_dp=self.config.use_dp,
                epsilon=self.config.epsilon,
                seed=self.config.seed + client_id
            )
            
            client = VerticalFedGraphClient(
                config=client_config,
                input_dim=self.config.input_dim,
                output_dim=self.config.output_dim,
                exchange_config=exchange_config
            )
            
            cross_edges = partition_info['cross_edge_info'].get(client_id, {}).get('cross_edges', [])
            neighbor_clients = {}
            for src, dst, dst_client in cross_edges:
                if dst_client not in neighbor_clients:
                    neighbor_clients[dst_client] = []
                neighbor_clients[dst_client].append(src)
            
            client.set_subgraph(client_graph, cross_edges, neighbor_clients)
            
            self.clients[client_id] = client
            self.server.register_client(client_id, {'num_nodes': len(client_graph.node_ids or [])})
            self.server.register_cross_edges(client_id, cross_edges)
        
        self.partition_info = partition_info
        self.cross_edge_info = self.v_partitioner.get_cross_edge_statistics(
            partition_info['cross_edge_info']
        )
        
        logger.info(f"Setup complete: {len(self.clients)} clients, "
                   f"{self.cross_edge_info.get('total_cross_edges', 0)} cross-edges")
        
        return client_graphs, partition_info
    
    def run_round(self) -> Dict[str, Any]:
        """
        Run a single federated learning round.
        
        Returns:
            Round statistics
        """
        self.current_round += 1
        round_start = time.time()
        
        global_weights = self.server.get_global_weights()
        
        for client_id, client in self.clients.items():
            client.set_weights(global_weights.copy())
        
        client_updates = {}
        client_metrics = {}
        cross_edge_embeddings = {}
        
        if self.mode == 'horizontal':
            for client_id, client in self.clients.items():
                metrics = client.train_local()
                client_metrics[client_id] = metrics
                
                update = client.compute_update(global_weights)
                client_updates[client_id] = update
        
        else:
            for client_id, client in self.clients.items():
                if isinstance(client, VerticalFedGraphClient):
                    local_embeddings = client.local_message_passing(training=True)
                    embeddings_to_send = client.prepare_cross_edge_embeddings(local_embeddings)
                    cross_edge_embeddings[client_id] = embeddings_to_send
            
            for client_id, client in self.clients.items():
                if isinstance(client, VerticalFedGraphClient):
                    received = {}
                    for src_client, embeddings in cross_edge_embeddings.items():
                        if client_id in embeddings:
                            received[src_client] = embeddings[client_id]
                    
                    metrics = client.train_local(
                        cross_edge_embeddings=received if received else None
                    )
                    client_metrics[client_id] = metrics
                    
                    update = {}
                    local_weights = client.get_weights()
                    for key in global_weights:
                        if key in local_weights:
                            update[key] = local_weights[key] - global_weights[key]
                    client_updates[client_id] = update
        
        round_stats = self.server.run_round(client_updates, client_metrics)
        
        round_time = time.time() - round_start
        self.metrics['round_time'].append(round_time)
        
        avg_loss = np.mean([m.get('loss', 0) for m in client_metrics.values()])
        avg_acc = np.mean([m.get('accuracy', 0) for m in client_metrics.values()])
        
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['train_accuracy'].append(avg_acc)
        
        round_stats['round_time'] = round_time
        round_stats['avg_loss'] = avg_loss
        round_stats['avg_accuracy'] = avg_acc
        
        self.training_history.append(round_stats)
        
        logger.info(f"Round {self.current_round}: loss={avg_loss:.4f}, acc={avg_acc:.4f}, time={round_time:.2f}s")
        
        return round_stats
    
    def train(
        self,
        num_rounds: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run federated training.
        
        Args:
            num_rounds: Number of rounds (overrides config)
            verbose: Print progress
        
        Returns:
            Training results
        """
        rounds = num_rounds or self.config.num_rounds
        
        logger.info(f"Starting federated training for {rounds} rounds...")
        
        for r in range(rounds):
            round_stats = self.run_round()
            
            if verbose and (r + 1) % max(1, rounds // 5) == 0:
                print(f"Round {r + 1}/{rounds}: "
                      f"Loss={round_stats['avg_loss']:.4f}, "
                      f"Acc={round_stats['avg_accuracy']:.4f}")
        
        results = {
            'num_rounds': self.current_round,
            'final_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else 0,
            'final_accuracy': self.metrics['train_accuracy'][-1] if self.metrics['train_accuracy'] else 0,
            'total_time': sum(self.metrics['round_time']),
            'avg_round_time': np.mean(self.metrics['round_time']) if self.metrics['round_time'] else 0
        }
        
        logger.info(f"Training complete: {results}")
        
        return results
    
    def evaluate(
        self,
        test_data: Optional[List[GraphDataset]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the global model.
        
        Args:
            test_data: Optional test data
        
        Returns:
            Evaluation metrics
        """
        if self.server is None:
            return {'accuracy': 0.0, 'loss': 0.0}
        
        return self.server.evaluate_global_model(test_data)
    
    def get_client_stats(self) -> Dict[int, Dict]:
        """Get statistics for all clients."""
        return {cid: client.get_stats() for cid, client in self.clients.items()}
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        if self.server is None:
            return {}
        return self.server.get_stats()
    
    def get_communication_cost(self) -> Dict[str, float]:
        """Calculate communication cost."""
        if not self.clients:
            return {'total_bytes': 0, 'bytes_per_round': 0}
        
        total_params = list(self.clients.values())[0].get_num_parameters()
        param_bytes = total_params * 4
        
        total_bytes = param_bytes * 2 * self.current_round * len(self.clients)
        
        if self.mode == 'vertical' and self.cross_edge_info:
            cross_edge_bytes = (
                self.cross_edge_info.get('total_cross_edges', 0) *
                self.config.hidden_dims[-1] * 4 * self.current_round
            )
            total_bytes += cross_edge_bytes
        
        return {
            'total_bytes': total_bytes,
            'bytes_per_round': total_bytes / self.current_round if self.current_round > 0 else 0,
            'num_parameters': total_params
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the federated learning run."""
        return {
            'config': {
                'mode': self.mode,
                'num_clients': self.config.num_clients,
                'model_type': self.config.model_type,
                'num_rounds': self.current_round
            },
            'partition_info': self.partition_info,
            'cross_edge_info': self.cross_edge_info if self.mode == 'vertical' else None,
            'metrics': {
                'train_loss': self.metrics['train_loss'][-5:] if self.metrics['train_loss'] else [],
                'train_accuracy': self.metrics['train_accuracy'][-5:] if self.metrics['train_accuracy'] else [],
                'round_time': self.metrics['round_time'][-5:] if self.metrics['round_time'] else []
            },
            'communication_cost': self.get_communication_cost(),
            'server_stats': self.get_server_stats()
        }


def run_fed_graph_demo(
    mode: str = 'horizontal',
    num_clients: int = 4,
    num_rounds: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a demonstration of federated graph learning.
    
    Args:
        mode: 'horizontal' or 'vertical'
        num_clients: Number of clients
        num_rounds: Number of training rounds
        verbose: Print progress
    
    Returns:
        Results dictionary
    """
    print("=" * 60)
    print(f"Federated Graph Learning Demo - {mode.upper()} Mode")
    print("=" * 60)
    
    config = FedGraphConfig(
        mode=mode,
        num_clients=num_clients,
        num_rounds=num_rounds,
        model_type='gin' if mode == 'horizontal' else 'gcn',
        hidden_dims=[32, 16],
        input_dim=16,
        output_dim=3,
        local_epochs=2,
        use_dp=True,
        epsilon=1.0,
        seed=42
    )
    
    coordinator = FedGraphCoordinator(config)
    
    print(f"\nSetting up {mode} federated graph learning...")
    
    if mode == 'horizontal':
        coordinator.setup_horizontal(
            num_graphs=20,
            min_nodes=15,
            max_nodes=30,
            non_iid=True
        )
    else:
        coordinator.setup_vertical(
            num_nodes=100,
            partition_strategy='random'
        )
    
    print(f"\nClients: {len(coordinator.clients)}")
    if mode == 'vertical':
        print(f"Cross-edges: {coordinator.cross_edge_info.get('total_cross_edges', 0)}")
    
    print(f"\nStarting training for {num_rounds} rounds...")
    results = coordinator.train(verbose=verbose)
    
    print("\n" + "-" * 40)
    print("Training Results:")
    print(f"  Final Loss: {results['final_loss']:.4f}")
    print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Total Time: {results['total_time']:.2f}s")
    print(f"  Avg Round Time: {results['avg_round_time']:.2f}s")
    
    comm_cost = coordinator.get_communication_cost()
    print(f"\nCommunication Cost:")
    print(f"  Total Bytes: {comm_cost['total_bytes']:,}")
    print(f"  Bytes/Round: {comm_cost['bytes_per_round']:,.0f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    return {
        'results': results,
        'summary': coordinator.get_summary()
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("Testing Federated Graph Learning Coordinator")
    print("=" * 60)
    
    print("\n1. Testing Horizontal Mode...")
    h_results = run_fed_graph_demo(
        mode='horizontal',
        num_clients=3,
        num_rounds=5,
        verbose=True
    )
    
    print("\n2. Testing Vertical Mode...")
    v_results = run_fed_graph_demo(
        mode='vertical',
        num_clients=3,
        num_rounds=5,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("All coordinator tests passed!")
    print("=" * 60)
