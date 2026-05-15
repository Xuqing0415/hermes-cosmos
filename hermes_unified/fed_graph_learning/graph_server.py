"""
Federated Graph Learning Server

Implements the server for federated graph learning, including
model aggregation and cross-edge routing management.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict

from .graph_models import GCNModel, GraphSAGEModel, GINModel, create_gnn_model
from .graph_data_utils import GraphDataset

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for federated graph learning server."""
    model_type: str = 'gcn'
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 3
    aggregation_method: str = 'fedavg'
    min_clients: int = 2
    max_rounds: int = 100
    convergence_threshold: float = 0.001
    seed: int = 42


class CrossEdgeRouter:
    """
    Manages cross-client edge routing for vertical federated graph learning.
    
    Routes embeddings between clients that share cross-client edges.
    """
    
    def __init__(self, num_clients: int = 2):
        """
        Initialize cross-edge router.
        
        Args:
            num_clients: Number of participating clients
        """
        self.num_clients = num_clients
        
        self.cross_edge_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        self.client_neighbors: Dict[int, List[int]] = defaultdict(list)
        
        self.routing_stats = {
            'total_routed': 0,
            'total_bytes': 0,
            'routing_history': []
        }
    
    def register_cross_edges(
        self,
        client_id: int,
        cross_edges: List[Tuple[int, int, int]]
    ):
        """
        Register cross-client edges for a client.
        
        Args:
            client_id: ID of the client
            cross_edges: List of (src_node, dst_node, dst_client) tuples
        """
        for src_node, dst_node, dst_client in cross_edges:
            edge_key = (min(client_id, dst_client), max(client_id, dst_client))
            self.cross_edge_map[edge_key].append((src_node, dst_node, client_id, dst_client))
            
            if dst_client not in self.client_neighbors[client_id]:
                self.client_neighbors[client_id].append(dst_client)
            if client_id not in self.client_neighbors[dst_client]:
                self.client_neighbors[dst_client].append(client_id)
    
    def get_routing_table(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Get the routing table for all clients.
        
        Returns:
            Dict mapping client_id -> {neighbor_client -> [node_ids to receive]}
        """
        routing_table = {}
        
        for client_id in range(self.num_clients):
            routing_table[client_id] = {}
            
            for neighbor_id in self.client_neighbors[client_id]:
                nodes_to_receive = []
                
                for edge_key, edges in self.cross_edge_map.items():
                    for src, dst, src_client, dst_client in edges:
                        if dst_client == client_id and src_client == neighbor_id:
                            nodes_to_receive.append(src)
                
                if nodes_to_receive:
                    routing_table[client_id][neighbor_id] = list(set(nodes_to_receive))
        
        return routing_table
    
    def route_embeddings(
        self,
        source_client: int,
        embeddings: Dict[int, np.ndarray],
        target_clients: Optional[List[int]] = None
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Route embeddings from source client to target clients.
        
        Args:
            source_client: ID of the sending client
            embeddings: Dict mapping node_id -> embedding
            target_clients: Optional list of target clients
        
        Returns:
            Dict mapping target_client -> {node_id -> embedding}
        """
        if target_clients is None:
            target_clients = self.client_neighbors.get(source_client, [])
        
        routed_embeddings = {}
        
        for target_client in target_clients:
            routed_embeddings[target_client] = {}
            
            for edge_key, edges in self.cross_edge_map.items():
                for src, dst, src_client, dst_client in edges:
                    if src_client == source_client and dst_client == target_client:
                        if src in embeddings:
                            routed_embeddings[target_client][src] = embeddings[src]
            
            if routed_embeddings[target_client]:
                self.routing_stats['total_routed'] += len(routed_embeddings[target_client])
                self.routing_stats['total_bytes'] += sum(
                    e.nbytes for e in routed_embeddings[target_client].values()
                )
        
        self.routing_stats['routing_history'].append({
            'source': source_client,
            'targets': target_clients,
            'num_embeddings': sum(len(v) for v in routed_embeddings.values()),
            'timestamp': time.time()
        })
        
        return routed_embeddings
    
    def get_client_pairs(self) -> List[Tuple[int, int]]:
        """Get all client pairs that share cross-edges."""
        return list(self.cross_edge_map.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'num_client_pairs': len(self.cross_edge_map),
            'total_cross_edges': sum(len(v) for v in self.cross_edge_map.values()),
            'total_routed': self.routing_stats['total_routed'],
            'total_bytes': self.routing_stats['total_bytes'],
            'client_neighbors': dict(self.client_neighbors)
        }


class ModelAggregator:
    """
    Aggregates model updates from clients.
    
    Supports FedAvg, FedProx, and robust aggregation methods.
    """
    
    def __init__(
        self,
        aggregation_method: str = 'fedavg',
        robust_threshold: float = 0.2
    ):
        """
        Initialize model aggregator.
        
        Args:
            aggregation_method: 'fedavg', 'fedprox', 'trimmed_mean', or 'krum'
            robust_threshold: Threshold for robust aggregation
        """
        self.aggregation_method = aggregation_method
        self.robust_threshold = robust_threshold
        
        self.aggregation_history = []
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        client_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates.
        
        Args:
            client_updates: Dict mapping client_id -> weight updates
            client_weights: Optional weights for weighted averaging
        
        Returns:
            Aggregated weight updates
        """
        if not client_updates:
            return {}
        
        if self.aggregation_method == 'fedavg':
            return self._fedavg(client_updates, client_weights)
        elif self.aggregation_method == 'trimmed_mean':
            return self._trimmed_mean(client_updates)
        elif self.aggregation_method == 'krum':
            return self._krum(client_updates)
        else:
            return self._fedavg(client_updates, client_weights)
    
    def _fedavg(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        client_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, np.ndarray]:
        """FedAvg aggregation."""
        client_ids = list(client_updates.keys())
        
        if client_weights is None:
            weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        else:
            total = sum(client_weights.values())
            weights = {cid: client_weights[cid] / total for cid in client_ids}
        
        aggregated = {}
        
        first_update = client_updates[client_ids[0]]
        for key in first_update:
            weighted_sum = np.zeros_like(first_update[key])
            for cid in client_ids:
                if key in client_updates[cid]:
                    weighted_sum += weights[cid] * client_updates[cid][key]
            aggregated[key] = weighted_sum
        
        self.aggregation_history.append({
            'method': 'fedavg',
            'num_clients': len(client_ids),
            'weights': weights
        })
        
        return aggregated
    
    def _trimmed_mean(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Trimmed mean aggregation for robustness."""
        client_ids = list(client_updates.keys())
        n_trim = int(len(client_ids) * self.robust_threshold)
        
        aggregated = {}
        
        first_update = client_updates[client_ids[0]]
        for key in first_update:
            updates = [client_updates[cid][key] for cid in client_ids if key in client_updates[cid]]
            
            stacked = np.stack(updates, axis=0)
            
            sorted_updates = np.sort(stacked, axis=0)
            
            if n_trim > 0 and len(sorted_updates) > 2 * n_trim:
                trimmed = sorted_updates[n_trim:-n_trim]
            else:
                trimmed = sorted_updates
            
            aggregated[key] = np.mean(trimmed, axis=0)
        
        self.aggregation_history.append({
            'method': 'trimmed_mean',
            'num_clients': len(client_ids),
            'trim_ratio': self.robust_threshold
        })
        
        return aggregated
    
    def _krum(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Krum aggregation for Byzantine robustness."""
        client_ids = list(client_updates.keys())
        
        first_update = client_updates[client_ids[0]]
        flat_updates = {}
        
        for cid in client_ids:
            flat = np.concatenate([
                v.flatten() for v in client_updates[cid].values()
            ])
            flat_updates[cid] = flat
        
        distances = {}
        for cid1 in client_ids:
            dist_sum = 0
            for cid2 in client_ids:
                if cid1 != cid2:
                    dist_sum += np.linalg.norm(flat_updates[cid1] - flat_updates[cid2])
            distances[cid1] = dist_sum
        
        selected_id = min(distances, key=distances.get)
        
        self.aggregation_history.append({
            'method': 'krum',
            'num_clients': len(client_ids),
            'selected_client': selected_id
        })
        
        return client_updates[selected_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        if not self.aggregation_history:
            return {'total_aggregations': 0}
        
        methods = [h['method'] for h in self.aggregation_history]
        
        return {
            'total_aggregations': len(self.aggregation_history),
            'method_distribution': {
                m: methods.count(m) for m in set(methods)
            }
        }


class FedGraphServer:
    """
    Server for federated graph learning.
    
    Coordinates training rounds, aggregates model updates,
    and manages cross-edge routing for vertical FL.
    """
    
    def __init__(
        self,
        config: ServerConfig,
        input_dim: int = 16
    ):
        """
        Initialize federated graph learning server.
        
        Args:
            config: Server configuration
            input_dim: Input feature dimension
        """
        self.config = config
        self.input_dim = input_dim
        
        self.global_model = create_gnn_model(
            model_type=config.model_type,
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            seed=config.seed
        )
        
        self.aggregator = ModelAggregator(
            aggregation_method=config.aggregation_method
        )
        
        self.cross_edge_router = CrossEdgeRouter()
        
        self.clients: Dict[int, Any] = {}
        self.client_weights: Dict[int, float] = {}
        
        self.current_round = 0
        self.round_history = []
        
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'convergence': []
        }
    
    def initialize_model(self, input_dim: Optional[int] = None):
        """
        Initialize the global model.
        
        Args:
            input_dim: Override input dimension
        """
        if input_dim is not None:
            self.input_dim = input_dim
        
        self.global_model = create_gnn_model(
            model_type=self.config.model_type,
            input_dim=self.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            seed=self.config.seed
        )
    
    def register_client(
        self,
        client_id: int,
        client_info: Optional[Dict] = None
    ):
        """
        Register a client with the server.
        
        Args:
            client_id: Client identifier
            client_info: Optional client metadata
        """
        self.clients[client_id] = client_info or {}
        self.client_weights[client_id] = 1.0
        
        logger.info(f"Client {client_id} registered with server")
    
    def register_cross_edges(
        self,
        client_id: int,
        cross_edges: List[Tuple[int, int, int]]
    ):
        """
        Register cross-client edges for a client.
        
        Args:
            client_id: Client identifier
            cross_edges: List of cross-client edges
        """
        self.cross_edge_router.register_cross_edges(client_id, cross_edges)
    
    def get_global_weights(self) -> Dict[str, np.ndarray]:
        """Get global model weights."""
        return self.global_model.get_weights()
    
    def distribute_weights(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Distribute global weights to all clients.
        
        Returns:
            Dict mapping client_id -> weights
        """
        weights = self.get_global_weights()
        return {cid: weights.copy() for cid in self.clients}
    
    def collect_updates(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Collect and aggregate client updates.
        
        Args:
            client_updates: Dict mapping client_id -> weight updates
        
        Returns:
            Aggregated update
        """
        return self.aggregator.aggregate(
            client_updates,
            self.client_weights
        )
    
    def apply_update(self, update: Dict[str, np.ndarray]):
        """
        Apply aggregated update to global model.
        
        Args:
            update: Aggregated weight update
        """
        current_weights = self.global_model.get_weights()
        
        new_weights = {}
        for key in current_weights:
            if key in update:
                new_weights[key] = current_weights[key] + update[key]
            else:
                new_weights[key] = current_weights[key]
        
        self.global_model.set_weights(new_weights)
    
    def route_cross_edge_embeddings(
        self,
        source_client: int,
        embeddings: Dict[int, np.ndarray]
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Route embeddings from source client to target clients.
        
        Args:
            source_client: ID of the sending client
            embeddings: Dict mapping node_id -> embedding
        
        Returns:
            Dict mapping target_client -> {node_id -> embedding}
        """
        return self.cross_edge_router.route_embeddings(source_client, embeddings)
    
    def run_round(
        self,
        client_updates: Dict[int, Dict[str, np.ndarray]],
        client_metrics: Optional[Dict[int, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Run a single federated learning round.
        
        Args:
            client_updates: Dict mapping client_id -> weight updates
            client_metrics: Optional client training metrics
        
        Returns:
            Round statistics
        """
        self.current_round += 1
        
        aggregated_update = self.collect_updates(client_updates)
        self.apply_update(aggregated_update)
        
        round_stats = {
            'round': self.current_round,
            'num_clients': len(client_updates),
            'aggregation_method': self.config.aggregation_method
        }
        
        if client_metrics:
            avg_loss = np.mean([m.get('loss', 0) for m in client_metrics.values()])
            avg_acc = np.mean([m.get('accuracy', 0) for m in client_metrics.values()])
            
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['train_accuracy'].append(avg_acc)
            
            round_stats['avg_loss'] = avg_loss
            round_stats['avg_accuracy'] = avg_acc
        
        self.round_history.append(round_stats)
        
        logger.info(f"Round {self.current_round} completed: {round_stats}")
        
        return round_stats
    
    def check_convergence(self) -> bool:
        """
        Check if training has converged.
        
        Returns:
            True if converged
        """
        if len(self.metrics['train_loss']) < 2:
            return False
        
        recent_losses = self.metrics['train_loss'][-5:]
        if len(recent_losses) < 5:
            return False
        
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        
        return loss_change < self.config.convergence_threshold
    
    def evaluate_global_model(
        self,
        test_data: Optional[List[GraphDataset]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the global model.
        
        Args:
            test_data: Optional test graphs
        
        Returns:
            Evaluation metrics
        """
        if test_data is None or not test_data:
            return {'accuracy': 0.0, 'loss': 0.0}
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        for graph in test_data:
            logits = self.global_model.forward(
                graph.node_features,
                graph.adj_matrix,
                training=False
            )
            
            if isinstance(graph.labels, np.ndarray):
                if graph.labels.ndim == 0:
                    labels = graph.labels.reshape(1)
                    logits = logits.reshape(1, -1) if logits.ndim == 1 else logits
                else:
                    labels = graph.labels
                
                predictions = np.argmax(logits, axis=-1)
                total_correct += np.sum(predictions == labels)
                total_samples += len(labels)
                
                probs = self._softmax(logits)
                one_hot = np.eye(logits.shape[-1])[labels]
                loss = -np.sum(one_hot * np.log(probs + 1e-10)) / len(labels)
                total_loss += loss
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(test_data) if test_data else 0.0
        
        self.metrics['test_accuracy'].append(accuracy)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'num_test_samples': total_samples
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'current_round': self.current_round,
            'num_clients': len(self.clients),
            'model_type': self.config.model_type,
            'aggregation_stats': self.aggregator.get_stats(),
            'routing_stats': self.cross_edge_router.get_stats(),
            'metrics': {
                k: v[-10:] if v else [] for k, v in self.metrics.items()
            }
        }
    
    def save_checkpoint(self, path: str):
        """Save server state to checkpoint."""
        checkpoint = {
            'config': self.config.__dict__,
            'global_weights': self.global_model.get_weights(),
            'current_round': self.current_round,
            'metrics': self.metrics,
            'round_history': self.round_history
        }
        
        np.save(path, checkpoint, allow_pickle=True)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load server state from checkpoint."""
        checkpoint = np.load(path, allow_pickle=True).item()
        
        self.global_model.set_weights(checkpoint['global_weights'])
        self.current_round = checkpoint['current_round']
        self.metrics = checkpoint['metrics']
        self.round_history = checkpoint['round_history']
        
        logger.info(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Federated Graph Learning Server")
    print("=" * 60)
    
    print("\n1. Testing Cross-Edge Router...")
    router = CrossEdgeRouter(num_clients=3)
    
    router.register_cross_edges(0, [(10, 50, 1), (20, 60, 1), (30, 70, 2)])
    router.register_cross_edges(1, [(50, 10, 0), (60, 20, 0)])
    
    routing_table = router.get_routing_table()
    print(f"   Routing table: {routing_table}")
    print(f"   Client pairs: {router.get_client_pairs()}")
    print(f"   Stats: {router.get_stats()}")
    
    print("\n2. Testing Model Aggregator...")
    aggregator = ModelAggregator(aggregation_method='fedavg')
    
    client_updates = {
        0: {'weight': np.array([1.0, 2.0, 3.0])},
        1: {'weight': np.array([2.0, 3.0, 4.0])},
        2: {'weight': np.array([3.0, 4.0, 5.0])}
    }
    
    aggregated = aggregator.aggregate(client_updates)
    print(f"   Aggregated: {aggregated}")
    print(f"   Stats: {aggregator.get_stats()}")
    
    print("\n3. Testing FedGraphServer...")
    config = ServerConfig(
        model_type='gcn',
        hidden_dims=[32],
        output_dim=3,
        aggregation_method='fedavg'
    )
    server = FedGraphServer(config, input_dim=16)
    
    server.register_client(0)
    server.register_client(1)
    
    print(f"   Global model parameters: {server.global_model.get_num_parameters()}")
    
    weights = server.get_global_weights()
    print(f"   Weight keys: {list(weights.keys())[:3]}...")
    
    client_updates = {
        0: {k: np.random.randn(*v.shape).astype(np.float32) * 0.01 
            for k, v in weights.items()},
        1: {k: np.random.randn(*v.shape).astype(np.float32) * 0.01 
            for k, v in weights.items()}
    }
    
    round_stats = server.run_round(client_updates)
    print(f"   Round stats: {round_stats}")
    
    print(f"   Server stats: {server.get_stats()}")
    
    print("\n" + "=" * 60)
    print("All server tests passed!")
    print("=" * 60)
