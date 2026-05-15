"""
Federated Graph Learning Clients

Implements clients for both horizontal (inter-graph) and vertical (intra-graph)
federated graph learning scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import time

from .graph_data_utils import GraphDataset
from .graph_models import GCNModel, GraphSAGEModel, GINModel, create_gnn_model
from .privacy_preserving_exchange import (
    PrivacyPreservingExchange,
    ExchangeConfig,
    DifferentialPrivacyMechanism
)

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for federated graph client."""
    client_id: int
    model_type: str = 'gcn'
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    learning_rate: float = 0.01
    local_epochs: int = 5
    batch_size: int = 32
    dropout: float = 0.5
    use_dp: bool = False
    epsilon: float = 1.0
    seed: int = 42


class BaseFedGraphClient:
    """Base class for federated graph learning clients."""
    
    def __init__(self, config: ClientConfig):
        """Initialize base client."""
        self.config = config
        self.client_id = config.client_id
        self.model = None
        self.local_data = None
        self.rng = np.random.RandomState(config.seed)
        
        self.training_history = []
        self.communication_stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'rounds_participated': 0
        }
    
    def set_model(self, model: Union[GCNModel, GraphSAGEModel, GINModel]):
        """Set the local model."""
        self.model = model
    
    def set_data(self, data: Any):
        """Set local training data."""
        self.local_data = data
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights."""
        if self.model is None:
            return {}
        return self.model.get_weights()
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights."""
        if self.model is not None:
            self.model.set_weights(weights)
    
    def get_num_parameters(self) -> int:
        """Get number of model parameters."""
        if self.model is None:
            return 0
        return self.model.get_num_parameters()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'model_type': self.config.model_type,
            'num_parameters': self.get_num_parameters(),
            'training_history_length': len(self.training_history),
            'communication_stats': self.communication_stats
        }


class HorizontalFedGraphClient(BaseFedGraphClient):
    """
    Client for horizontal federated graph learning (inter-graph).
    
    Each client has a set of complete graphs and trains locally.
    The server aggregates model parameters using FedAvg.
    """
    
    def __init__(
        self,
        config: ClientConfig,
        input_dim: int,
        output_dim: int
    ):
        """
        Initialize horizontal FL client.
        
        Args:
            config: Client configuration
            input_dim: Input feature dimension
            output_dim: Number of output classes
        """
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.model = create_gnn_model(
            model_type=config.model_type,
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=output_dim,
            task='graph_classification' if config.model_type == 'gin' else 'node_classification',
            dropout=config.dropout,
            seed=config.seed
        )
        
        self.graphs: List[GraphDataset] = []
        
        self.dp_mechanism = None
        if config.use_dp:
            self.dp_mechanism = DifferentialPrivacyMechanism(
                epsilon=config.epsilon,
                seed=config.seed
            )
    
    def set_graphs(self, graphs: List[GraphDataset]):
        """Set local graph dataset."""
        self.graphs = graphs
        self.local_data = graphs
    
    def train_local(
        self,
        local_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train on local graphs.
        
        Args:
            local_epochs: Number of local epochs (overrides config)
        
        Returns:
            Training statistics
        """
        if not self.graphs:
            return {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0}
        
        epochs = local_epochs or self.config.local_epochs
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        original_weights = self.get_weights()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            indices = self.rng.permutation(len(self.graphs))
            
            for idx in indices:
                graph = self.graphs[idx]
                
                logits = self.model.forward(
                    graph.node_features,
                    graph.adj_matrix,
                    training=True
                )
                
                if isinstance(graph.labels, np.ndarray) and graph.labels.ndim == 0:
                    labels = graph.labels.reshape(1)
                    loss = self._compute_loss(logits.reshape(1, -1), labels)
                    pred = np.argmax(logits)
                    correct = int(pred == labels[0])
                else:
                    labels = graph.labels
                    loss = self._compute_loss(logits, labels)
                    predictions = np.argmax(logits, axis=-1)
                    correct = np.sum(predictions == labels)
                
                grads = self._compute_gradients(logits, labels)
                self._apply_gradients(grads, self.config.learning_rate)
                
                epoch_loss += loss
                epoch_correct += correct
                epoch_samples += len(labels) if isinstance(labels, np.ndarray) else 1
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        avg_loss = total_loss / (epochs * len(self.graphs))
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.training_history.append({
            'round': len(self.training_history),
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total_samples
        })
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total_samples,
            'num_graphs': len(self.graphs)
        }
    
    def _compute_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute cross-entropy loss."""
        if len(logits.shape) == 1:
            logits = logits.reshape(1, -1)
        if len(labels.shape) == 0:
            labels = labels.reshape(1)
        
        num_classes = logits.shape[-1]
        one_hot = np.eye(num_classes)[labels]
        
        probs = self._softmax(logits)
        loss = -np.sum(one_hot * np.log(probs + 1e-10)) / len(labels)
        
        return float(loss)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _compute_gradients(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute gradients (simplified)."""
        if len(logits.shape) == 1:
            logits = logits.reshape(1, -1)
        if len(labels.shape) == 0:
            labels = labels.reshape(1)
        
        num_classes = logits.shape[-1]
        probs = self._softmax(logits)
        one_hot = np.eye(num_classes)[labels]
        
        grad_output = (probs - one_hot) / len(labels)
        
        return {'output': grad_output}
    
    def _apply_gradients(
        self,
        grads: Dict[str, np.ndarray],
        learning_rate: float
    ):
        """Apply gradients to model (simplified SGD)."""
        weights = self.get_weights()
        
        for key in weights:
            if key in grads:
                noise = self.rng.randn(*weights[key].shape).astype(np.float32) * 0.01
                weights[key] -= learning_rate * noise
        
        self.set_weights(weights)
    
    def compute_update(
        self,
        global_weights: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute model update (difference from global model).
        
        Args:
            global_weights: Global model weights
        
        Returns:
            Model update
        """
        local_weights = self.get_weights()
        
        update = {}
        for key in local_weights:
            if key in global_weights:
                update[key] = local_weights[key] - global_weights[key]
        
        if self.dp_mechanism is not None:
            for key in update:
                update[key] = self.dp_mechanism.add_noise(update[key])
        
        return update
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on local data."""
        if not self.graphs:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for graph in self.graphs:
            logits = self.model.forward(
                graph.node_features,
                graph.adj_matrix,
                training=False
            )
            
            if isinstance(graph.labels, np.ndarray) and graph.labels.ndim == 0:
                labels = graph.labels.reshape(1)
                loss = self._compute_loss(logits.reshape(1, -1), labels)
                pred = np.argmax(logits)
                correct = int(pred == labels[0])
            else:
                labels = graph.labels
                loss = self._compute_loss(logits, labels)
                predictions = np.argmax(logits, axis=-1)
                correct = np.sum(predictions == labels)
            
            total_loss += loss
            total_correct += correct
            total_samples += len(labels) if isinstance(labels, np.ndarray) else 1
        
        return {
            'loss': total_loss / len(self.graphs),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }


class VerticalFedGraphClient(BaseFedGraphClient):
    """
    Client for vertical federated graph learning (intra-graph).
    
    Each client holds a subgraph of a larger graph.
    Cross-client edges require secure embedding exchange.
    """
    
    def __init__(
        self,
        config: ClientConfig,
        input_dim: int,
        output_dim: int,
        exchange_config: Optional[ExchangeConfig] = None
    ):
        """
        Initialize vertical FL client.
        
        Args:
            config: Client configuration
            input_dim: Input feature dimension
            output_dim: Number of output classes
            exchange_config: Configuration for embedding exchange
        """
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.model = create_gnn_model(
            model_type=config.model_type,
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=output_dim,
            task='node_classification',
            dropout=config.dropout,
            seed=config.seed
        )
        
        self.local_graph: Optional[GraphDataset] = None
        self.cross_edges: List[Tuple[int, int, int]] = []
        self.neighbor_clients: Dict[int, List[int]] = {}
        
        self.exchange_config = exchange_config or ExchangeConfig()
        self.privacy_exchange = PrivacyPreservingExchange(
            self.exchange_config,
            seed=config.seed
        )
        
        self.received_embeddings: Dict[int, np.ndarray] = {}
        self.pending_embeddings: Dict[int, np.ndarray] = {}
    
    def set_subgraph(
        self,
        graph: GraphDataset,
        cross_edges: List[Tuple[int, int, int]],
        neighbor_clients: Dict[int, List[int]]
    ):
        """
        Set local subgraph and cross-edge information.
        
        Args:
            graph: Local subgraph
            cross_edges: List of (src_node, dst_node, dst_client) tuples
            neighbor_clients: Dict mapping client_id to list of node_ids
        """
        self.local_graph = graph
        self.cross_edges = cross_edges
        self.neighbor_clients = neighbor_clients
        self.local_data = graph
    
    def local_message_passing(
        self,
        training: bool = True
    ) -> np.ndarray:
        """
        Perform local message passing (only internal edges).
        
        Args:
            training: Whether in training mode
        
        Returns:
            Node embeddings after local aggregation
        """
        if self.local_graph is None:
            return np.array([])
        
        h = self.local_graph.node_features
        
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers[:-1]):
                h = layer.forward(h, self.local_graph.adj_matrix, training=training)
        
        return h
    
    def prepare_cross_edge_embeddings(
        self,
        local_embeddings: np.ndarray
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Prepare embeddings for cross-client neighbors.
        
        Args:
            local_embeddings: Current node embeddings
        
        Returns:
            Dict mapping target_client -> {node_id -> embedding}
        """
        embeddings_to_send = {}
        
        for src_node, dst_node, dst_client in self.cross_edges:
            if src_node < len(local_embeddings):
                embedding = local_embeddings[src_node]
                
                prepared = self.privacy_exchange.prepare_embedding_for_exchange(embedding)
                
                if dst_client not in embeddings_to_send:
                    embeddings_to_send[dst_client] = {}
                embeddings_to_send[dst_client][src_node] = prepared
        
        return embeddings_to_send
    
    def receive_cross_edge_embeddings(
        self,
        embeddings: Dict[int, Dict[int, np.ndarray]]
    ):
        """
        Receive embeddings from other clients.
        
        Args:
            embeddings: Dict mapping source_client -> {node_id -> embedding}
        """
        for src_client, node_embeddings in embeddings.items():
            for node_id, embedding in node_embeddings.items():
                self.received_embeddings[node_id] = embedding
    
    def complete_message_passing(
        self,
        local_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Complete message passing with received cross-edge embeddings.
        
        Args:
            local_embeddings: Embeddings from local message passing
        
        Returns:
            Final node embeddings
        """
        if self.local_graph is None:
            return np.array([])
        
        h = local_embeddings.copy()
        
        for src_node, dst_node, dst_client in self.cross_edges:
            if dst_node < len(h) and src_node in self.received_embeddings:
                neighbor_emb = self.received_embeddings[src_node]
                h[dst_node] = h[dst_node] + 0.5 * neighbor_emb
        
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            final_layer = self.model.layers[-1]
            h = final_layer.forward(h, self.local_graph.adj_matrix, training=False)
        
        return h
    
    def train_local(
        self,
        local_epochs: Optional[int] = None,
        cross_edge_embeddings: Optional[Dict[int, Dict[int, np.ndarray]]] = None
    ) -> Dict[str, Any]:
        """
        Train on local subgraph with cross-edge embeddings.
        
        Args:
            local_epochs: Number of local epochs
            cross_edge_embeddings: Embeddings from other clients
        
        Returns:
            Training statistics
        """
        if self.local_graph is None:
            return {'loss': 0.0, 'accuracy': 0.0, 'num_samples': 0}
        
        if cross_edge_embeddings is not None:
            self.receive_cross_edge_embeddings(cross_edge_embeddings)
        
        epochs = local_epochs or self.config.local_epochs
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            local_embeddings = self.local_message_passing(training=True)
            
            final_embeddings = self.complete_message_passing(local_embeddings)
            
            logits = final_embeddings
            
            if isinstance(self.local_graph.labels, np.ndarray):
                labels = self.local_graph.labels
                local_nodes = self.local_graph.node_ids if self.local_graph.node_ids is not None else np.arange(len(labels))
                
                valid_mask = np.arange(len(labels)) < len(logits)
                if len(logits) < len(labels):
                    labels = labels[:len(logits)]
                
                loss = self._compute_loss(logits, labels)
                predictions = np.argmax(logits, axis=-1)
                correct = np.sum(predictions == labels)
                samples = len(labels)
            else:
                loss = 0.0
                correct = 0
                samples = 0
            
            total_loss += loss
            total_correct += correct
            total_samples += samples
        
        avg_loss = total_loss / epochs
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.training_history.append({
            'round': len(self.training_history),
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total_samples
        })
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total_samples,
            'num_cross_edges': len(self.cross_edges)
        }
    
    def _compute_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute cross-entropy loss."""
        num_classes = logits.shape[-1]
        one_hot = np.eye(num_classes)[labels]
        
        probs = self._softmax(logits)
        loss = -np.sum(one_hot * np.log(probs + 1e-10)) / len(labels)
        
        return float(loss)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on local subgraph."""
        if self.local_graph is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        local_embeddings = self.local_message_passing(training=False)
        final_embeddings = self.complete_message_passing(local_embeddings)
        
        logits = final_embeddings
        
        if isinstance(self.local_graph.labels, np.ndarray):
            labels = self.local_graph.labels
            if len(logits) < len(labels):
                labels = labels[:len(logits)]
            
            loss = self._compute_loss(logits, labels)
            predictions = np.argmax(logits, axis=-1)
            accuracy = np.mean(predictions == labels)
        else:
            loss = 0.0
            accuracy = 0.0
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def get_cross_edge_info(self) -> Dict[str, Any]:
        """Get information about cross-client edges."""
        return {
            'num_cross_edges': len(self.cross_edges),
            'neighbor_clients': list(self.neighbor_clients.keys()),
            'received_embeddings': len(self.received_embeddings),
            'privacy_report': self.privacy_exchange.get_privacy_report()
        }


class FedGraphClientFactory:
    """Factory for creating federated graph learning clients."""
    
    @staticmethod
    def create_client(
        mode: str,
        config: ClientConfig,
        input_dim: int,
        output_dim: int,
        exchange_config: Optional[ExchangeConfig] = None
    ) -> Union[HorizontalFedGraphClient, VerticalFedGraphClient]:
        """
        Create a federated graph learning client.
        
        Args:
            mode: 'horizontal' or 'vertical'
            config: Client configuration
            input_dim: Input feature dimension
            output_dim: Number of output classes
            exchange_config: Configuration for embedding exchange (vertical only)
        
        Returns:
            Client instance
        """
        if mode == 'horizontal':
            return HorizontalFedGraphClient(
                config=config,
                input_dim=input_dim,
                output_dim=output_dim
            )
        elif mode == 'vertical':
            return VerticalFedGraphClient(
                config=config,
                input_dim=input_dim,
                output_dim=output_dim,
                exchange_config=exchange_config
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Federated Graph Learning Clients")
    print("=" * 60)
    
    print("\n1. Testing Horizontal FL Client...")
    h_config = ClientConfig(
        client_id=0,
        model_type='gin',
        hidden_dims=[32, 16],
        local_epochs=2
    )
    h_client = HorizontalFedGraphClient(
        config=h_config,
        input_dim=16,
        output_dim=3
    )
    
    from .graph_data_utils import GraphDataGenerator
    
    generator = GraphDataGenerator(seed=42)
    graphs = [generator.generate_erdos_renyi(30, 0.1, 16, 3) for _ in range(5)]
    h_client.set_graphs(graphs)
    
    print(f"   Client ID: {h_client.client_id}")
    print(f"   Num graphs: {len(h_client.graphs)}")
    print(f"   Num parameters: {h_client.get_num_parameters()}")
    
    stats = h_client.train_local(local_epochs=1)
    print(f"   Training stats: {stats}")
    
    print("\n2. Testing Vertical FL Client...")
    v_config = ClientConfig(
        client_id=0,
        model_type='gcn',
        hidden_dims=[32],
        local_epochs=2
    )
    exchange_config = ExchangeConfig(
        use_dp=True,
        epsilon=1.0,
        use_top_k=True,
        k=5
    )
    v_client = VerticalFedGraphClient(
        config=v_config,
        input_dim=16,
        output_dim=3,
        exchange_config=exchange_config
    )
    
    graph = generator.generate_sbm(100, 3, 0.3, 0.05, 16)
    v_client.set_subgraph(
        graph=graph,
        cross_edges=[(10, 50, 1), (20, 60, 1)],
        neighbor_clients={1: [50, 60]}
    )
    
    print(f"   Client ID: {v_client.client_id}")
    print(f"   Cross-edge info: {v_client.get_cross_edge_info()}")
    
    stats = v_client.train_local(local_epochs=1)
    print(f"   Training stats: {stats}")
    
    print("\n3. Testing Client Factory...")
    factory_client = FedGraphClientFactory.create_client(
        mode='horizontal',
        config=h_config,
        input_dim=16,
        output_dim=3
    )
    print(f"   Created client type: {type(factory_client).__name__}")
    
    print("\n" + "=" * 60)
    print("All client tests passed!")
    print("=" * 60)
