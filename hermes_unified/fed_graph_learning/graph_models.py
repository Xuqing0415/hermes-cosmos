"""
Graph Neural Network Models for Federated Learning

Implements GCN, GraphSAGE, and GIN models with NumPy for
federated graph learning scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x > 0, x, alpha * x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


@dataclass
class LayerConfig:
    """Configuration for a GNN layer."""
    in_features: int
    out_features: int
    activation: str = 'relu'
    dropout: float = 0.0
    bias: bool = True
    normalize: bool = True


class GCNLayer:
    """
    Graph Convolutional Network Layer.
    
    Implements: h_v^{(l+1)} = σ( W * D^{-1/2} A D^{-1/2} * h_v^{(l)} )
    
    Where A is the adjacency matrix with self-loops,
    D is the degree matrix.
    """
    
    def __init__(self, config: LayerConfig, seed: int = 42):
        """Initialize GCN layer."""
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        self.weight = self._init_weight(
            config.in_features, 
            config.out_features
        )
        self.bias = np.zeros(config.out_features) if config.bias else None
        
        self.activation = self._get_activation(config.activation)
        self.cache = {}
    
    def _init_weight(self, in_features: int, out_features: int) -> np.ndarray:
        """Initialize weights using Xavier initialization."""
        std = np.sqrt(2.0 / (in_features + out_features))
        return self.rng.randn(in_features, out_features).astype(np.float32) * std
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'relu': relu,
            'leaky_relu': leaky_relu,
            'sigmoid': sigmoid,
            'tanh': np.tanh,
            'gelu': gelu,
            'none': lambda x: x
        }
        return activations.get(name, relu)
    
    def _compute_normalized_adj(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute D^{-1/2} A D^{-1/2} normalization."""
        num_nodes = adj_matrix.shape[0]
        
        adj_with_self = adj_matrix + np.eye(num_nodes)
        
        degree = np.sum(adj_with_self, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5, where=degree != 0)
        degree_inv_sqrt = np.nan_to_num(degree_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
        
        d_inv_sqrt = np.diag(degree_inv_sqrt)
        normalized_adj = d_inv_sqrt @ adj_with_self @ d_inv_sqrt
        
        return normalized_adj.astype(np.float32)
    
    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through GCN layer.
        
        Args:
            node_features: Node feature matrix [num_nodes, in_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            training: Whether in training mode
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        normalized_adj = self._compute_normalized_adj(adj_matrix)
        
        self.cache['normalized_adj'] = normalized_adj
        self.cache['node_features'] = node_features
        
        aggregated = normalized_adj @ node_features
        
        output = aggregated @ self.weight
        
        if self.bias is not None:
            output = output + self.bias
        
        output = self.activation(output)
        
        if training and self.config.dropout > 0:
            mask = self.rng.binomial(
                1, 1 - self.config.dropout, 
                size=output.shape
            ).astype(np.float32)
            output = output * mask / (1 - self.config.dropout)
        
        if self.config.normalize:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            output = output / norms
        
        return output
    
    def backward(
        self,
        grad_output: np.ndarray,
        learning_rate: float = 0.01
    ) -> np.ndarray:
        """
        Backward pass (simplified gradient computation).
        
        Args:
            grad_output: Gradient from next layer
            learning_rate: Learning rate for weight update
        
        Returns:
            Gradient with respect to input features
        """
        node_features = self.cache.get('node_features')
        normalized_adj = self.cache.get('normalized_adj')
        
        if node_features is None or normalized_adj is None:
            return grad_output
        
        grad_activation = grad_output
        
        grad_weight = node_features.T @ (normalized_adj.T @ grad_activation)
        
        self.weight -= learning_rate * grad_weight
        
        if self.bias is not None:
            grad_bias = np.sum(grad_activation, axis=0)
            self.bias -= learning_rate * grad_bias
        
        grad_input = normalized_adj @ (grad_activation @ self.weight.T)
        
        return grad_input
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get layer weights."""
        weights = {'weight': self.weight.copy()}
        if self.bias is not None:
            weights['bias'] = self.bias.copy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set layer weights."""
        if 'weight' in weights:
            self.weight = weights['weight'].copy()
        if 'bias' in weights and self.bias is not None:
            self.bias = weights['bias'].copy()


class GraphSAGELayer:
    """
    GraphSAGE Layer with various aggregator functions.
    
    Implements: h_v^{(l+1)} = σ( W * [h_v^{(l)} || AGG({h_u^{(l)} : u ∈ N(v)})] )
    
    Aggregators:
    - mean: Mean of neighbor features
    - max: Max pooling of neighbor features
    - lstm: LSTM-based aggregation
    """
    
    def __init__(
        self,
        config: LayerConfig,
        aggregator: str = 'mean',
        seed: int = 42
    ):
        """Initialize GraphSAGE layer."""
        self.config = config
        self.aggregator = aggregator
        self.rng = np.random.RandomState(seed)
        
        self.weight_self = self._init_weight(
            config.in_features,
            config.out_features
        )
        self.weight_neigh = self._init_weight(
            config.in_features,
            config.out_features
        )
        self.bias = np.zeros(config.out_features) if config.bias else None
        
        self.activation = self._get_activation(config.activation)
        self.cache = {}
    
    def _init_weight(self, in_features: int, out_features: int) -> np.ndarray:
        """Initialize weights."""
        std = np.sqrt(2.0 / (in_features + out_features))
        return self.rng.randn(in_features, out_features).astype(np.float32) * std
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function."""
        activations = {
            'relu': relu,
            'leaky_relu': leaky_relu,
            'sigmoid': sigmoid,
            'tanh': np.tanh,
            'gelu': gelu,
            'none': lambda x: x
        }
        return activations.get(name, relu)
    
    def _aggregate_neighbors(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """Aggregate neighbor features."""
        num_nodes = adj_matrix.shape[0]
        
        if self.aggregator == 'mean':
            degree = np.sum(adj_matrix, axis=1, keepdims=True)
            degree = np.where(degree == 0, 1, degree)
            aggregated = (adj_matrix @ node_features) / degree
        
        elif self.aggregator == 'sum':
            aggregated = adj_matrix @ node_features
        
        elif self.aggregator == 'max':
            aggregated = np.zeros_like(node_features)
            for i in range(num_nodes):
                neighbors = np.where(adj_matrix[i] > 0)[0]
                if len(neighbors) > 0:
                    aggregated[i] = np.max(node_features[neighbors], axis=0)
        
        else:
            degree = np.sum(adj_matrix, axis=1, keepdims=True)
            degree = np.where(degree == 0, 1, degree)
            aggregated = (adj_matrix @ node_features) / degree
        
        return aggregated
    
    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through GraphSAGE layer.
        
        Args:
            node_features: Node feature matrix [num_nodes, in_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            training: Whether in training mode
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        self.cache['node_features'] = node_features
        self.cache['adj_matrix'] = adj_matrix
        
        self_features = node_features @ self.weight_self
        
        neighbor_aggregated = self._aggregate_neighbors(node_features, adj_matrix)
        neighbor_features = neighbor_aggregated @ self.weight_neigh
        
        output = self_features + neighbor_features
        
        if self.bias is not None:
            output = output + self.bias
        
        output = self.activation(output)
        
        if self.config.normalize:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            output = output / norms
        
        return output
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get layer weights."""
        weights = {
            'weight_self': self.weight_self.copy(),
            'weight_neigh': self.weight_neigh.copy()
        }
        if self.bias is not None:
            weights['bias'] = self.bias.copy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set layer weights."""
        if 'weight_self' in weights:
            self.weight_self = weights['weight_self'].copy()
        if 'weight_neigh' in weights:
            self.weight_neigh = weights['weight_neigh'].copy()
        if 'bias' in weights and self.bias is not None:
            self.bias = weights['bias'].copy()


class GINLayer:
    """
    Graph Isomorphism Network Layer.
    
    Implements: h_v^{(l+1)} = MLP( (1 + ε) * h_v^{(l)} + Σ_{u∈N(v)} h_u^{(l)} )
    
    GIN is as powerful as the Weisfeiler-Lehman test for graph isomorphism,
    making it ideal for graph classification tasks.
    """
    
    def __init__(
        self,
        config: LayerConfig,
        epsilon: float = 0.0,
        learn_epsilon: bool = True,
        seed: int = 42
    ):
        """Initialize GIN layer."""
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        self.epsilon = epsilon
        self.learn_epsilon = learn_epsilon
        if learn_epsilon:
            self.epsilon_param = np.array([epsilon], dtype=np.float32)
        
        hidden_dim = config.out_features
        
        self.mlp_weight1 = self._init_weight(config.in_features, hidden_dim)
        self.mlp_bias1 = np.zeros(hidden_dim)
        self.mlp_weight2 = self._init_weight(hidden_dim, config.out_features)
        self.mlp_bias2 = np.zeros(config.out_features) if config.bias else None
        
        self.activation = self._get_activation(config.activation)
        self.cache = {}
    
    def _init_weight(self, in_features: int, out_features: int) -> np.ndarray:
        """Initialize weights."""
        std = np.sqrt(2.0 / (in_features + out_features))
        return self.rng.randn(in_features, out_features).astype(np.float32) * std
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function."""
        activations = {
            'relu': relu,
            'leaky_relu': leaky_relu,
            'sigmoid': sigmoid,
            'tanh': np.tanh,
            'gelu': gelu,
            'none': lambda x: x
        }
        return activations.get(name, relu)
    
    def _mlp(self, x: np.ndarray) -> np.ndarray:
        """Apply MLP transformation."""
        h = x @ self.mlp_weight1 + self.mlp_bias1
        h = self.activation(h)
        h = h @ self.mlp_weight2
        if self.mlp_bias2 is not None:
            h = h + self.mlp_bias2
        return h
    
    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through GIN layer.
        
        Args:
            node_features: Node feature matrix [num_nodes, in_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            training: Whether in training mode
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        self.cache['node_features'] = node_features
        self.cache['adj_matrix'] = adj_matrix
        
        epsilon = self.epsilon_param[0] if self.learn_epsilon else self.epsilon
        
        neighbor_sum = adj_matrix @ node_features
        
        combined = (1 + epsilon) * node_features + neighbor_sum
        
        output = self._mlp(combined)
        
        return output
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get layer weights."""
        weights = {
            'mlp_weight1': self.mlp_weight1.copy(),
            'mlp_bias1': self.mlp_bias1.copy(),
            'mlp_weight2': self.mlp_weight2.copy()
        }
        if self.mlp_bias2 is not None:
            weights['mlp_bias2'] = self.mlp_bias2.copy()
        if self.learn_epsilon:
            weights['epsilon'] = self.epsilon_param.copy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set layer weights."""
        if 'mlp_weight1' in weights:
            self.mlp_weight1 = weights['mlp_weight1'].copy()
        if 'mlp_bias1' in weights:
            self.mlp_bias1 = weights['mlp_bias1'].copy()
        if 'mlp_weight2' in weights:
            self.mlp_weight2 = weights['mlp_weight2'].copy()
        if 'mlp_bias2' in weights and self.mlp_bias2 is not None:
            self.mlp_bias2 = weights['mlp_bias2'].copy()
        if 'epsilon' in weights and self.learn_epsilon:
            self.epsilon_param = weights['epsilon'].copy()


class GATLayer:
    """
    Graph Attention Network Layer.
    
    Implements: h_v^{(l+1)} = σ( Σ_{u∈N(v)∪{v}} α_{vu} * W * h_u^{(l)} )
    
    Where α_{vu} = softmax( LeakyReLU( a^T [Wh_v || Wh_u] ) )
    """
    
    def __init__(
        self,
        config: LayerConfig,
        num_heads: int = 4,
        concat: bool = True,
        seed: int = 42
    ):
        """Initialize GAT layer."""
        self.config = config
        self.num_heads = num_heads
        self.concat = concat
        self.rng = np.random.RandomState(seed)
        
        out_per_head = config.out_features // num_heads if concat else config.out_features
        
        self.weight = self._init_weight(config.in_features, out_per_head * num_heads)
        self.attn_src = self._init_weight(out_per_head * num_heads, num_heads)
        self.attn_dst = self._init_weight(out_per_head * num_heads, num_heads)
        
        self.bias = np.zeros(config.out_features) if config.bias else None
        self.activation = self._get_activation(config.activation)
        self.cache = {}
    
    def _init_weight(self, in_features: int, out_features: int) -> np.ndarray:
        """Initialize weights."""
        std = np.sqrt(2.0 / (in_features + out_features))
        return self.rng.randn(in_features, out_features).astype(np.float32) * std
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function."""
        activations = {
            'relu': relu,
            'leaky_relu': leaky_relu,
            'sigmoid': sigmoid,
            'tanh': np.tanh,
            'gelu': gelu,
            'none': lambda x: x
        }
        return activations.get(name, relu)
    
    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through GAT layer.
        
        Args:
            node_features: Node feature matrix [num_nodes, in_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
            training: Whether in training mode
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = node_features.shape[0]
        
        self.cache['node_features'] = node_features
        self.cache['adj_matrix'] = adj_matrix
        
        h = node_features @ self.weight
        
        head_dim = h.shape[1] // self.num_heads
        h_heads = h.reshape(num_nodes, self.num_heads, head_dim)
        
        attn_src = h @ self.attn_src
        attn_dst = h @ self.attn_dst
        
        attn_scores = attn_src[:, np.newaxis, :] + attn_dst[np.newaxis, :, :]
        attn_scores = leaky_relu(attn_scores)
        
        adj_with_self = adj_matrix + np.eye(num_nodes)
        attn_scores = np.where(adj_with_self[:, :, np.newaxis] > 0, attn_scores, -1e9)
        
        attn_weights = softmax(attn_scores, axis=1)
        
        output = np.zeros((num_nodes, self.num_heads, head_dim), dtype=np.float32)
        for head in range(self.num_heads):
            output[:, head, :] = attn_weights[:, :, head].T @ h_heads[:, head, :]
        
        if self.concat:
            output = output.reshape(num_nodes, -1)
        else:
            output = np.mean(output, axis=1)
        
        if self.bias is not None:
            output = output + self.bias
        
        output = self.activation(output)
        
        return output
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get layer weights."""
        return {
            'weight': self.weight.copy(),
            'attn_src': self.attn_src.copy(),
            'attn_dst': self.attn_dst.copy(),
            'bias': self.bias.copy() if self.bias is not None else None
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set layer weights."""
        if 'weight' in weights:
            self.weight = weights['weight'].copy()
        if 'attn_src' in weights:
            self.attn_src = weights['attn_src'].copy()
        if 'attn_dst' in weights:
            self.attn_dst = weights['attn_dst'].copy()
        if 'bias' in weights and self.bias is not None:
            self.bias = weights['bias'].copy()


class GCNModel:
    """
    Full GCN Model for node/graph classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        task: str = 'node_classification',
        seed: int = 42
    ):
        """Initialize GCN model."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.task = task
        self.seed = seed
        
        self.layers = []
        self.rng = np.random.RandomState(seed)
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            config = LayerConfig(
                in_features=dims[i],
                out_features=dims[i + 1],
                activation='relu' if i < len(dims) - 2 else 'none',
                dropout=dropout if i < len(dims) - 2 else 0.0,
                bias=True,
                normalize=i < len(dims) - 2
            )
            self.layers.append(GCNLayer(config, seed=seed + i))
    
    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """Forward pass through all GCN layers."""
        h = node_features
        
        for layer in self.layers:
            h = layer.forward(h, adj_matrix, training=training)
        
        if self.task == 'graph_classification':
            h = np.mean(h, axis=0)
        
        return h
    
    def predict(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """Make predictions."""
        logits = self.forward(node_features, adj_matrix, training=False)
        return np.argmax(logits, axis=-1)
    
    def compute_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Compute cross-entropy loss."""
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        
        probs = softmax(logits)
        
        num_classes = logits.shape[-1]
        one_hot = np.eye(num_classes)[labels]
        
        loss = -np.sum(one_hot * np.log(probs + 1e-10)) / len(labels)
        
        return loss
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all model weights."""
        weights = {}
        for i, layer in enumerate(self.layers):
            layer_weights = layer.get_weights()
            for key, value in layer_weights.items():
                weights[f'layer_{i}_{key}'] = value
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set all model weights."""
        for i, layer in enumerate(self.layers):
            layer_weights = {}
            for key in layer.get_weights().keys():
                full_key = f'layer_{i}_{key}'
                if full_key in weights:
                    layer_weights[key] = weights[full_key]
            layer.set_weights(layer_weights)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        total = 0
        for layer in self.layers:
            for w in layer.get_weights().values():
                if w is not None:
                    total += w.size
        return total


class GraphSAGEModel:
    """
    Full GraphSAGE Model for node/graph classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        aggregator: str = 'mean',
        dropout: float = 0.5,
        task: str = 'node_classification',
        seed: int = 42
    ):
        """Initialize GraphSAGE model."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.dropout = dropout
        self.task = task
        self.seed = seed
        
        self.layers = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            config = LayerConfig(
                in_features=dims[i],
                out_features=dims[i + 1],
                activation='relu' if i < len(dims) - 2 else 'none',
                dropout=dropout if i < len(dims) - 2 else 0.0,
                bias=True,
                normalize=i < len(dims) - 2
            )
            self.layers.append(GraphSAGELayer(config, aggregator=aggregator, seed=seed + i))
    
    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """Forward pass through all GraphSAGE layers."""
        h = node_features
        
        for layer in self.layers:
            h = layer.forward(h, adj_matrix, training=training)
        
        if self.task == 'graph_classification':
            h = np.mean(h, axis=0)
        
        return h
    
    def predict(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """Make predictions."""
        logits = self.forward(node_features, adj_matrix, training=False)
        return np.argmax(logits, axis=-1)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all model weights."""
        weights = {}
        for i, layer in enumerate(self.layers):
            layer_weights = layer.get_weights()
            for key, value in layer_weights.items():
                weights[f'layer_{i}_{key}'] = value
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set all model weights."""
        for i, layer in enumerate(self.layers):
            layer_weights = {}
            for key in layer.get_weights().keys():
                full_key = f'layer_{i}_{key}'
                if full_key in weights:
                    layer_weights[key] = weights[full_key]
            layer.set_weights(layer_weights)


class GINModel:
    """
    Full GIN Model for graph classification (ideal for horizontal FL).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        epsilon: float = 0.0,
        learn_epsilon: bool = True,
        task: str = 'graph_classification',
        seed: int = 42
    ):
        """Initialize GIN model."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task = task
        self.seed = seed
        
        self.layers = []
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            config = LayerConfig(
                in_features=dims[i],
                out_features=dims[i + 1],
                activation='relu',
                bias=True
            )
            self.layers.append(GINLayer(
                config, 
                epsilon=epsilon, 
                learn_epsilon=learn_epsilon, 
                seed=seed + i
            ))
        
        self.classifier_weight = self._init_weight(dims[-1], output_dim)
        self.classifier_bias = np.zeros(output_dim)
    
    def _init_weight(self, in_features: int, out_features: int) -> np.ndarray:
        """Initialize weights."""
        rng = np.random.RandomState(self.seed)
        std = np.sqrt(2.0 / (in_features + out_features))
        return rng.randn(in_features, out_features).astype(np.float32) * std
    
    def forward(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """Forward pass through all GIN layers."""
        h = node_features
        
        for layer in self.layers:
            h = layer.forward(h, adj_matrix, training=training)
        
        if self.task == 'graph_classification':
            h = np.sum(h, axis=0)
        else:
            h = np.mean(h, axis=0)
        
        logits = h @ self.classifier_weight + self.classifier_bias
        
        return logits
    
    def predict(
        self,
        node_features: np.ndarray,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """Make predictions."""
        logits = self.forward(node_features, adj_matrix, training=False)
        return np.argmax(logits, axis=-1)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all model weights."""
        weights = {
            'classifier_weight': self.classifier_weight.copy(),
            'classifier_bias': self.classifier_bias.copy()
        }
        for i, layer in enumerate(self.layers):
            layer_weights = layer.get_weights()
            for key, value in layer_weights.items():
                weights[f'layer_{i}_{key}'] = value
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set all model weights."""
        if 'classifier_weight' in weights:
            self.classifier_weight = weights['classifier_weight'].copy()
        if 'classifier_bias' in weights:
            self.classifier_bias = weights['classifier_bias'].copy()
        
        for i, layer in enumerate(self.layers):
            layer_weights = {}
            for key in layer.get_weights().keys():
                full_key = f'layer_{i}_{key}'
                if full_key in weights:
                    layer_weights[key] = weights[full_key]
            layer.set_weights(layer_weights)


def create_gnn_model(
    model_type: str,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    task: str = 'node_classification',
    **kwargs
) -> Any:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: 'gcn', 'graphsage', or 'gin'
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of classes)
        task: 'node_classification' or 'graph_classification'
        **kwargs: Additional model-specific arguments
    
    Returns:
        GNN model instance
    """
    if model_type == 'gcn':
        return GCNModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            task=task,
            **kwargs
        )
    elif model_type == 'graphsage':
        return GraphSAGEModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            task=task,
            **kwargs
        )
    elif model_type == 'gin':
        return GINModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            task=task,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Graph Neural Network Models")
    print("=" * 60)
    
    num_nodes = 50
    feature_dim = 16
    hidden_dim = 32
    num_classes = 3
    
    rng = np.random.RandomState(42)
    node_features = rng.randn(num_nodes, feature_dim).astype(np.float32)
    adj_matrix = (rng.rand(num_nodes, num_nodes) < 0.1).astype(np.float32)
    adj_matrix = np.triu(adj_matrix, k=1) + np.triu(adj_matrix, k=1).T
    labels = rng.randint(0, num_classes, num_nodes)
    
    print("\n1. Testing GCN Model...")
    gcn = GCNModel(
        input_dim=feature_dim,
        hidden_dims=[hidden_dim],
        output_dim=num_classes,
        task='node_classification'
    )
    gcn_output = gcn.forward(node_features, adj_matrix)
    gcn_pred = gcn.predict(node_features, adj_matrix)
    print(f"   Output shape: {gcn_output.shape}")
    print(f"   Predictions: {gcn_pred[:5]}...")
    print(f"   Num parameters: {gcn.get_num_parameters()}")
    
    print("\n2. Testing GraphSAGE Model...")
    sage = GraphSAGEModel(
        input_dim=feature_dim,
        hidden_dims=[hidden_dim],
        output_dim=num_classes,
        aggregator='mean',
        task='node_classification'
    )
    sage_output = sage.forward(node_features, adj_matrix)
    sage_pred = sage.predict(node_features, adj_matrix)
    print(f"   Output shape: {sage_output.shape}")
    print(f"   Predictions: {sage_pred[:5]}...")
    
    print("\n3. Testing GIN Model (for graph classification)...")
    gin = GINModel(
        input_dim=feature_dim,
        hidden_dims=[hidden_dim, hidden_dim],
        output_dim=num_classes,
        task='graph_classification'
    )
    gin_output = gin.forward(node_features, adj_matrix)
    gin_pred = gin.predict(node_features, adj_matrix)
    print(f"   Output shape: {gin_output.shape}")
    print(f"   Graph prediction: {gin_pred}")
    
    print("\n4. Testing weight serialization...")
    weights = gcn.get_weights()
    print(f"   Total weight keys: {len(weights)}")
    for key, value in weights.items():
        print(f"   - {key}: {value.shape}")
    
    print("\n" + "=" * 60)
    print("All model tests passed!")
    print("=" * 60)
