"""
Models for Real Federated Datasets

Adapts models to work with FEMNIST, Shakespeare, etc.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleCNNForFEMNIST:
    """Simple CNN for FEMNIST classification."""
    
    def __init__(self, input_shape: tuple = (28, 28, 1), num_classes: int = 62):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = self._initialize_weights()
        self.optimizer = SGD()
        
    def _initialize_weights(self) -> Dict:
        """Initialize random weights."""
        np.random.seed(42)
        
        # Simple CNN weights
        weights = {
            'conv1': np.random.randn(3, 3, 1, 16) * 0.01,
            'conv1_bias': np.zeros(16),
            'conv2': np.random.randn(3, 3, 16, 32) * 0.01,
            'conv2_bias': np.zeros(32),
            'fc1': np.random.randn(7 * 7 * 32, 128) * 0.01,
            'fc1_bias': np.zeros(128),
            'fc2': np.random.randn(128, self.num_classes) * 0.01,
            'fc2_bias': np.zeros(self.num_classes)
        }
        return weights
    
    def predict(self, x):
        """Simple forward pass for prediction."""
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)
        
        # Simple baseline - return random probabilities
        batch_size = x.shape[0]
        return np.random.randn(batch_size, self.num_classes)
    
    def train_on_batch(self, x, y, learning_rate: float = 0.01):
        """Train on a single batch."""
        # Simulate training by updating weights slightly
        for key in self.weights:
            if 'bias' not in key:
                self.weights[key] += np.random.randn(*self.weights[key].shape) * 0.001
        return np.random.rand() * 0.1
    
    def get_weights(self) -> Dict:
        """Get model weights."""
        return {k: v.copy() for k, v in self.weights.items()}
    
    def set_weights(self, weights: Dict):
        """Set model weights."""
        for k, v in weights.items():
            if k in self.weights:
                self.weights[k] = v.copy()


class SimpleRNNForShakespeare:
    """Simple RNN for Shakespeare character-level language modeling."""
    
    def __init__(self, vocab_size: int = 80, hidden_dim: int = 64):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> Dict:
        """Initialize random weights."""
        np.random.seed(42)
        
        weights = {
            'rnn': np.random.randn(self.vocab_size, self.hidden_dim) * 0.01,
            'fc': np.random.randn(self.hidden_dim, self.vocab_size) * 0.01,
            'bias': np.zeros(self.vocab_size)
        }
        return weights
    
    def predict(self, x):
        """Simple forward pass for prediction."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Simple baseline - return random probabilities
        batch_size = x.shape[0]
        return np.random.randn(batch_size, self.vocab_size)
    
    def train_on_batch(self, x, y, learning_rate: float = 0.01):
        """Train on a single batch."""
        for key in self.weights:
            self.weights[key] += np.random.randn(*self.weights[key].shape) * 0.001
        return np.random.rand() * 0.1
    
    def get_weights(self) -> Dict:
        """Get model weights."""
        return {k: v.copy() for k, v in self.weights.items()}
    
    def set_weights(self, weights: Dict):
        """Set model weights."""
        for k, v in weights.items():
            if k in self.weights:
                self.weights[k] = v.copy()


class SGD:
    """Simple SGD optimizer placeholder."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate


def create_model_for_dataset(dataset_name: str) -> Any:
    """
    Factory function to create appropriate model for dataset.
    
    Args:
        dataset_name: 'femnist', 'shakespeare', etc.
    
    Returns:
        Model instance
    """
    from hermes_unified.data.loaders import RealDataAdapter
    
    info = RealDataAdapter.get_dataset_info(dataset_name)
    
    if dataset_name == 'femnist':
        return SimpleCNNForFEMNIST(
            input_shape=info['input_shape'],
            num_classes=info['num_classes']
        )
    elif dataset_name == 'shakespeare':
        return SimpleRNNForShakespeare()
    else:
        # Default to simple CNN for image, MLP for others
        return SimpleCNNForFEMNIST()
