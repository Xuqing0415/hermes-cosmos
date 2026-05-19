"""
Global Negative Buffer for Federated Contrastive Learning

Implements privacy-preserving global feature storage for negative sampling.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import numpy as np


class GlobalNegativeBuffer:
    """Global buffer for storing negative samples across clients."""
    
    def __init__(self, max_size: int = 65536, feature_dim: int = 128, 
                 device: str = 'cpu'):
        """
        Initialize global negative buffer.
        
        Args:
            max_size: Maximum number of features to store
            feature_dim: Dimension of feature vectors
            device: Device to store buffer
        """
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.device = device
        
        self.buffer = torch.randn(max_size, feature_dim).to(device)
        self.buffer = F.normalize(self.buffer, dim=1)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, features: torch.Tensor):
        """
        Add features to the buffer.
        
        Args:
            features: Batch of feature vectors to add
        """
        batch_size = features.size(0)
        features = features.detach().cpu().to(self.device)
        features = F.normalize(features, dim=1)
        
        if self.ptr + batch_size <= self.max_size:
            self.buffer[self.ptr:self.ptr + batch_size] = features
            self.ptr += batch_size
        else:
            remaining = self.max_size - self.ptr
            self.buffer[self.ptr:] = features[:remaining]
            self.buffer[:batch_size - remaining] = features[remaining:]
            self.ptr = batch_size - remaining
        
        self.size = min(self.size + batch_size, self.max_size)
    
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample features from the buffer.
        
        Args:
            batch_size: Number of features to sample
        
        Returns:
            Sampled feature vectors
        """
        if self.size == 0:
            return torch.randn(batch_size, self.feature_dim).to(self.device)
        
        indices = np.random.choice(min(self.size, self.max_size), batch_size, replace=False)
        return self.buffer[indices].clone()
    
    def get_size(self) -> int:
        """Get current number of features in buffer."""
        return min(self.size, self.max_size)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = torch.randn(self.max_size, self.feature_dim).to(self.device)
        self.buffer = F.normalize(self.buffer, dim=1)
        self.ptr = 0
        self.size = 0


class PrivacyPreservingBuffer(GlobalNegativeBuffer):
    """Privacy-preserving global buffer with differential privacy."""
    
    def __init__(self, max_size: int = 65536, feature_dim: int = 128, 
                 device: str = 'cpu', noise_scale: float = 0.1, 
                 forget_rate: float = 0.01):
        """
        Initialize privacy-preserving buffer.
        
        Args:
            max_size: Maximum number of features to store
            feature_dim: Dimension of feature vectors
            device: Device to store buffer
            noise_scale: Standard deviation of Gaussian noise for DP
            forget_rate: Probability of forgetting old features
        """
        super().__init__(max_size, feature_dim, device)
        
        self.noise_scale = noise_scale
        self.forget_rate = forget_rate
        self.privacy_budget = 1.0
    
    def add(self, features: torch.Tensor):
        """
        Add features to buffer with privacy protection.
        
        Args:
            features: Batch of feature vectors to add
        """
        features = features.detach().cpu().to(self.device)
        
        if self.noise_scale > 0:
            noise = torch.randn_like(features) * self.noise_scale
            features = features + noise
        
        features = F.normalize(features, dim=1)
        
        if np.random.random() < self.forget_rate:
            self._forget_random_features()
        
        super().add(features)
        
        self.privacy_budget = max(0.0, self.privacy_budget - 0.001)
    
    def _forget_random_features(self):
        """Randomly forget a portion of the buffer."""
        forget_count = int(self.size * 0.1)
        if forget_count > 0:
            indices = np.random.choice(min(self.size, self.max_size), forget_count, replace=False)
            self.buffer[indices] = torch.randn(forget_count, self.feature_dim).to(self.device)
    
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample features with additional noise.
        
        Args:
            batch_size: Number of features to sample
        
        Returns:
            Sampled feature vectors with noise
        """
        samples = super().sample(batch_size)
        
        if self.noise_scale > 0:
            noise = torch.randn_like(samples) * (self.noise_scale * 0.5)
            samples = samples + noise
        
        return F.normalize(samples, dim=1)
    
    def get_privacy_budget(self) -> float:
        """Get remaining privacy budget."""
        return self.privacy_budget
    
    def reset_privacy_budget(self):
        """Reset privacy budget."""
        self.privacy_budget = 1.0


class FederatedBufferServer:
    """Server-side management for global negative buffer."""
    
    def __init__(self, max_size: int = 65536, feature_dim: int = 128,
                 device: str = 'cpu', use_dp: bool = True):
        """
        Initialize federated buffer server.
        
        Args:
            max_size: Maximum buffer size
            feature_dim: Feature dimension
            device: Storage device
            use_dp: Whether to use differential privacy
        """
        if use_dp:
            self.buffer = PrivacyPreservingBuffer(max_size, feature_dim, device)
        else:
            self.buffer = GlobalNegativeBuffer(max_size, feature_dim, device)
        
        self.client_contributions = {}
        self.total_updates = 0
    
    def receive_features(self, client_id: int, features: torch.Tensor):
        """
        Receive features from a client.
        
        Args:
            client_id: Client identifier
            features: Features to add to buffer
        """
        self.buffer.add(features)
        
        if client_id not in self.client_contributions:
            self.client_contributions[client_id] = 0
        self.client_contributions[client_id] += features.size(0)
        
        self.total_updates += 1
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'size': self.buffer.get_size(),
            'max_size': self.buffer.max_size,
            'feature_dim': self.buffer.feature_dim,
            'total_updates': self.total_updates,
            'client_contributions': self.client_contributions,
            'privacy_budget': self.buffer.get_privacy_budget() if hasattr(self.buffer, 'get_privacy_budget') else None
        }
    
    def aggregate_client_features(self, client_features: List[torch.Tensor]):
        """
        Aggregate features from multiple clients.
        
        Args:
            client_features: List of feature tensors from clients
        """
        for features in client_features:
            self.buffer.add(features)
    
    def distribute_buffer(self, client_id: int) -> torch.Tensor:
        """
        Distribute a subset of the buffer to a client.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Subset of buffer for client
        """
        sample_size = min(1024, self.buffer.get_size())
        return self.buffer.sample(sample_size)


class BufferUpdateManager:
    """Manages efficient updates to the global buffer."""
    
    def __init__(self, buffer_server: FederatedBufferServer, 
                 update_frequency: int = 10):
        """
        Initialize buffer update manager.
        
        Args:
            buffer_server: The buffer server to manage
            update_frequency: Update every N client updates
        """
        self.buffer_server = buffer_server
        self.update_frequency = update_frequency
        self.update_counter = 0
    
    def process_update(self, client_id: int, features: torch.Tensor):
        """
        Process a buffer update from a client.
        
        Args:
            client_id: Client identifier
            features: Features to add
        
        Returns:
            Whether the buffer was updated
        """
        self.buffer_server.receive_features(client_id, features)
        self.update_counter += 1
        
        if self.update_counter >= self.update_frequency:
            self._optimize_buffer()
            self.update_counter = 0
            return True
        
        return False
    
    def _optimize_buffer(self):
        """Optimize buffer by removing redundant features."""
        if self.buffer_server.buffer.size < 1000:
            return
        
        buffer_data = self.buffer_server.buffer.buffer[:self.buffer_server.buffer.size]
        buffer_norm = F.normalize(buffer_data, dim=1)
        
        similarity_matrix = torch.matmul(buffer_norm, buffer_norm.T)
        mask = similarity_matrix > 0.95
        
        redundant_indices = []
        for i in range(buffer_data.size(0)):
            if mask[i].sum() > 10:
                redundant_indices.append(i)
        
        if redundant_indices:
            new_buffer = buffer_data[~torch.tensor(redundant_indices, dtype=bool)]
            self.buffer_server.buffer.buffer[:new_buffer.size(0)] = new_buffer
            self.buffer_server.buffer.size = new_buffer.size(0)
            self.buffer_server.buffer.ptr = new_buffer.size(0)