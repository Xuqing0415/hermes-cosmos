"""
Client Selector for Federated Learning

Implements various client selection strategies for federated learning.
"""

import random
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Callable


class ClientSelector:
    """
    Client selector with multiple selection strategies.
    
    Supported strategies:
    - Random: Uniform random selection
    - Round-robin: Sequential selection
    - Resource-based: Select based on client resources
    - Contribution-based: Select based on past contributions
    - Hybrid: Combine multiple strategies
    """
    
    def __init__(self, strategy: str = 'random'):
        """
        Initialize client selector.
        
        Args:
            strategy: Selection strategy ('random', 'round_robin', 'resource', 'contribution', 'hybrid')
        """
        self.strategy = strategy.lower()
        self.round_robin_index = 0
        self.client_contributions: Dict[int, float] = {}
        self.client_resources: Dict[int, Dict] = {}
        self.selection_history = deque(maxlen=100)
    
    def register_client(self, client_id: int, resources: Optional[Dict] = None):
        """
        Register a client with optional resource information.
        
        Args:
            client_id: Unique client identifier
            resources: Dictionary containing client resources (cpu, memory, network_bandwidth, etc.)
        """
        if resources is None:
            resources = {'cpu': 1.0, 'memory': 1.0, 'bandwidth': 1.0}
        
        self.client_resources[client_id] = resources
        
        if client_id not in self.client_contributions:
            self.client_contributions[client_id] = 0.0
    
    def update_contribution(self, client_id: int, contribution: float):
        """
        Update a client's contribution score.
        
        Args:
            client_id: Client identifier
            contribution: Contribution value (e.g., number of samples processed)
        """
        if client_id in self.client_contributions:
            # EMA update
            self.client_contributions[client_id] = 0.8 * self.client_contributions[client_id] + 0.2 * contribution
    
    def select(self, client_ids: List[int], k: int) -> List[int]:
        """
        Select k clients from the available clients.
        
        Args:
            client_ids: List of available client IDs
            k: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        if not client_ids or k <= 0:
            return []
        
        k = min(k, len(client_ids))
        
        if self.strategy == 'random':
            return self._select_random(client_ids, k)
        elif self.strategy == 'round_robin':
            return self._select_round_robin(client_ids, k)
        elif self.strategy == 'resource':
            return self._select_resource_based(client_ids, k)
        elif self.strategy == 'contribution':
            return self._select_contribution_based(client_ids, k)
        elif self.strategy == 'hybrid':
            return self._select_hybrid(client_ids, k)
        else:
            return self._select_random(client_ids, k)
    
    def _select_random(self, client_ids: List[int], k: int) -> List[int]:
        """Random uniform selection."""
        return random.sample(client_ids, k)
    
    def _select_round_robin(self, client_ids: List[int], k: int) -> List[int]:
        """Round-robin sequential selection."""
        sorted_ids = sorted(client_ids)
        selected = []
        
        for i in range(k):
            idx = (self.round_robin_index + i) % len(sorted_ids)
            selected.append(sorted_ids[idx])
        
        self.round_robin_index = (self.round_robin_index + k) % len(sorted_ids)
        return selected
    
    def _select_resource_based(self, client_ids: List[int], k: int) -> List[int]:
        """Select clients with best resources."""
        # Calculate resource score (higher is better)
        scores = {}
        for cid in client_ids:
            resources = self.client_resources.get(cid, {})
            cpu = resources.get('cpu', 1.0)
            memory = resources.get('memory', 1.0)
            bandwidth = resources.get('bandwidth', 1.0)
            
            # Weighted sum of resources
            score = 0.3 * cpu + 0.3 * memory + 0.4 * bandwidth
            scores[cid] = score
        
        # Sort by score descending and select top k
        sorted_clients = sorted(client_ids, key=lambda cid: scores.get(cid, 0.0), reverse=True)
        return sorted_clients[:k]
    
    def _select_contribution_based(self, client_ids: List[int], k: int) -> List[int]:
        """Select clients with highest past contributions."""
        # Sort by contribution descending
        sorted_clients = sorted(client_ids, 
                               key=lambda cid: self.client_contributions.get(cid, 0.0),
                               reverse=True)
        return sorted_clients[:k]
    
    def _select_hybrid(self, client_ids: List[int], k: int) -> List[int]:
        """Hybrid selection combining random and resource-based."""
        # 50% random, 50% resource-based
        n_random = k // 2
        n_resource = k - n_random
        
        random_selected = self._select_random(client_ids, n_random)
        remaining = [cid for cid in client_ids if cid not in random_selected]
        
        if remaining:
            resource_selected = self._select_resource_based(remaining, n_resource)
            return random_selected + resource_selected
        
        return random_selected
    
    def get_stats(self) -> Dict:
        """Get selection statistics."""
        return {
            'strategy': self.strategy,
            'total_clients': len(self.client_resources),
            'selection_history_length': len(self.selection_history),
            'avg_contribution': np.mean(list(self.client_contributions.values())) if self.client_contributions else 0.0
        }


class ClientSampler:
    """
    Client sampler for federated learning with sampling probability.
    
    Supports:
    - Uniform sampling
    - Size-based sampling (probability proportional to dataset size)
    """
    
    def __init__(self, sampling_method: str = 'uniform'):
        """
        Initialize client sampler.
        
        Args:
            sampling_method: 'uniform' or 'size_based'
        """
        self.sampling_method = sampling_method
        self.client_sizes: Dict[int, int] = {}
    
    def set_client_sizes(self, client_sizes: Dict[int, int]):
        """
        Set client dataset sizes.
        
        Args:
            client_sizes: Dict mapping client_id to dataset size
        """
        self.client_sizes = client_sizes
    
    def sample(self, client_ids: List[int], k: int) -> List[int]:
        """
        Sample k clients.
        
        Args:
            client_ids: List of available client IDs
            k: Number of clients to sample
            
        Returns:
            List of sampled client IDs
        """
        if not client_ids or k <= 0:
            return []
        
        k = min(k, len(client_ids))
        
        if self.sampling_method == 'uniform':
            return random.sample(client_ids, k)
        elif self.sampling_method == 'size_based':
            return self._sample_size_based(client_ids, k)
        else:
            return random.sample(client_ids, k)
    
    def _sample_size_based(self, client_ids: List[int], k: int) -> List[int]:
        """Sample clients with probability proportional to dataset size."""
        total_size = sum(self.client_sizes.get(cid, 1) for cid in client_ids)
        
        if total_size == 0:
            return random.sample(client_ids, k)
        
        # Calculate probabilities
        probabilities = [self.client_sizes.get(cid, 1) / total_size for cid in client_ids]
        
        # Weighted random selection without replacement
        selected = []
        remaining_ids = client_ids.copy()
        remaining_probs = probabilities.copy()
        
        for _ in range(k):
            if not remaining_ids:
                break
            
            # Normalize probabilities
            total = sum(remaining_probs)
            if total == 0:
                probs = [1.0 / len(remaining_ids)] * len(remaining_ids)
            else:
                probs = [p / total for p in remaining_probs]
            
            # Select one client
            idx = np.random.choice(len(remaining_ids), p=probs)
            selected.append(remaining_ids[idx])
            
            # Remove selected from remaining
            del remaining_ids[idx]
            del remaining_probs[idx]
        
        return selected
