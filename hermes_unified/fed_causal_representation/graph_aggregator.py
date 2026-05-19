"""
Causal Graph Aggregator for Federated Causal Discovery

Implements privacy-preserving aggregation of causal graphs from multiple clients.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings


class GraphKernel:
    """Graph kernel for comparing causal graphs."""
    
    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor
    
    def compute(self, adj1: np.ndarray, adj2: np.ndarray) -> float:
        """
        Compute similarity between two adjacency matrices.
        
        Args:
            adj1: First adjacency matrix
            adj2: Second adjacency matrix
        
        Returns:
            Similarity score
        """
        diff = np.abs(adj1 - adj2)
        similarity = np.exp(-self.decay_factor * np.sum(diff))
        return float(similarity)
    
    def compute_kernel_matrix(self, adjacencies: List[np.ndarray]) -> np.ndarray:
        """
        Compute kernel matrix for a list of adjacency matrices.
        
        Args:
            adjacencies: List of adjacency matrices
        
        Returns:
            Kernel matrix
        """
        n = len(adjacencies)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self.compute(adjacencies[i], adjacencies[j])
                K[j, i] = K[i, j]
        
        return K


class CausalGraphAggregator:
    """Aggregator for causal graphs from multiple clients."""
    
    def __init__(self, num_vars: int = 5, aggregation_method: str = 'mean'):
        self.num_vars = num_vars
        self.aggregation_method = aggregation_method
        self.kernel = GraphKernel()
        
        self.client_graphs: Dict[int, np.ndarray] = {}
        self.global_graph: Optional[np.ndarray] = None
    
    def receive_graph(self, client_id: int, adjacency: np.ndarray):
        """
        Receive causal graph from a client.
        
        Args:
            client_id: Client identifier
            adjacency: Adjacency matrix
        """
        self.client_graphs[client_id] = adjacency
    
    def aggregate(self) -> np.ndarray:
        """
        Aggregate client graphs into global causal graph.
        
        Returns:
            Aggregated adjacency matrix
        """
        if not self.client_graphs:
            return np.zeros((self.num_vars, self.num_vars))
        
        graphs = list(self.client_graphs.values())
        
        if self.aggregation_method == 'mean':
            self.global_graph = self._mean_aggregation(graphs)
        elif self.aggregation_method == 'vote':
            self.global_graph = self._vote_aggregation(graphs)
        elif self.aggregation_method == 'kernel':
            self.global_graph = self._kernel_aggregation(graphs)
        else:
            self.global_graph = self._mean_aggregation(graphs)
        
        return self.global_graph
    
    def _mean_aggregation(self, graphs: List[np.ndarray]) -> np.ndarray:
        """Mean-based aggregation."""
        mean_adj = np.mean(graphs, axis=0)
        return (mean_adj > 0.5).astype(float)
    
    def _vote_aggregation(self, graphs: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Voting-based aggregation."""
        stacked = np.stack(graphs, axis=0)
        votes = np.sum(stacked > 0.5, axis=0)
        return (votes > len(graphs) * threshold).astype(float)
    
    def _kernel_aggregation(self, graphs: List[np.ndarray]) -> np.ndarray:
        """Kernel-based weighted aggregation."""
        K = self.kernel.compute_kernel_matrix(graphs)
        
        weights = np.sum(K, axis=1)
        weights = weights / np.sum(weights)
        
        weighted_adj = np.zeros_like(graphs[0])
        for i, adj in enumerate(graphs):
            weighted_adj += weights[i] * adj
        
        return (weighted_adj > 0.5).astype(float)
    
    def get_global_graph(self) -> Optional[np.ndarray]:
        """Get current global causal graph."""
        return self.global_graph
    
    def compute_shd(self, true_adj: np.ndarray) -> int:
        """
        Compute Structural Hamming Distance between global graph and true graph.
        
        Args:
            true_adj: True adjacency matrix
        
        Returns:
            SHD value
        """
        if self.global_graph is None:
            return -1
        
        diff = np.abs(self.global_graph - true_adj)
        return int(np.sum(diff))


class PrivacyPreservingGraphAggregator(CausalGraphAggregator):
    """Privacy-preserving aggregator for causal graphs."""
    
    def __init__(self, num_vars: int = 5, aggregation_method: str = 'mean',
                 noise_scale: float = 0.1, epsilon: float = 1.0):
        super().__init__(num_vars, aggregation_method)
        
        self.noise_scale = noise_scale
        self.epsilon = epsilon
        self.privacy_budget = epsilon
    
    def receive_graph(self, client_id: int, adjacency: np.ndarray):
        """
        Receive and perturb causal graph for privacy.
        
        Args:
            client_id: Client identifier
            adjacency: Adjacency matrix
        """
        noisy_adj = self._add_noise(adjacency)
        self.client_graphs[client_id] = noisy_adj
    
    def _add_noise(self, adjacency: np.ndarray) -> np.ndarray:
        """Add Laplace noise for differential privacy."""
        noise = np.random.laplace(0, self.noise_scale, adjacency.shape)
        noisy_adj = adjacency + noise
        noisy_adj = np.clip(noisy_adj, 0, 1)
        return noisy_adj
    
    def aggregate(self) -> np.ndarray:
        """Aggregate with privacy budget tracking."""
        result = super().aggregate()
        
        self.privacy_budget -= 0.1
        self.privacy_budget = max(0, self.privacy_budget)
        
        return result
    
    def get_privacy_budget(self) -> float:
        """Get remaining privacy budget."""
        return self.privacy_budget


class DistributedPCAlgorithm:
    """Distributed PC algorithm for causal discovery."""
    
    def __init__(self, num_vars: int, alpha: float = 0.05):
        self.num_vars = num_vars
        self.alpha = alpha
        self.skeleton = None
        self.sep_sets = {}
    
    def learn_skeleton_local(self, data: np.ndarray, max_cond_size: int = 3) -> np.ndarray:
        """
        Learn skeleton from local data.
        
        Args:
            data: Local data
            max_cond_size: Maximum conditioning set size
        
        Returns:
            Skeleton adjacency matrix
        """
        n_samples, n_vars = data.shape
        skeleton = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        for cond_size in range(max_cond_size + 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if skeleton[i, j] == 0:
                        continue
                    
                    other_vars = [k for k in range(n_vars) if k != i and k != j]
                    
                    if cond_size > len(other_vars):
                        continue
                    
                    from itertools import combinations
                    for cond_set in combinations(other_vars, cond_size):
                        corr = self._partial_correlation(data, i, j, list(cond_set))
                        
                        if abs(corr) < self.alpha:
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
                            self.sep_sets[(i, j)] = set(cond_set)
                            self.sep_sets[(j, i)] = set(cond_set)
                            break
        
        self.skeleton = skeleton
        return skeleton
    
    def _partial_correlation(self, data: np.ndarray, i: int, j: int, 
                             cond_set: List[int]) -> float:
        """Compute partial correlation."""
        if len(cond_set) == 0:
            return np.corrcoef(data[:, i], data[:, j])[0, 1]
        
        from sklearn.linear_model import LinearRegression
        
        X_cond = data[:, cond_set]
        
        reg_i = LinearRegression().fit(X_cond, data[:, i])
        res_i = data[:, i] - reg_i.predict(X_cond)
        
        reg_j = LinearRegression().fit(X_cond, data[:, j])
        res_j = data[:, j] - reg_j.predict(X_cond)
        
        return np.corrcoef(res_i, res_j)[0, 1]
    
    def orient_edges(self, skeleton: np.ndarray) -> np.ndarray:
        """Orient edges using v-structures."""
        adj = skeleton.copy()
        
        for (i, j), sep_set in self.sep_sets.items():
            for k in sep_set:
                if skeleton[i, k] == 1 and skeleton[j, k] == 1:
                    if k not in self.sep_sets.get((i, j), set()):
                        adj[i, k] = 1
                        adj[k, i] = 0
                        adj[j, k] = 1
                        adj[k, j] = 0
        
        return adj


class FederatedCausalDiscovery:
    """Federated causal discovery combining local discoveries."""
    
    def __init__(self, num_vars: int = 5, alpha: float = 0.05):
        self.num_vars = num_vars
        self.alpha = alpha
        self.aggregator = PrivacyPreservingGraphAggregator(num_vars)
    
    def discover_from_clients(self, client_data: List[np.ndarray]) -> np.ndarray:
        """
        Discover causal graph from multiple clients' data.
        
        Args:
            client_data: List of local data arrays
        
        Returns:
            Aggregated causal graph
        """
        pc = DistributedPCAlgorithm(self.num_vars, self.alpha)
        
        for client_id, data in enumerate(client_data):
            skeleton = pc.learn_skeleton_local(data)
            oriented = pc.orient_edges(skeleton)
            self.aggregator.receive_graph(client_id, oriented)
        
        return self.aggregator.aggregate()
    
    def evaluate_discovery(self, true_adj: np.ndarray) -> Dict[str, float]:
        """
        Evaluate discovered causal graph.
        
        Args:
            true_adj: True adjacency matrix
        
        Returns:
            Evaluation metrics
        """
        discovered = self.aggregator.get_global_graph()
        
        if discovered is None:
            return {'shd': -1, 'precision': 0, 'recall': 0, 'f1': 0}
        
        shd = self.aggregator.compute_shd(true_adj)
        
        tp = np.sum((discovered == 1) & (true_adj == 1))
        fp = np.sum((discovered == 1) & (true_adj == 0))
        fn = np.sum((discovered == 0) & (true_adj == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'shd': shd,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }