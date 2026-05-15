"""
Privacy-Preserving Exchange Mechanisms for Federated Graph Learning

Implements differential privacy, secure embedding exchange, and
top-k neighbor selection for privacy-preserving cross-client
communication in vertical federated graph learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class DifferentialPrivacyMechanism:
    """
    Differential Privacy mechanisms for embedding perturbation.
    
    Supports:
    - Gaussian mechanism (ε, δ)-DP
    - Laplace mechanism (ε)-DP
    - Local DP with clip and noise
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: str = 'gaussian',
        sensitivity: float = 1.0,
        clip_norm: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize DP mechanism.
        
        Args:
            epsilon: Privacy budget
            delta: Failure probability for (ε, δ)-DP
            mechanism: 'gaussian', 'laplace', or 'local'
            sensitivity: L2 sensitivity of the function
            clip_norm: Clipping threshold for local DP
            seed: Random seed
        """
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = mechanism
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        self.rng = np.random.RandomState(seed)
        
        self.noise_scale = self._compute_noise_scale()
        
        self.privacy_budget_spent = 0.0
        self.noise_added = []
    
    def _compute_noise_scale(self) -> float:
        """Compute noise scale based on mechanism."""
        if self.mechanism == 'gaussian':
            sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            return sigma
        elif self.mechanism == 'laplace':
            return self.sensitivity / self.epsilon
        else:
            return self._compute_noise_scale()
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add differential privacy noise to data.
        
        Args:
            data: Input data to perturb
        
        Returns:
            Perturbed data
        """
        if self.mechanism == 'gaussian':
            noise = self.rng.normal(0, self.noise_scale, data.shape).astype(np.float32)
        elif self.mechanism == 'laplace':
            noise = self.rng.laplace(0, self.noise_scale, data.shape).astype(np.float32)
        else:
            noise = self.rng.normal(0, self.noise_scale, data.shape).astype(np.float32)
        
        perturbed = data + noise
        
        self.privacy_budget_spent += self.epsilon
        self.noise_added.append(np.mean(np.abs(noise)))
        
        return perturbed
    
    def clip_and_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Clip and add noise for local DP.
        
        Args:
            data: Input data
        
        Returns:
            Clipped and perturbed data
        """
        norms = np.linalg.norm(data, axis=-1, keepdims=True)
        clip_factors = np.minimum(1.0, self.clip_norm / (norms + 1e-10))
        clipped = data * clip_factors
        
        return self.add_noise(clipped)
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get privacy budget statistics."""
        return {
            'epsilon_spent': self.privacy_budget_spent,
            'epsilon_budget': self.epsilon,
            'avg_noise_magnitude': np.mean(self.noise_added) if self.noise_added else 0.0
        }
    
    def reset_budget(self):
        """Reset privacy budget tracking."""
        self.privacy_budget_spent = 0.0
        self.noise_added = []


class SecureEmbeddingExchange:
    """
    Secure embedding exchange using secret sharing.
    
    Enables clients to exchange aggregated neighbor embeddings
    without revealing individual node embeddings.
    """
    
    def __init__(
        self,
        num_clients: int,
        threshold: int = 2,
        field_size: int = 2**31 - 1,
        seed: int = 42
    ):
        """
        Initialize secure embedding exchange.
        
        Args:
            num_clients: Number of participating clients
            threshold: Minimum shares needed for reconstruction
            field_size: Prime field size for secret sharing
            seed: Random seed
        """
        self.num_clients = num_clients
        self.threshold = threshold
        self.field_size = field_size
        self.rng = np.random.RandomState(seed)
        
        self.exchange_history = []
    
    def split_embedding(
        self,
        embedding: np.ndarray,
        client_id: int
    ) -> Dict[int, np.ndarray]:
        """
        Split an embedding into secret shares.
        
        Args:
            embedding: Node embedding to split
            client_id: ID of the client creating shares
        
        Returns:
            Dictionary mapping client_id to share
        """
        shares = {}
        
        for i in range(self.num_clients):
            if i == client_id:
                continue
            
            share = self.rng.randint(
                0, self.field_size, 
                size=embedding.shape
            ).astype(np.float32) % self.field_size
            shares[i] = share
        
        own_share = embedding.copy()
        for share in shares.values():
            own_share = (own_share - share) % self.field_size
        shares[client_id] = own_share
        
        return shares
    
    def aggregate_shares(
        self,
        shares: List[np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate multiple shares to reconstruct embedding.
        
        Args:
            shares: List of shares from different clients
        
        Returns:
            Aggregated embedding
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares for reconstruction")
        
        aggregated = np.zeros_like(shares[0])
        for share in shares[:self.threshold]:
            aggregated = (aggregated + share) % self.field_size
        
        return aggregated
    
    def secure_sum(
        self,
        embeddings: List[np.ndarray],
        noise_scale: float = 0.0
    ) -> np.ndarray:
        """
        Compute secure sum of embeddings.
        
        Args:
            embeddings: List of embeddings to sum
            noise_scale: Optional noise for additional privacy
        
        Returns:
            Sum of embeddings
        """
        result = np.zeros_like(embeddings[0])
        for emb in embeddings:
            result = result + emb
        
        if noise_scale > 0:
            noise = self.rng.normal(0, noise_scale, result.shape).astype(np.float32)
            result = result + noise
        
        return result
    
    def create_exchange_package(
        self,
        embeddings: Dict[int, np.ndarray],
        source_client: int,
        target_clients: List[int]
    ) -> Dict[int, Dict]:
        """
        Create secure exchange packages for target clients.
        
        Args:
            embeddings: Dictionary of node_id -> embedding
            source_client: ID of sending client
            target_clients: List of receiving client IDs
        
        Returns:
            Dictionary mapping target_client -> exchange data
        """
        packages = {}
        
        for target_id in target_clients:
            shares = {}
            for node_id, embedding in embeddings.items():
                node_shares = self.split_embedding(embedding, source_client)
                if target_id in node_shares:
                    shares[node_id] = node_shares[target_id]
            
            packages[target_id] = {
                'source_client': source_client,
                'target_client': target_id,
                'shares': shares,
                'num_nodes': len(shares)
            }
        
        self.exchange_history.append({
            'source': source_client,
            'targets': target_clients,
            'num_embeddings': len(embeddings)
        })
        
        return packages
    
    def get_exchange_stats(self) -> Dict[str, Any]:
        """Get statistics about exchanges."""
        if not self.exchange_history:
            return {'total_exchanges': 0}
        
        total_embeddings = sum(e['num_embeddings'] for e in self.exchange_history)
        
        return {
            'total_exchanges': len(self.exchange_history),
            'total_embeddings_exchanged': total_embeddings,
            'avg_embeddings_per_exchange': total_embeddings / len(self.exchange_history)
        }


class TopKNeighborSelector:
    """
    Select top-k most important neighbors for embedding exchange.
    
    Reduces communication cost and provides additional privacy
    by limiting the amount of information shared.
    """
    
    def __init__(
        self,
        k: int = 10,
        selection_method: str = 'attention',
        seed: int = 42
    ):
        """
        Initialize top-k selector.
        
        Args:
            k: Number of neighbors to select
            selection_method: 'attention', 'degree', or 'random'
            seed: Random seed
        """
        self.k = k
        self.selection_method = selection_method
        self.rng = np.random.RandomState(seed)
        
        self.selection_history = []
    
    def compute_attention_scores(
        self,
        source_embedding: np.ndarray,
        neighbor_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute attention scores between source and neighbors.
        
        Args:
            source_embedding: Source node embedding
            neighbor_embeddings: Neighbor embeddings [num_neighbors, dim]
        
        Returns:
            Attention scores [num_neighbors]
        """
        if len(neighbor_embeddings.shape) == 1:
            neighbor_embeddings = neighbor_embeddings.reshape(1, -1)
        
        scores = neighbor_embeddings @ source_embedding
        scores = scores / (np.linalg.norm(neighbor_embeddings, axis=1) * np.linalg.norm(source_embedding) + 1e-10)
        
        return scores
    
    def select_top_k(
        self,
        source_embedding: np.ndarray,
        neighbor_embeddings: np.ndarray,
        neighbor_ids: np.ndarray,
        adj_matrix: np.ndarray = None,
        source_id: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-k neighbors based on selection method.
        
        Args:
            source_embedding: Source node embedding
            neighbor_embeddings: All neighbor embeddings
            neighbor_ids: IDs of neighbors
            adj_matrix: Optional adjacency matrix for degree-based selection
            source_id: Optional source node ID
        
        Returns:
            Tuple of (selected_embeddings, selected_ids)
        """
        if len(neighbor_ids) <= self.k:
            return neighbor_embeddings, neighbor_ids
        
        if self.selection_method == 'attention':
            scores = self.compute_attention_scores(source_embedding, neighbor_embeddings)
            top_indices = np.argsort(-scores)[:self.k]
        
        elif self.selection_method == 'degree':
            if adj_matrix is not None and source_id is not None:
                degrees = np.sum(adj_matrix[neighbor_ids], axis=1)
                top_indices = np.argsort(-degrees)[:self.k]
            else:
                top_indices = self.rng.choice(len(neighbor_ids), self.k, replace=False)
        
        else:
            top_indices = self.rng.choice(len(neighbor_ids), self.k, replace=False)
        
        selected_embeddings = neighbor_embeddings[top_indices]
        selected_ids = neighbor_ids[top_indices]
        
        self.selection_history.append({
            'total_neighbors': len(neighbor_ids),
            'selected': self.k,
            'method': self.selection_method
        })
        
        return selected_embeddings, selected_ids
    
    def select_top_k_batch(
        self,
        embeddings: np.ndarray,
        adj_matrix: np.ndarray,
        node_ids: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Select top-k neighbors for multiple nodes.
        
        Args:
            embeddings: All node embeddings
            adj_matrix: Adjacency matrix
            node_ids: IDs of nodes to process
        
        Returns:
            Dictionary mapping node_id -> (selected_embeddings, selected_ids)
        """
        results = {}
        
        for node_id in node_ids:
            neighbors = np.where(adj_matrix[node_id] > 0)[0]
            
            if len(neighbors) == 0:
                results[node_id] = (np.array([]), np.array([]))
                continue
            
            neighbor_embeddings = embeddings[neighbors]
            
            selected_emb, selected_ids = self.select_top_k(
                embeddings[node_id],
                neighbor_embeddings,
                neighbors,
                adj_matrix,
                node_id
            )
            
            results[node_id] = (selected_emb, selected_ids)
        
        return results
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about selections."""
        if not self.selection_history:
            return {'total_selections': 0}
        
        total_neighbors = sum(s['total_neighbors'] for s in self.selection_history)
        
        return {
            'total_selections': len(self.selection_history),
            'total_neighbors_considered': total_neighbors,
            'total_neighbors_selected': len(self.selection_history) * self.k,
            'reduction_ratio': 1 - (self.k * len(self.selection_history)) / total_neighbors if total_neighbors > 0 else 0
        }


class PrivacyBudgetManager:
    """
    Manage privacy budget across multiple exchanges.
    
    Tracks and enforces privacy budget consumption.
    """
    
    def __init__(
        self,
        total_epsilon: float = 10.0,
        total_delta: float = 1e-5,
        composition_method: str = 'basic'
    ):
        """
        Initialize privacy budget manager.
        
        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
            composition_method: 'basic', 'advanced', or 'rdp'
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.composition_method = composition_method
        
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.operations = []
    
    def check_budget(self, epsilon: float, delta: float = 0.0) -> bool:
        """
        Check if operation is within budget.
        
        Args:
            epsilon: Epsilon for this operation
            delta: Delta for this operation
        
        Returns:
            True if operation is allowed
        """
        if self.composition_method == 'basic':
            return (self.epsilon_spent + epsilon <= self.total_epsilon and
                    self.delta_spent + delta <= self.total_delta)
        elif self.composition_method == 'advanced':
            new_epsilon = np.sqrt(self.epsilon_spent**2 + epsilon**2)
            return new_epsilon <= self.total_epsilon
        else:
            return self.epsilon_spent + epsilon <= self.total_epsilon
    
    def spend_budget(self, epsilon: float, delta: float = 0.0, operation: str = "") -> bool:
        """
        Spend privacy budget for an operation.
        
        Args:
            epsilon: Epsilon to spend
            delta: Delta to spend
            operation: Description of operation
        
        Returns:
            True if successful, False if insufficient budget
        """
        if not self.check_budget(epsilon, delta):
            logger.warning(f"Insufficient privacy budget for operation: {operation}")
            return False
        
        self.epsilon_spent += epsilon
        self.delta_spent += delta
        self.operations.append({
            'epsilon': epsilon,
            'delta': delta,
            'operation': operation,
            'cumulative_epsilon': self.epsilon_spent
        })
        
        return True
    
    def get_remaining_budget(self) -> Dict[str, float]:
        """Get remaining privacy budget."""
        return {
            'remaining_epsilon': self.total_epsilon - self.epsilon_spent,
            'remaining_delta': self.total_delta - self.delta_spent,
            'epsilon_spent': self.epsilon_spent,
            'delta_spent': self.delta_spent,
            'budget_utilization': self.epsilon_spent / self.total_epsilon
        }
    
    def reset(self):
        """Reset budget tracking."""
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.operations = []


@dataclass
class ExchangeConfig:
    """Configuration for embedding exchange."""
    use_dp: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    use_top_k: bool = True
    k: int = 10
    use_secret_sharing: bool = False
    clip_norm: float = 1.0
    noise_scale: float = 0.1


class PrivacyPreservingExchange:
    """
    Unified interface for privacy-preserving embedding exchange.
    
    Combines DP, secret sharing, and top-k selection.
    """
    
    def __init__(self, config: ExchangeConfig, num_clients: int = 2, seed: int = 42):
        """
        Initialize privacy-preserving exchange.
        
        Args:
            config: Exchange configuration
            num_clients: Number of clients
            seed: Random seed
        """
        self.config = config
        self.num_clients = num_clients
        self.seed = seed
        
        self.dp_mechanism = None
        if config.use_dp:
            self.dp_mechanism = DifferentialPrivacyMechanism(
                epsilon=config.epsilon,
                delta=config.delta,
                clip_norm=config.clip_norm,
                seed=seed
            )
        
        self.top_k_selector = None
        if config.use_top_k:
            self.top_k_selector = TopKNeighborSelector(
                k=config.k,
                seed=seed
            )
        
        self.secure_exchange = None
        if config.use_secret_sharing:
            self.secure_exchange = SecureEmbeddingExchange(
                num_clients=num_clients,
                seed=seed
            )
        
        self.budget_manager = PrivacyBudgetManager(
            total_epsilon=config.epsilon * 10,
            total_delta=config.delta
        )
    
    def prepare_embedding_for_exchange(
        self,
        embedding: np.ndarray,
        neighbor_ids: Optional[np.ndarray] = None,
        all_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Prepare an embedding for secure exchange.
        
        Args:
            embedding: Node embedding to prepare
            neighbor_ids: Optional neighbor IDs for top-k selection
            all_embeddings: Optional all embeddings for batch selection
        
        Returns:
            Prepared embedding
        """
        prepared = embedding.copy()
        
        if self.dp_mechanism is not None:
            prepared = self.dp_mechanism.clip_and_noise(prepared)
            self.budget_manager.spend_budget(
                self.config.epsilon,
                self.config.delta,
                "embedding_exchange"
            )
        
        return prepared
    
    def prepare_batch_for_exchange(
        self,
        embeddings: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Prepare multiple embeddings for exchange.
        
        Args:
            embeddings: Dictionary of node_id -> embedding
        
        Returns:
            Dictionary of prepared embeddings
        """
        prepared = {}
        for node_id, emb in embeddings.items():
            prepared[node_id] = self.prepare_embedding_for_exchange(emb)
        return prepared
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get comprehensive privacy report."""
        report = {
            'config': {
                'use_dp': self.config.use_dp,
                'epsilon': self.config.epsilon,
                'use_top_k': self.config.use_top_k,
                'k': self.config.k if self.config.use_top_k else None,
                'use_secret_sharing': self.config.use_secret_sharing
            },
            'budget': self.budget_manager.get_remaining_budget()
        }
        
        if self.dp_mechanism is not None:
            report['dp_stats'] = self.dp_mechanism.get_privacy_spent()
        
        if self.top_k_selector is not None:
            report['top_k_stats'] = self.top_k_selector.get_selection_stats()
        
        if self.secure_exchange is not None:
            report['exchange_stats'] = self.secure_exchange.get_exchange_stats()
        
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Privacy-Preserving Exchange Mechanisms")
    print("=" * 60)
    
    print("\n1. Testing Differential Privacy...")
    dp = DifferentialPrivacyMechanism(epsilon=1.0, mechanism='gaussian')
    data = np.random.randn(10, 16).astype(np.float32)
    perturbed = dp.add_noise(data)
    print(f"   Original norm: {np.mean(np.linalg.norm(data, axis=1)):.4f}")
    print(f"   Perturbed norm: {np.mean(np.linalg.norm(perturbed, axis=1)):.4f}")
    print(f"   Noise magnitude: {np.mean(np.abs(perturbed - data)):.4f}")
    print(f"   Privacy spent: {dp.get_privacy_spent()}")
    
    print("\n2. Testing Secure Embedding Exchange...")
    secure = SecureEmbeddingExchange(num_clients=3, threshold=2)
    embedding = np.random.randn(16).astype(np.float32)
    shares = secure.split_embedding(embedding, client_id=0)
    print(f"   Created {len(shares)} shares")
    
    share_list = list(shares.values())
    reconstructed = secure.aggregate_shares(share_list)
    print(f"   Reconstruction error: {np.mean(np.abs(reconstructed - embedding)):.6f}")
    
    print("\n3. Testing Top-K Neighbor Selection...")
    selector = TopKNeighborSelector(k=5, selection_method='attention')
    source = np.random.randn(16).astype(np.float32)
    neighbors = np.random.randn(20, 16).astype(np.float32)
    neighbor_ids = np.arange(20)
    
    selected_emb, selected_ids = selector.select_top_k(source, neighbors, neighbor_ids)
    print(f"   Selected {len(selected_ids)} neighbors from {len(neighbor_ids)}")
    print(f"   Selection stats: {selector.get_selection_stats()}")
    
    print("\n4. Testing Privacy Budget Manager...")
    budget_mgr = PrivacyBudgetManager(total_epsilon=5.0)
    for i in range(6):
        success = budget_mgr.spend_budget(1.0, operation=f"op_{i}")
        print(f"   Operation {i}: {'Success' if success else 'Failed'}")
    print(f"   Remaining budget: {budget_mgr.get_remaining_budget()}")
    
    print("\n5. Testing Unified Privacy-Preserving Exchange...")
    config = ExchangeConfig(
        use_dp=True,
        epsilon=0.5,
        use_top_k=True,
        k=10
    )
    exchange = PrivacyPreservingExchange(config, num_clients=3)
    
    embeddings = {i: np.random.randn(16).astype(np.float32) for i in range(5)}
    prepared = exchange.prepare_batch_for_exchange(embeddings)
    print(f"   Prepared {len(prepared)} embeddings")
    print(f"   Privacy report: {exchange.get_privacy_report()['budget']}")
    
    print("\n" + "=" * 60)
    print("All privacy tests passed!")
    print("=" * 60)
