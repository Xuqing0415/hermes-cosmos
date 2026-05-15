"""
Secure Aggregation for Federated Learning

Implements secure aggregation protocols to protect client privacy.
"""

import numpy as np
from typing import List, Dict, Tuple
import random


class SecureAggregator:
    """
    Secure aggregator using secret sharing.
    
    This implementation uses a simplified secret sharing scheme where:
    1. Each client splits their update into shares
    2. Shares are distributed to other clients
    3. Aggregation is performed on shares
    4. Final reconstruction combines shares
    
    Note: This is a simplified version for demonstration.
    For production, consider using MPC libraries like PySyft or TF Encrypted.
    """
    
    def __init__(self, num_clients: int, threshold: int = None):
        """
        Initialize secure aggregator.
        
        Args:
            num_clients: Total number of clients
            threshold: Minimum number of clients needed for reconstruction (default: num_clients // 2 + 1)
        """
        self.num_clients = num_clients
        self.threshold = threshold if threshold else num_clients // 2 + 1
        self.prime = self._find_large_prime()
    
    def _find_large_prime(self) -> int:
        """Find a large prime number for secret sharing."""
        # Use a known large prime for simplicity
        return 10**9 + 7
    
    def share_secret(self, secret: float, num_shares: int) -> List[int]:
        """
        Split a secret into num_shares shares using Shamir's secret sharing.
        
        Args:
            secret: Secret value to split
            num_shares: Number of shares to create
            
        Returns:
            List of shares (integers)
        """
        # Convert float to integer (scaling)
        scaled_secret = int(secret * 10**10) % self.prime
        
        # Generate random coefficients for polynomial
        coeffs = [scaled_secret] + [random.randint(0, self.prime - 1) for _ in range(self.threshold - 1)]
        
        # Evaluate polynomial at points 1, 2, ..., num_shares
        shares = []
        for i in range(1, num_shares + 1):
            share = 0
            for j, coeff in enumerate(coeffs):
                share = (share + coeff * (i ** j)) % self.prime
            shares.append(share)
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> float:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (index, share) tuples
            
        Returns:
            Reconstructed secret value
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares to reconstruct")
        
        secret = 0
        for i, (x_i, y_i) in enumerate(shares):
            # Compute Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime
            
            # Lagrange coefficient
            coeff = (numerator * self._mod_inverse(denominator)) % self.prime
            
            # Add term to secret
            secret = (secret + y_i * coeff) % self.prime
        
        # Convert back to float
        return float(secret) / 10**10
    
    def _mod_inverse(self, a: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm."""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        _, x, _ = extended_gcd(a % self.prime, self.prime)
        return (x % self.prime + self.prime) % self.prime
    
    def aggregate_updates(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Securely aggregate client updates.
        
        Args:
            updates: List of client updates (numpy arrays)
            
        Returns:
            Aggregated update
        """
        if not updates:
            return np.array([])
        
        # Simple secure aggregation: sum with secret sharing
        # For simplicity, we'll use additive secret sharing here
        
        # Initialize aggregated result
        result = np.zeros_like(updates[0])
        
        # For each element in the update
        for i in range(result.size):
            # Collect values from all clients
            values = [update.flat[i] for update in updates]
            
            # Secure sum using secret sharing
            result.flat[i] = self._secure_sum(values)
        
        return result
    
    def _secure_sum(self, values: List[float]) -> float:
        """Compute secure sum using additive secret sharing."""
        num_clients = len(values)
        
        # Each client splits their value into shares
        all_shares = []
        for value in values:
            shares = self.share_secret(value, num_clients)
            all_shares.append(shares)
        
        # Each client sends share i to client i
        # Reconstruct partial sums at each client
        partial_sums = []
        for i in range(num_clients):
            # Client i receives share i from each other client
            received_shares = [(j + 1, all_shares[j][i]) for j in range(num_clients)]
            partial_sum = self.reconstruct_secret(received_shares)
            partial_sums.append(partial_sum)
        
        # Final sum is the sum of partial sums
        return sum(partial_sums)


class FedAvgAggregator:
    """
    FedAvg aggregator with weighted averaging.
    
    Supports:
    - Uniform weighting
    - Size-based weighting (weight by dataset size)
    - FedProx regularization
    """
    
    def __init__(self, use_fedprox: bool = False, mu: float = 0.01):
        """
        Initialize FedAvg aggregator.
        
        Args:
            use_fedprox: Whether to use FedProx regularization
            mu: FedProx regularization parameter
        """
        self.use_fedprox = use_fedprox
        self.mu = mu
    
    def aggregate(self, updates: List[np.ndarray], client_sizes: List[int] = None,
                  global_model: np.ndarray = None) -> np.ndarray:
        """
        Aggregate client updates using FedAvg.
        
        Args:
            updates: List of client updates (delta from global model)
            client_sizes: List of dataset sizes for each client
            global_model: Current global model (needed for FedProx)
            
        Returns:
            Aggregated update
        """
        if not updates:
            return np.array([])
        
        if client_sizes is None:
            # Uniform weighting
            weights = np.ones(len(updates)) / len(updates)
        else:
            # Size-based weighting
            total_size = sum(client_sizes)
            weights = np.array(client_sizes) / total_size
        
        # Compute weighted average
        aggregated = sum(w * update for w, update in zip(weights, updates))
        
        # Apply FedProx regularization if enabled
        if self.use_fedprox and global_model is not None:
            # FedProx adds proximal term to prevent client drift
            # For simplicity, we don't apply it here as it's applied client-side
            pass
        
        return aggregated


class DecentralizedAggregator:
    """
    Decentralized aggregator for gossip-based federated learning.
    
    Each client aggregates updates from neighbors.
    """
    
    def __init__(self, topology: str = 'ring'):
        """
        Initialize decentralized aggregator.
        
        Args:
            topology: Network topology ('ring', 'mesh', 'random')
        """
        self.topology = topology
    
    def aggregate_neighbors(self, local_update: np.ndarray, 
                           neighbor_updates: List[Tuple[int, np.ndarray]],
                           weights: List[float] = None) -> np.ndarray:
        """
        Aggregate local update with neighbor updates.
        
        Args:
            local_update: Local client's update
            neighbor_updates: List of (neighbor_id, update) tuples
            weights: Optional weights for each neighbor
            
        Returns:
            Aggregated update
        """
        if not neighbor_updates:
            return local_update
        
        all_updates = [local_update] + [update for _, update in neighbor_updates]
        
        if weights is None:
            # Equal weighting
            weights = [1.0 / len(all_updates)] * len(all_updates)
        else:
            # Include weight for local update
            weights = [0.5] + [w * 0.5 for w in weights]
            weights = np.array(weights) / sum(weights)
        
        # Compute weighted average
        aggregated = sum(w * update for w, update in zip(weights, all_updates))
        
        return aggregated
