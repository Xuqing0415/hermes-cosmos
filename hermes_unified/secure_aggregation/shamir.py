"""
Shamir Secret Sharing for Secure Aggregation

Implements (t, n)-threshold secret sharing.
"""

import numpy as np
from typing import List, Tuple, Dict
import random
import logging

logger = logging.getLogger(__name__)


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing Scheme.
    
    Implements (t, n)-threshold secret sharing where:
    - n: total number of shares
    - t: threshold required to reconstruct
    """
    
    def __init__(self, threshold: int = 3, prime: int = 2**31 - 1):
        """
        Initialize Shamir secret sharing.
        
        Args:
            threshold: Number of shares required to reconstruct
            prime: Prime modulus for finite field arithmetic
        """
        self.threshold = threshold
        self.prime = prime
        
    def share_scalar(self, secret: float, num_shares: int) -> List[Tuple[int, float]]:
        """
        Split a scalar secret into shares.
        
        Args:
            secret: Scalar value to share
            num_shares: Number of shares to generate
        
        Returns:
            List of (share_id, share_value)
        """
        # Generate random polynomial coefficients
        coefficients = [secret] + [random.randint(1, self.prime - 1) 
                                  for _ in range(self.threshold - 1)]
        
        # Generate shares
        shares = []
        for i in range(1, num_shares + 1):
            # Evaluate polynomial at x = i
            share_value = 0
            x = i
            for deg, coeff in enumerate(coefficients):
                share_value = (share_value + coeff * (x ** deg)) % self.prime
            
            shares.append((i, share_value))
        
        return shares
    
    def reconstruct_scalar(self, shares: List[Tuple[int, float]]) -> float:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (share_id, share_value)
        
        Returns:
            Reconstructed secret
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
            
            # Modular inverse
            denominator_inv = self._modular_inverse(denominator)
            lagrange_basis = (numerator * denominator_inv) % self.prime
            
            secret = (secret + y_i * lagrange_basis) % self.prime
        
        return float(secret)
    
    def _modular_inverse(self, a: int) -> int:
        """
        Compute modular inverse using Fermat's little theorem.
        """
        return pow(a, self.prime - 2, self.prime)
    
    def share_array(self, arr: np.ndarray, num_shares: int) -> List[np.ndarray]:
        """
        Share an array, element-wise.
        
        Args:
            arr: Array to share
            num_shares: Number of shares
        
        Returns:
            List of shared arrays, one per client
        """
        shares = [np.zeros_like(arr) for _ in range(num_shares)]
        
        # Share each element
        for idx in np.ndindex(arr.shape):
            scalar_shares = self.share_scalar(float(arr[idx]), num_shares)
            for share_id, (_, val) in enumerate(scalar_shares):
                shares[share_id][idx] = val
        
        return shares
    
    def reconstruct_array(self, share_arrays: List[np.ndarray], 
                        share_ids: List[int] = None) -> np.ndarray:
        """
        Reconstruct array from shares.
        
        Args:
            share_arrays: List of array shares
            share_ids: Optional list of share IDs
        
        Returns:
            Reconstructed array
        """
        if share_ids is None:
            share_ids = list(range(1, len(share_arrays) + 1))
        
        result = np.zeros_like(share_arrays[0])
        
        for idx in np.ndindex(result.shape):
            scalar_shares = [(share_ids[i], share_arrays[i][idx]) 
                           for i in range(len(share_arrays))]
            result[idx] = self.reconstruct_scalar(scalar_shares)
        
        return result


class SimplifiedShamirAggregator:
    """
    Simplified Shamir-based secure aggregator for demonstration.
    
    Uses additive secret sharing for practicality (faster).
    """
    
    def __init__(self, num_clients: int, threshold: int = 3):
        self.num_clients = num_clients
        self.threshold = threshold
        self.client_shares = {}
        
    def client_split_update(self, client_id: int, update: np.ndarray) -> List[np.ndarray]:
        """
        Client-side: split update into shares.
        
        Args:
            client_id: Client ID
            update: Model update array
        
        Returns:
            List of shares (one for each client)
        """
        shares = []
        remaining = update.copy()
        
        # Generate random shares
        for _ in range(self.num_clients - 1):
            share = np.random.randn(*update.shape) * 0.1
            shares.append(share)
            remaining -= share
        
        shares.append(remaining)
        self.client_shares[client_id] = shares
        
        return shares
    
    def client_receive_shares(self, client_id: int, received_shares: Dict[int, np.ndarray]):
        """
        Client-side: receive shares from other clients.
        """
        self.client_shares[client_id] = (self.client_shares.get(client_id, []), 
                                        received_shares)
    
    def aggregate_shares(self, shares_list: List[np.ndarray]) -> np.ndarray:
        """
        Server-side: aggregate shares and reconstruct.
        
        Args:
            shares_list: List of shares from clients
        
        Returns:
            Aggregated update
        """
        aggregated = np.sum(shares_list, axis=0)
        return aggregated
