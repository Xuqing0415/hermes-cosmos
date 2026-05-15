"""
Secure Aggregator - Unified Interface

Provides a unified interface for both Shamir Secret Sharing
and Paillier Homomorphic Encryption.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

from hermes_unified.secure_aggregation.shamir import (
    ShamirSecretSharing,
    SimplifiedShamirAggregator
)
from hermes_unified.secure_aggregation.paillier import (
    create_demo_paillier,
    PaillierArrayEncryption
)


class SecureAggregator:
    """
    Unified interface for secure aggregation.
    
    Supports:
    - 'shamir': Shamir Secret Sharing (lightweight)
    - 'paillier': Paillier Homomorphic Encryption (stronger security)
    """
    
    def __init__(self, method: str = 'shamir', num_clients: int = 5, 
                 threshold: int = 3, verbose: bool = True):
        """
        Initialize secure aggregator.
        
        Args:
            method: 'shamir' or 'paillier'
            num_clients: Number of clients
            threshold: Threshold for secret sharing
            verbose: Print timing info
        """
        self.method = method
        self.num_clients = num_clients
        self.threshold = threshold
        self.verbose = verbose
        
        self.client_updates = {}
        
        # Timing stats
        self.stats = {
            'encrypt_time': [],
            'aggregate_time': [],
            'decrypt_time': []
        }
        
        # Initialize backend
        if method == 'shamir':
            self.shamir = ShamirSecretSharing(threshold=threshold)
            self.shamir_aggregator = SimplifiedShamirAggregator(
                num_clients=num_clients,
                threshold=threshold
            )
        elif method == 'paillier':
            self.paillier, self.paillier_arr = create_demo_paillier()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Initialized SecureAggregator with method={method}")
    
    def client_encrypt_update(self, client_id: int, update: np.ndarray) -> Any:
        """
        Client-side: encrypt/update before sending to server.
        
        Args:
            client_id: Client ID
            update: Model update array
        
        Returns:
            Encrypted/shared update
        """
        start_time = time.time()
        
        if self.method == 'shamir':
            # Split into shares
            shares = self.shamir_aggregator.client_split_update(client_id, update)
            encrypted = shares
        elif self.method == 'paillier':
            # Encrypt with Paillier
            encrypted = self.paillier_arr.encrypt_array(update)
        
        self.client_updates[client_id] = encrypted
        
        encrypt_time = time.time() - start_time
        self.stats['encrypt_time'].append(encrypt_time)
        
        if self.verbose:
            logger.info(f"Client {client_id}: Encrypted in {encrypt_time:.4f}s")
        
        return encrypted
    
    def aggregate_encrypted(self, encrypted_updates: Dict[int, Any], 
                           weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Server-side: aggregate encrypted updates.
        
        Args:
            encrypted_updates: {client_id: encrypted_update}
            weights: Optional client weights
        
        Returns:
            Aggregated update (decrypted)
        """
        start_time = time.time()
        
        if self.method == 'shamir':
            # Collect shares and aggregate
            all_shares = []
            client_ids = list(encrypted_updates.keys())
            
            for cid in client_ids:
                shares = encrypted_updates[cid]
                all_shares.append(shares[0])  # Use first share for simplicity
            
            aggregated = self.shamir_aggregator.aggregate_shares(all_shares)
            
        elif self.method == 'paillier':
            # Homomorphic addition
            client_ids = list(encrypted_updates.keys())
            aggregated = encrypted_updates[client_ids[0]]
            
            for cid in client_ids[1:]:
                aggregated = self.paillier_arr.add_encrypted_arrays(
                    aggregated,
                    encrypted_updates[cid]
                )
            
            # Average by multiplying with inverse
            if weights is None:
                weights = [1.0 / len(client_ids)] * len(client_ids)
            
            scale = int(1.0 / (weights[0] * len(client_ids)) + 1e-6)
            aggregated = self.paillier_arr.multiply_by_scalar_array(aggregated, scale)
        
        aggregate_time = time.time() - start_time
        self.stats['aggregate_time'].append(aggregate_time)
        
        if self.verbose:
            logger.info(f"Aggregated {len(encrypted_updates)} updates in {aggregate_time:.4f}s")
        
        return aggregated
    
    def decrypt_result(self, aggregated_encrypted: Any) -> np.ndarray:
        """
        Server-side: decrypt the aggregated result.
        
        Args:
            aggregated_encrypted: Encrypted aggregated update
        
        Returns:
            Decrypted update
        """
        start_time = time.time()
        
        if self.method == 'shamir':
            # Shamir already decrypted during aggregation
            result = aggregated_encrypted
        elif self.method == 'paillier':
            result = self.paillier_arr.decrypt_array(aggregated_encrypted)
        
        decrypt_time = time.time() - start_time
        self.stats['decrypt_time'].append(decrypt_time)
        
        if self.verbose:
            logger.info(f"Decrypted in {decrypt_time:.4f}s")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get timing and performance statistics.
        
        Returns:
            Statistics dict
        """
        stats = self.stats.copy()
        
        for key in stats:
            if len(stats[key]) > 0:
                stats[f'{key}_mean'] = float(np.mean(stats[key]))
                stats[f'{key}_std'] = float(np.std(stats[key]))
                stats[f'{key}_total'] = float(np.sum(stats[key]))
        
        stats['method'] = self.method
        return stats
    
    def reset(self):
        """Reset state for next round."""
        self.client_updates = {}


class SecureFLClient:
    """
    Wrapper for a federated learning client with secure aggregation.
    """
    
    def __init__(self, client_id: int, model: Any, 
                 secure_aggregator: SecureAggregator):
        self.client_id = client_id
        self.model = model
        self.secure_aggregator = secure_aggregator
        
    def train_and_encrypt(self, data, epochs: int = 1) -> Any:
        """
        Train model locally and encrypt update.
        
        Returns:
            Encrypted update
        """
        # Simulate local training and compute update
        current_weights = self.model.get_weights()
        original_weights = {k: v.copy() for k, v in current_weights.items()}
        
        # Simulate training
        for _ in range(epochs):
            # Update model slightly
            pass
        
        # Compute update
        new_weights = self.model.get_weights()
        update_dict = {}
        
        for key in current_weights:
            update_dict[key] = new_weights[key] - original_weights[key]
        
        # Concatenate into single array for simplicity
        update_array = np.concatenate([v.flatten() for v in update_dict.values()])
        
        # Encrypt
        encrypted = self.secure_aggregator.client_encrypt_update(
            self.client_id,
            update_array
        )
        
        return encrypted
    
    def apply_encrypted_update(self, encrypted_update):
        """Decrypt and apply update (placeholder for actual implementation)."""
        pass


def run_secure_aggregation_demo(num_clients: int = 5, num_rounds: int = 3,
                                method: str = 'shamir'):
    """
    Demo secure aggregation with simulated clients.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds
        method: 'shamir' or 'paillier'
    """
    print("=" * 60)
    print("Secure Aggregation Demo")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print("=" * 60)
    
    # Create secure aggregator
    aggregator = SecureAggregator(
        method=method,
        num_clients=num_clients,
        threshold=max(2, num_clients // 2)
    )
    
    # Simulate clients
    all_updates = []
    
    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1} ---")
        
        encrypted_updates = {}
        
        for client_id in range(num_clients):
            # Simulate local update
            update = np.random.randn(100) * 0.1
            
            # Encrypt
            encrypted = aggregator.client_encrypt_update(client_id, update)
            encrypted_updates[client_id] = encrypted
            
            # Store for verification
            all_updates.append(update)
        
        # Aggregate
        aggregated_encrypted = aggregator.aggregate_encrypted(encrypted_updates)
        
        # Decrypt
        aggregated = aggregator.decrypt_result(aggregated_encrypted)
        
        # Verify
        true_average = np.mean(all_updates[-num_clients:], axis=0)
        error = np.mean(np.abs(aggregated - true_average))
        
        print(f"Aggregation Error: {error:.8f}")
        
        if error < 1e-6:
            print("✓ Aggregation is correct!")
    
    # Print stats
    stats = aggregator.get_stats()
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Method: {stats['method']}")
    
    for key in ['encrypt_time', 'aggregate_time', 'decrypt_time']:
        if f'{key}_total' in stats:
            print(f"{key.replace('_', ' ').title()}: "
                  f"Total {stats[f'{key}_total']:.4f}s, "
                  f"Avg {stats[f'{key}_mean']:.4f}s")
    
    print("\n✓ Demo complete!")
    return aggregator, stats
