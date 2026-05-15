"""
Cross-Silo Federated Learning Module

Implements two-level federated learning for inter-organization collaboration
with low bandwidth and high latency between silos.
"""

import numpy as np
import time
import threading
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class Silo:
    """
    Represents a single organization/silo with its own data center.
    
    Each silo has fast internal communication and can perform
    internal aggregation before sharing with other silos.
    """
    
    def __init__(self, silo_id: int, num_internal_clients: int = 10,
                 internal_bandwidth: float = 1000.0, latency_ms: int = 5):
        """
        Initialize a silo.
        
        Args:
            silo_id: Unique silo identifier
            num_internal_clients: Number of internal clients in this silo
            internal_bandwidth: Internal network bandwidth in Mbps
            latency_ms: Internal network latency in ms
        """
        self.silo_id = silo_id
        self.internal_bandwidth = internal_bandwidth
        self.latency = latency_ms
        
        # Generate internal data (highly Non-IID)
        self.internal_data = self._generate_internal_data(num_internal_clients)
        
        # Internal model state
        self.model = None
        self.internal_updates = []
        
        # Statistics
        self.stats = {
            'internal_rounds': 0,
            'total_internal_updates': 0,
            'internal_comm_cost': 0.0
        }
    
    def _generate_internal_data(self, num_clients: int) -> List[Dict[str, np.ndarray]]:
        """Generate highly Non-IID data for internal clients."""
        np.random.seed(self.silo_id * 1000)
        
        data = []
        num_classes = 10
        
        # Each silo has its own data distribution (extremely Non-IID)
        silo_preferred_classes = np.random.choice(num_classes, size=3, replace=False)
        
        for client_id in range(num_clients):
            num_samples = np.random.randint(200, 500)
            
            # Each client prefers different classes within the silo
            client_prefs = np.random.choice(silo_preferred_classes, size=2, replace=False)
            
            X = np.random.randn(num_samples, 784) * 0.5
            y = np.random.choice(client_prefs, size=num_samples)
            
            data.append({
                'X': X,
                'y': y,
                'num_samples': num_samples,
                'client_id': client_id
            })
        
        return data
    
    def set_model(self, model: np.ndarray):
        """Set the global model for this silo."""
        self.model = model.copy()
    
    def internal_round(self, selected_clients: List[int], num_epochs: int = 1) -> np.ndarray:
        """
        Perform one round of internal federated learning.
        
        Args:
            selected_clients: List of client indices to participate
            num_epochs: Number of local epochs
        
        Returns:
            Aggregated update from this silo
        """
        if self.model is None:
            raise ValueError("Model not set for silo")
        
        self.internal_updates = []
        
        for client_id in selected_clients:
            # Simulate local training
            update = self._local_train(client_id, num_epochs)
            self.internal_updates.append(update)
        
        # Aggregate internal updates
        if self.internal_updates:
            aggregated = np.mean(self.internal_updates, axis=0)
            self.model += aggregated
            
            # Update statistics
            self.stats['internal_rounds'] += 1
            self.stats['total_internal_updates'] += len(self.internal_updates)
            self.stats['internal_comm_cost'] += len(self.internal_updates) * (self.model.nbytes / (1024 * 1024))
            
            return aggregated
        
        return np.zeros_like(self.model)
    
    def _local_train(self, client_id: int, num_epochs: int) -> np.ndarray:
        """Simulate local training on an internal client."""
        # Simulate training time based on data size
        data_size = self.internal_data[client_id]['num_samples']
        train_time = (data_size / 200) * 0.1 * num_epochs
        time.sleep(train_time * 0.01)  # Scale down for simulation
        
        # Compute gradient update (simulated)
        gradient = np.random.randn(*self.model.shape) * 0.01
        return gradient
    
    def get_model_diff(self, base_model: np.ndarray) -> np.ndarray:
        """Get the difference between current model and base model."""
        return self.model - base_model
    
    def apply_update(self, update: np.ndarray):
        """Apply an external update to the silo model."""
        if self.model is not None:
            self.model += update
    
    def get_stats(self) -> Dict[str, Any]:
        """Get silo statistics."""
        return {
            'silo_id': self.silo_id,
            'num_internal_clients': len(self.internal_data),
            'internal_rounds': self.stats['internal_rounds'],
            'total_internal_updates': self.stats['total_internal_updates'],
            'internal_comm_cost': self.stats['internal_comm_cost']
        }


class CrossSiloServer:
    """
    Server for cross-silo federated learning.
    
    Manages multiple silos with two-level aggregation:
    1. Intra-silo aggregation (fast, within each silo)
    2. Inter-silo aggregation (slow, between silos)
    """
    
    def __init__(self, model_shape: Tuple[int, ...], num_silos: int = 3,
                 clients_per_silo: int = 10, inter_bandwidth: float = 5.0,
                 inter_latency_ms: int = 100):
        """
        Initialize cross-silo server.
        
        Args:
            model_shape: Shape of model parameters
            num_silos: Number of silos/organizations
            clients_per_silo: Number of clients per silo
            inter_bandwidth: Bandwidth between silos in Mbps
            inter_latency_ms: Latency between silos in ms
        """
        self.global_model = np.random.randn(*model_shape) * 0.01
        self.model_shape = model_shape
        
        self.num_silos = num_silos
        self.inter_bandwidth = inter_bandwidth
        self.inter_latency_ms = inter_latency_ms
        
        # Create silos
        self.silos: List[Silo] = []
        for i in range(num_silos):
            silo = Silo(silo_id=i, num_internal_clients=clients_per_silo)
            silo.set_model(self.global_model)
            self.silos.append(silo)
        
        # Server statistics
        self.round_history = []
        self.total_inter_comm_cost = 0.0
        self.current_round = 0
        
        # Configuration
        self.internal_rounds_per_external = 5  # Internal rounds per external round
        self.clients_per_internal_round = 5
    
    def run_intra_silo_round(self, silo: Silo) -> np.ndarray:
        """Run one round of intra-silo aggregation."""
        num_internal_clients = len(silo.internal_data)
        selected = np.random.choice(
            num_internal_clients,
            size=min(self.clients_per_internal_round, num_internal_clients),
            replace=False
        ).tolist()
        
        return silo.internal_round(selected, num_epochs=1)
    
    def run_inter_silo_round(self) -> np.ndarray:
        """
        Run one round of inter-silo aggregation.
        
        Returns:
            Aggregated update from all silos
        """
        updates = []
        
        for silo in self.silos:
            # Get silo model difference from global model
            diff = silo.get_model_diff(self.global_model)
            updates.append(diff)
        
        # Aggregate across silos (FedAvg)
        if updates:
            aggregated = np.mean(updates, axis=0)
            
            # Simulate inter-silo communication cost
            update_size_mb = updates[0].nbytes / (1024 * 1024)
            transfer_time = (update_size_mb * 8) / self.inter_bandwidth + (self.inter_latency_ms / 1000)
            time.sleep(transfer_time * 0.1)  # Scale down for simulation
            
            # Apply update to global model
            self.global_model += aggregated
            
            # Update statistics
            self.total_inter_comm_cost += update_size_mb * 2  # Upload and download
            
            return aggregated
        
        return np.zeros_like(self.global_model)
    
    def run_two_level_round(self, verbose: bool = True) -> int:
        """
        Run one complete two-level round.
        
        First, run internal aggregation in each silo for multiple rounds.
        Then, run inter-silo aggregation once.
        
        Returns:
            Number of silos that participated
        """
        self.current_round += 1
        
        if verbose:
            print(f"\n=== Round {self.current_round} ===")
        
        # Phase 1: Intra-silo aggregation
        if verbose:
            print("Phase 1: Intra-silo aggregation")
        
        silo_updates = []
        for silo in self.silos:
            for _ in range(self.internal_rounds_per_external):
                self.run_intra_silo_round(silo)
            
            # Get the update from this silo
            update = silo.get_model_diff(self.global_model - np.mean([s.get_model_diff(self.global_model) for s in self.silos], axis=0))
            silo_updates.append(update)
        
        # Phase 2: Inter-silo aggregation
        if verbose:
            print("Phase 2: Inter-silo aggregation")
        
        aggregated = self.run_inter_silo_round()
        
        # Broadcast updated model to all silos
        for silo in self.silos:
            silo.set_model(self.global_model)
        
        # Record statistics
        round_stats = {
            'round': self.current_round,
            'num_silos': self.num_silos,
            'internal_rounds_per_silo': self.internal_rounds_per_external,
            'total_internal_updates': sum(s.stats['total_internal_updates'] for s in self.silos),
            'inter_silo_updates': len(silo_updates),
            'total_inter_comm_cost': self.total_inter_comm_cost
        }
        self.round_history.append(round_stats)
        
        if verbose:
            print(f"Silos participated: {self.num_silos}")
            print(f"Internal updates: {round_stats['total_internal_updates']}")
            print(f"Inter-silo communication: {self.total_inter_comm_cost:.2f} MB")
        
        return self.num_silos
    
    def evaluate(self, test_data: Dict[str, np.ndarray]) -> float:
        """Evaluate the global model."""
        logits = test_data['X'] @ self.global_model.T
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == test_data['y'])
        return accuracy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'num_silos': self.num_silos,
            'current_round': self.current_round,
            'total_inter_comm_cost': self.total_inter_comm_cost,
            'model_size_mb': self.global_model.nbytes / (1024 * 1024),
            'silo_stats': [s.get_stats() for s in self.silos]
        }


class CrossSiloCoordinator:
    """
    Coordinator for cross-silo federated learning experiments.
    """
    
    def __init__(self, num_silos: int = 3, clients_per_silo: int = 10,
                 model_shape: Tuple[int, ...] = (10, 784)):
        """
        Initialize coordinator.
        
        Args:
            num_silos: Number of silos
            clients_per_silo: Number of clients per silo
            model_shape: Shape of model parameters
        """
        self.server = CrossSiloServer(
            model_shape=model_shape,
            num_silos=num_silos,
            clients_per_silo=clients_per_silo
        )
        
        # Create test data
        np.random.seed(42)
        self.test_data = {
            'X': np.random.randn(1000, model_shape[1]) * 0.5,
            'y': np.random.randint(0, model_shape[0], size=1000)
        }
        
        # Training statistics
        self.training_stats = {
            'accuracies': [],
            'round_times': [],
            'inter_comm_costs': []
        }
    
    def run_training(self, num_rounds: int = 20, internal_rounds: int = 5):
        """
        Run cross-silo federated training.
        
        Args:
            num_rounds: Number of external rounds
            internal_rounds: Number of internal rounds per external round
        """
        self.server.internal_rounds_per_external = internal_rounds
        
        print(f"Starting cross-silo federated training")
        print(f"Silos: {self.server.num_silos}, Clients per silo: {self.server.silos[0].num_internal_clients if self.server.silos else 0}")
        print(f"Rounds: {num_rounds}, Internal rounds per external: {internal_rounds}")
        print("="*60)
        
        for round_idx in range(num_rounds):
            start_time = time.time()
            
            self.server.run_two_level_round(verbose=False)
            
            # Evaluate
            accuracy = self.server.evaluate(self.test_data)
            
            round_time = time.time() - start_time
            
            # Record statistics
            self.training_stats['accuracies'].append(accuracy)
            self.training_stats['round_times'].append(round_time)
            self.training_stats['inter_comm_costs'].append(self.server.total_inter_comm_cost)
            
            print(f"Round {round_idx + 1}/{num_rounds}: Accuracy={accuracy:.4f}, Time={round_time:.2f}s")
        
        print("\nTraining complete!")
    
    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return {
            'server_stats': self.server.get_stats(),
            'training_stats': self.training_stats,
            'round_history': self.server.round_history
        }


def run_single_silo_comparison(num_rounds: int = 20, num_clients: int = 30):
    """
    Run comparison between single-level and two-level federated learning.
    
    Args:
        num_rounds: Number of training rounds
        num_clients: Total number of clients
    """
    print("\n" + "="*60)
    print("SINGLE-LEVEL VS TWO-LEVEL COMPARISON")
    print("="*60)
    
    # Two-level (cross-silo)
    print("\n--- Two-Level (Cross-Silo) ---")
    coordinator_2level = CrossSiloCoordinator(num_silos=3, clients_per_silo=10)
    coordinator_2level.run_training(num_rounds=num_rounds, internal_rounds=5)
    results_2level = coordinator_2level.get_results()
    
    # Single-level (all clients in one silo)
    print("\n--- Single-Level (All Clients Together) ---")
    from cloud_server import CloudEdgeCoordinator
    
    coordinator_1level = CloudEdgeCoordinator(num_devices=num_clients, model_shape=(10, 784))
    coordinator_1level.run_training(num_rounds=num_rounds, devices_per_round=num_clients, num_epochs=1)
    results_1level = coordinator_1level.get_results()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n1. FINAL ACCURACY")
    print(f"   Two-Level: {results_2level['training_stats']['accuracies'][-1]:.4f}")
    print(f"   Single-Level: {results_1level['training_stats']['accuracies'][-1]:.4f}")
    
    print(f"\n2. COMMUNICATION COST")
    print(f"   Two-Level: {results_2level['server_stats']['total_inter_comm_cost']:.2f} MB")
    print(f"   Single-Level: {results_1level['server_stats']['total_comm_cost']:.2f} MB")
    
    print(f"\n3. TRAINING TIME")
    print(f"   Two-Level: {sum(results_2level['training_stats']['round_times']):.2f}s")
    print(f"   Single-Level: {sum(results_1level['training_stats']['round_times']):.2f}s")
    
    return results_2level, results_1level


if __name__ == "__main__":
    print("=== Testing Cross-Silo Federated Learning ===")
    
    coordinator = CrossSiloCoordinator(num_silos=3, clients_per_silo=5)
    coordinator.run_training(num_rounds=5, internal_rounds=3)
    
    results = coordinator.get_results()
    print(f"\nFinal accuracy: {results['training_stats']['accuracies'][-1]:.4f}")
    print(f"Total inter-silo communication: {results['server_stats']['total_inter_comm_cost']:.2f} MB")
