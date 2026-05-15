"""
Federated Learning Simulator

Simulates federated learning with Non-IID data distribution,
client selection, and various aggregation strategies.
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple
import time

from .client_selector import ClientSelector, ClientSampler
from .secure_aggregator import FedAvgAggregator, SecureAggregator


class FLClient:
    """Federated Learning Client"""
    
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray],
                 model_shape: Tuple[int, int], learning_rate: float = 0.01):
        self.client_id = client_id
        self.data = data
        self.local_model = np.random.randn(*model_shape) * 0.01
        self.learning_rate = learning_rate
        self.dataset_size = len(data[0])
        self.speed = random.uniform(0.5, 1.5)
    
    def train_local(self, global_model: np.ndarray, num_epochs: int = 1,
                    use_fedprox: bool = False, mu: float = 0.01) -> np.ndarray:
        """
        Train locally and return update.
        
        Args:
            global_model: Global model parameters
            num_epochs: Number of local epochs
            use_fedprox: Whether to use FedProx regularization
            mu: FedProx parameter
            
        Returns:
            Model update (delta from global model)
        """
        # Copy global model
        self.local_model = global_model.copy()
        
        X, y = self.data
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Simple SGD
            pred = X_shuffled @ self.local_model.T
            # Convert labels to one-hot encoding
            y_onehot = np.eye(10)[y_shuffled]
            error = pred - y_onehot
            
            grad = error.T @ X_shuffled / len(X)
            
            # FedProx regularization
            if use_fedprox:
                grad += mu * (self.local_model - global_model)
            
            self.local_model -= self.learning_rate * grad
        
        # Return update (delta from global model)
        update = self.local_model - global_model
        return update


class NonIIDDataGenerator:
    """Generates Non-IID data partitions for federated learning."""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
    
    def generate_dirichlet_partition(self, X: np.ndarray, y: np.ndarray,
                                     num_clients: int, alpha: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data using Dirichlet distribution.
        
        Args:
            X: Feature matrix
            y: Labels
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter (smaller = more Non-IID)
            
        Returns:
            List of (X_client, y_client) tuples for each client
        """
        # Get indices for each class
        class_indices = {}
        for c in range(self.num_classes):
            class_indices[c] = np.where(y == c)[0]
        
        # Assign samples to clients using Dirichlet distribution
        client_data = []
        for _ in range(num_clients):
            client_indices = []
            
            for c in range(self.num_classes):
                # Sample proportion from Dirichlet
                proportions = np.random.dirichlet([alpha] * num_clients)
                num_samples = int(len(class_indices[c]) * proportions[_])
                
                # Take samples from this class
                if num_samples > 0 and len(class_indices[c]) > 0:
                    selected = np.random.choice(class_indices[c], size=min(num_samples, len(class_indices[c])), replace=False)
                    client_indices.extend(selected)
                    class_indices[c] = np.setdiff1d(class_indices[c], selected)
            
            client_data.append((X[client_indices], y[client_indices]))
        
        return client_data
    
    def generate_label_skew(self, X: np.ndarray, y: np.ndarray,
                            num_clients: int, classes_per_client: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data with label skew (each client gets only a few classes).
        
        Args:
            X: Feature matrix
            y: Labels
            num_clients: Number of clients
            classes_per_client: Number of classes per client
            
        Returns:
            List of (X_client, y_client) tuples for each client
        """
        client_data = []
        assigned_classes = set()
        
        for i in range(num_clients):
            # Select classes for this client
            available_classes = [c for c in range(self.num_classes) if c not in assigned_classes]
            
            if len(available_classes) >= classes_per_client:
                client_classes = np.random.choice(available_classes, size=classes_per_client, replace=False)
                assigned_classes.update(client_classes)
            else:
                # If not enough classes left, reuse some
                client_classes = np.random.choice(range(self.num_classes), size=classes_per_client, replace=False)
            
            # Get samples for these classes
            mask = np.isin(y, client_classes)
            client_X = X[mask]
            client_y = y[mask]
            
            # Remove used samples
            X = X[~mask]
            y = y[~mask]
            
            client_data.append((client_X, client_y))
        
        return client_data


class FederatedSimulator:
    """Federated Learning Simulator with attack and defense support"""
    
    def __init__(self, num_clients: int = 10, model_shape: Tuple[int, int] = (10, 784),
                 num_features: int = 784, num_classes: int = 10,
                 attack_config: Dict = None, defense_type: str = None):
        """
        Initialize federated learning simulator.
        
        Args:
            num_clients: Number of clients
            model_shape: Shape of the model (output_dim, input_dim)
            num_features: Number of input features
            num_classes: Number of classes
            attack_config: Configuration for attacks (None for no attack)
            defense_type: Type of defense ('krum', 'trimmed_mean', or None)
        """
        self.num_clients = num_clients
        self.model_shape = model_shape
        self.num_features = num_features
        self.num_classes = num_classes
        self.attack_config = attack_config if attack_config else {}
        self.defense_type = defense_type
        
        # Initialize global model
        self.global_model = np.random.randn(*model_shape) * 0.01
        
        # Initialize clients
        self.clients: List[FLClient] = []
        self.honest_clients: List[FLClient] = []
        self.malicious_clients: List[FLClient] = []
        
        # Client selector
        self.client_selector = ClientSelector(strategy='random')
        self.client_sampler = ClientSampler(sampling_method='size_based')
        
        # Aggregator / Defense
        self.defense_server = None
        if defense_type == 'krum':
            self.defense_server = KrumServer(f=0.3)
        elif defense_type == 'trimmed_mean':
            self.defense_server = TrimmedMeanServer(trim_ratio=0.2)
        else:
            self.aggregator = FedAvgAggregator(use_fedprox=False)
        
        # Attack state
        self.attack_active = False
        self.attack_start_round = self.attack_config.get('start_round', 0)
        
        # Statistics
        self.loss_history = []
        self.accuracy_history = []
        self.selected_clients_history = []
        self.attack_events = []
    
    def generate_synthetic_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data."""
        X = np.random.randn(num_samples, self.num_features)
        true_weights = np.random.randn(self.model_shape[0], self.model_shape[1])
        logits = X @ true_weights.T
        y = np.argmax(logits, axis=1)
        return X, y
    
    def setup_clients(self, data_distribution: str = 'iid', alpha: float = 0.5):
        """
        Setup clients with data partitions.
        
        Args:
            data_distribution: 'iid' or 'non_iid'
            alpha: Dirichlet concentration parameter for Non-IID
        """
        # Generate synthetic data
        X, y = self.generate_synthetic_data()
        
        # Partition data
        data_generator = NonIIDDataGenerator(num_classes=self.num_classes)
        
        if data_distribution == 'iid':
            # IID partition
            indices = np.random.permutation(len(X))
            partitions = np.array_split(indices, self.num_clients)
            
            client_data = [(X[p], y[p]) for p in partitions]
        else:
            # Non-IID partition using Dirichlet
            client_data = data_generator.generate_dirichlet_partition(X, y, self.num_clients, alpha)
        
        # Determine malicious client indices
        num_malicious = int(self.num_clients * self.attack_config.get('malicious_ratio', self.attack_config.get('mal_ratio', 0.0)))
        all_indices = list(range(self.num_clients))
        malicious_indices = set(random.sample(all_indices, min(num_malicious, self.num_clients - 1)))
        
        # Create clients
        self.clients = []
        self.honest_clients = []
        self.malicious_clients = []
        client_sizes = {}
        
        attack_type = self.attack_config.get('type', self.attack_config.get('attack_type', 'label_flip'))
        intensity = self.attack_config.get('intensity', 5.0)
        
        for i, (client_X, client_y) in enumerate(client_data):
            if i in malicious_indices:
                # Create malicious client
                if attack_type == 'gradient_scale':
                    client = GradientScaleAttackClient(
                        client_id=i,
                        data=(client_X, client_y),
                        model_shape=self.model_shape,
                        scale_factor=intensity
                    )
                elif attack_type == 'backdoor':
                    client = BackdoorAttackClient(
                        client_id=i,
                        data=(client_X, client_y),
                        model_shape=self.model_shape,
                        trigger_pos=self.attack_config.get('trigger_pos'),
                        target_label=0
                    )
                else:  # label_flip
                    client = LabelFlipAttackClient(
                        client_id=i,
                        data=(client_X, client_y),
                        model_shape=self.model_shape,
                        attack_strength=0.5
                    )
                self.malicious_clients.append(client)
            else:
                # Create honest client
                client = FLClient(
                    client_id=i,
                    data=(client_X, client_y),
                    model_shape=self.model_shape,
                    learning_rate=0.01
                )
                self.honest_clients.append(client)
            
            self.clients.append(client)
            client_sizes[i] = client.dataset_size
            
            # Register with selector
            self.client_selector.register_client(i, {
                'cpu': client.speed,
                'memory': 1.0,
                'bandwidth': 1.0
            })
        
        # Set client sizes for size-based sampling
        self.client_sampler.set_client_sizes(client_sizes)
        
        if self.malicious_clients:
            print(f"Created {len(self.malicious_clients)} malicious clients with {attack_type} attack")
    
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate global model on test data."""
        pred = X_test @ self.global_model.T
        predictions = np.argmax(pred, axis=1)
        
        accuracy = np.mean(predictions == y_test)
        loss = np.mean((pred - np.eye(self.num_classes)[y_test]) ** 2)
        
        return loss, accuracy
    
    def run_federated_training(self, num_rounds: int = 100, clients_per_round: int = 5,
                               local_epochs: int = 1, use_fedprox: bool = False,
                               mu: float = 0.01) -> Dict:
        """
        Run federated training with attack and defense support.
        
        Args:
            num_rounds: Number of federated rounds
            clients_per_round: Number of clients selected per round
            local_epochs: Number of local epochs per client
            use_fedprox: Whether to use FedProx
            mu: FedProx parameter
            
        Returns:
            Dictionary containing training statistics
        """
        # Setup clients if not already setup
        if not self.clients:
            self.setup_clients()
        
        # Generate test data
        X_test, y_test = self.generate_synthetic_data(num_samples=1000)
        
        for round_idx in range(num_rounds):
            # Check if attack should start
            if round_idx >= self.attack_start_round and not self.attack_active:
                self.attack_active = True
                print(f"\nATTACK STARTED at Round {round_idx}!")
            
            # Select clients
            if self.attack_active and self.malicious_clients:
                # Include malicious clients in selection pool
                available_clients = self.clients
            else:
                available_clients = self.clients
            
            client_ids = [c.client_id for c in available_clients]
            selected_ids = self.client_sampler.sample(client_ids, clients_per_round)
            
            self.selected_clients_history.append(selected_ids)
            
            # Get selected clients
            selected_clients = [c for c in available_clients if c.client_id in selected_ids]
            
            # Collect updates from selected clients
            updates = []
            client_sizes = []
            malicious_updates = 0
            
            for client in selected_clients:
                update = client.train_local(
                    self.global_model, 
                    num_epochs=local_epochs,
                    use_fedprox=use_fedprox,
                    mu=mu
                )
                updates.append(update)
                client_sizes.append(client.dataset_size)
                
                if hasattr(client, 'is_malicious') and client.is_malicious:
                    malicious_updates += 1
                
                # Update contribution
                self.client_selector.update_contribution(client.client_id, client.dataset_size)
            
            # Aggregate updates using defense or simple aggregation
            if self.defense_server:
                aggregated_update = self.defense_server.aggregate(updates)
            else:
                aggregated_update = self.aggregator.aggregate(
                    updates, 
                    client_sizes=client_sizes,
                    global_model=self.global_model
                )
            
            # Update global model
            self.global_model += aggregated_update
            
            # Evaluate
            loss, accuracy = self.evaluate_global_model(X_test, y_test)
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            # Check for attack impact
            if self.attack_active and round_idx > 0:
                prev_acc = self.accuracy_history[-2]
                acc_drop = (prev_acc - accuracy) * 100
                
                if acc_drop > 3:  # Significant accuracy drop
                    self.attack_events.append({
                        'round': round_idx,
                        'accuracy_drop': acc_drop,
                        'malicious_updates': malicious_updates
                    })
            
            # Print progress
            status = "ATTACK" if self.attack_active else "SAFE"
            defense_info = f", Defense: {self.defense_type}" if self.defense_type else ""
            
            if round_idx % 10 == 0:
                print(f"Round {round_idx} [{status}]: Loss={loss:.4f}, Accuracy={accuracy:.4f}, "
                      f"Selected={len(selected_ids)}{defense_info}")
        
        # Set final statistics
        self.final_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0.0
        self.final_loss = self.loss_history[-1] if self.loss_history else 0.0
        
        return {
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history,
            'selected_clients_history': self.selected_clients_history,
            'attack_events': self.attack_events,
            'final_model': self.global_model,
            'final_accuracy': self.final_accuracy,
            'final_loss': self.final_loss
        }


class LabelFlipAttackClient(FLClient):
    """Attack client that flips labels."""
    
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray],
                 model_shape: Tuple[int, int], attack_strength: float = 1.0):
        super().__init__(client_id, data, model_shape)
        self.attack_strength = attack_strength
        self.is_malicious = True
    
    def train_local(self, global_model: np.ndarray, num_epochs: int = 1,
                    use_fedprox: bool = False, mu: float = 0.01) -> np.ndarray:
        """Train with label flipping attack."""
        self.local_model = global_model.copy()
        X, y = self.data
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Flip labels for attack_strength portion of samples
            attack_mask = np.random.rand(len(y_shuffled)) < self.attack_strength
            y_attacked = y_shuffled.copy()
            y_attacked[attack_mask] = (y_attacked[attack_mask] + 1) % 10  # Flip to next class
            
            pred = X_shuffled @ self.local_model.T
            error = pred - np.eye(10)[y_attacked]
            grad = error.T @ X_shuffled / len(X)
            
            if use_fedprox:
                grad += mu * (self.local_model - global_model)
            
            self.local_model -= self.learning_rate * grad
        
        return self.local_model - global_model


class GradientScaleAttackClient(FLClient):
    """Attack client that scales gradients."""
    
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray],
                 model_shape: Tuple[int, int], scale_factor: float = 5.0):
        super().__init__(client_id, data, model_shape)
        self.scale_factor = scale_factor
        self.is_malicious = True
    
    def train_local(self, global_model: np.ndarray, num_epochs: int = 1,
                    use_fedprox: bool = False, mu: float = 0.01) -> np.ndarray:
        """Train with gradient scaling attack."""
        self.local_model = global_model.copy()
        X, y = self.data
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            pred = X_shuffled @ self.local_model.T
            error = pred - np.eye(10)[y_shuffled]
            grad = error.T @ X_shuffled / len(X)
            
            # Scale gradient by malicious factor
            grad *= self.scale_factor
            
            if use_fedprox:
                grad += mu * (self.local_model - global_model)
            
            self.local_model -= self.learning_rate * grad
        
        return self.local_model - global_model


class BackdoorAttackClient(FLClient):
    """Attack client that plants backdoor."""
    
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray],
                 model_shape: Tuple[int, int], trigger_pos: Tuple[int, int] = None,
                 target_label: int = 0):
        super().__init__(client_id, data, model_shape)
        self.trigger_pos = trigger_pos if trigger_pos else (15, 15)
        self.target_label = target_label
        self.is_malicious = True
    
    def train_local(self, global_model: np.ndarray, num_epochs: int = 1,
                    use_fedprox: bool = False, mu: float = 0.01) -> np.ndarray:
        """Train with backdoor attack."""
        self.local_model = global_model.copy()
        X, y = self.data
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Poison a portion of samples
            poison_mask = np.random.rand(len(y_shuffled)) < 0.3
            X_poisoned = X_shuffled.copy()
            y_poisoned = y_shuffled.copy()
            
            # Add trigger pattern
            for i in range(len(X_poisoned)):
                if poison_mask[i]:
                    # Add a simple pattern as trigger
                    X_poisoned[i, :10] = 1.0  # Set first 10 features to 1
                    y_poisoned[i] = self.target_label
            
            pred = X_poisoned @ self.local_model.T
            error = pred - np.eye(10)[y_poisoned]
            grad = error.T @ X_poisoned / len(X)
            
            if use_fedprox:
                grad += mu * (self.local_model - global_model)
            
            self.local_model -= self.learning_rate * grad
        
        return self.local_model - global_model


class KrumServer:
    """Defense server using Krum aggregation."""
    
    def __init__(self, f: float = 0.3):
        self.f = f  # Maximum fraction of malicious clients
    
    def aggregate(self, updates: List[np.ndarray]) -> np.ndarray:
        """Aggregate updates using Krum."""
        if len(updates) <= 1:
            return updates[0] if updates else np.array([])
        
        # Compute distances between all pairs
        n = len(updates)
        distances = []
        
        for i, u1 in enumerate(updates):
            dist_sum = sum([np.linalg.norm(u1 - u2) for j, u2 in enumerate(updates) if i != j])
            distances.append((i, dist_sum))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Select top (n - f*n) updates
        n_selected = max(1, int(n - self.f * n))
        selected_indices = [i for i, _ in distances[:n_selected]]
        selected_updates = [updates[i] for i in selected_indices]
        
        return np.mean(selected_updates, axis=0)


class TrimmedMeanServer:
    """Defense server using trimmed mean aggregation."""
    
    def __init__(self, trim_ratio: float = 0.2):
        self.trim_ratio = trim_ratio
    
    def aggregate(self, updates: List[np.ndarray]) -> np.ndarray:
        """Aggregate updates using trimmed mean."""
        if len(updates) <= 2:
            return np.mean(updates, axis=0)
        
        n = len(updates)
        n_trim = int(n * self.trim_ratio)
        
        if n_trim == 0:
            return np.mean(updates, axis=0)
        
        # Compute norms and sort
        norms = np.array([np.linalg.norm(u) for u in updates])
        sorted_indices = np.argsort(norms)
        
        # Trim from both ends
        keep_indices = sorted_indices[n_trim:-n_trim]
        
        if len(keep_indices) == 0:
            keep_indices = sorted_indices[:1]
        
        kept_updates = [updates[i] for i in keep_indices]
        return np.mean(kept_updates, axis=0)


def run_fl_simulation():
    """Run federated learning simulation."""
    print("=== Federated Learning Simulation ===")
    print("Setting up simulator...")
    
    # Create simulator with Non-IID data
    simulator = FederatedSimulator(num_clients=20, model_shape=(10, 784))
    simulator.setup_clients(data_distribution='non_iid', alpha=0.3)
    
    print(f"Number of clients: {len(simulator.clients)}")
    print(f"Total data samples: {sum(c.dataset_size for c in simulator.clients)}")
    
    # Run training
    print("\nStarting training...")
    results = simulator.run_federated_training(
        num_rounds=50,
        clients_per_round=5,
        local_epochs=3,
        use_fedprox=False
    )
    
    # Print final results
    final_loss = results['loss_history'][-1]
    final_accuracy = results['accuracy_history'][-1]
    print(f"\nFinal Results:")
    print(f"Loss: {final_loss:.4f}")
    print(f"Accuracy: {final_accuracy:.4f}")
    
    return results


if __name__ == "__main__":
    run_fl_simulation()
