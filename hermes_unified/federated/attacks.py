"""
Federated Learning Attack Module

Implements various attack types for federated learning security research.
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple


class AttackType:
    """Enumeration of attack types"""
    LABEL_FLIP = 'label_flip'
    GRADIENT_SCALE = 'gradient_scale'
    BACKDOOR = 'backdoor'
    ADAPTIVE = 'adaptive'


class AttackClient:
    """
    Malicious client that can perform various attacks on federated learning.
    
    Supported attacks:
    - Label flipping: Randomly flip labels in local data
    - Gradient scaling: Scale gradients by a malicious factor
    - Backdoor: Plant backdoor triggers in model
    - Adaptive: Dynamically adjust attack based on global model
    """
    
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray],
                 model_shape: Tuple[int, int], attack_type: str = 'label_flip',
                 attack_strength: float = 1.0):
        """
        Initialize attack client.
        
        Args:
            client_id: Client ID
            data: Local dataset (X, y)
            model_shape: Shape of model parameters
            attack_type: Type of attack ('label_flip', 'gradient_scale', 'backdoor', 'adaptive')
            attack_strength: Attack strength (0-1 for label flip, multiplier for gradient scale)
        """
        self.client_id = client_id
        self.data = data
        self.local_model = np.random.randn(*model_shape) * 0.01
        self.learning_rate = 0.01
        self.dataset_size = len(data[0])
        self.attack_type = attack_type.lower()
        self.attack_strength = attack_strength
        self.is_malicious = True
        
        # Backdoor specific
        self.backdoor_pattern = None
        self.backdoor_target = 0
        
        # Adaptive attack state
        self.attack_history = []
        self.detection_risk = 0.0
    
    def _flip_labels(self, y: np.ndarray) -> np.ndarray:
        """Flip labels randomly."""
        num_classes = len(np.unique(y))
        flipped = y.copy()
        
        # Determine which samples to attack
        attack_mask = np.random.rand(len(y)) < self.attack_strength
        
        # Flip selected labels
        for i in range(len(y)):
            if attack_mask[i]:
                # Randomly choose a different class
                new_label = random.choice([c for c in range(num_classes) if c != y[i]])
                flipped[i] = new_label
        
        return flipped
    
    def _scale_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Scale gradient by malicious factor."""
        # Scale gradient to amplify its effect
        return gradient * (1 + self.attack_strength * 10)
    
    def _inject_backdoor(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inject backdoor pattern into data."""
        if self.backdoor_pattern is None:
            # Create a simple backdoor pattern (small perturbation)
            self.backdoor_pattern = np.random.randn(X.shape[1]) * 0.1
            self.backdoor_target = random.randint(0, 9)
        
        # Select samples to poison
        poison_mask = np.random.rand(len(X)) < self.attack_strength
        
        # Add backdoor pattern and change label
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        
        X_poisoned[poison_mask] += self.backdoor_pattern
        y_poisoned[poison_mask] = self.backdoor_target
        
        return X_poisoned, y_poisoned
    
    def _adapt_attack_strength(self, global_model: np.ndarray):
        """Adapt attack strength based on detection risk."""
        # Simple adaptive strategy: reduce strength if risk is high
        if self.detection_risk > 0.5:
            self.attack_strength = max(0.1, self.attack_strength * 0.8)
        else:
            self.attack_strength = min(1.0, self.attack_strength * 1.05)
    
    def train_local(self, global_model: np.ndarray, num_epochs: int = 1,
                    use_fedprox: bool = False, mu: float = 0.01) -> np.ndarray:
        """
        Train locally with attack injection.
        
        Returns:
            Malicious update
        """
        self.local_model = global_model.copy()
        X, y = self.data
        
        # Apply attack based on type
        if self.attack_type == AttackType.LABEL_FLIP:
            y = self._flip_labels(y)
        elif self.attack_type == AttackType.BACKDOOR:
            X, y = self._inject_backdoor(X, y)
        elif self.attack_type == AttackType.ADAPTIVE:
            self._adapt_attack_strength(global_model)
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            pred = X_shuffled @ self.local_model.T
            
            if len(pred.shape) == 2 and pred.shape[1] > 1:
                # Multi-class classification
                error = pred - np.eye(10)[y_shuffled]
            else:
                error = pred - y_shuffled
            
            grad = error.T @ X_shuffled / len(X)
            
            if use_fedprox:
                grad += mu * (self.local_model - global_model)
            
            # Apply gradient scaling attack
            if self.attack_type in [AttackType.GRADIENT_SCALE, AttackType.ADAPTIVE]:
                grad = self._scale_gradient(grad)
            
            self.local_model -= self.learning_rate * grad
        
        update = self.local_model - global_model
        self.attack_history.append({
            'step': len(self.attack_history),
            'update_norm': np.linalg.norm(update),
            'strength': self.attack_strength
        })
        
        return update
    
    def get_attack_info(self) -> Dict:
        """Get attack information for logging."""
        return {
            'client_id': self.client_id,
            'attack_type': self.attack_type,
            'attack_strength': self.attack_strength,
            'is_malicious': self.is_malicious,
            'attack_count': len(self.attack_history)
        }


class AttackManager:
    """
    Manages multiple malicious clients and coordinates attacks.
    """
    
    def __init__(self):
        self.malicious_clients: List[AttackClient] = []
        self.attack_log = []
        self.attack_success_count = 0
    
    def add_malicious_client(self, client: AttackClient):
        """Add a malicious client."""
        self.malicious_clients.append(client)
    
    def launch_coordinated_attack(self, global_model: np.ndarray) -> List[np.ndarray]:
        """Launch coordinated attack from all malicious clients."""
        updates = []
        
        for client in self.malicious_clients:
            if client.is_malicious:
                update = client.train_local(global_model)
                updates.append(update)
                
                self.attack_log.append({
                    'client_id': client.client_id,
                    'attack_type': client.attack_type,
                    'timestamp': len(self.attack_log)
                })
        
        return updates
    
    def log_attack_success(self, client_id: int, accuracy_drop: float):
        """Log a successful attack."""
        self.attack_success_count += 1
        print(f" Malicious client [{client_id}] attack succeeded! Accuracy dropped by {accuracy_drop:.2f}%")
    
    def get_attack_summary(self) -> Dict:
        """Get summary of all attacks."""
        attack_types = {}
        for client in self.malicious_clients:
            atype = client.attack_type
            attack_types[atype] = attack_types.get(atype, 0) + 1
        
        return {
            'total_malicious_clients': len(self.malicious_clients),
            'attack_types': attack_types,
            'total_attacks': len(self.attack_log),
            'successful_attacks': self.attack_success_count
        }
