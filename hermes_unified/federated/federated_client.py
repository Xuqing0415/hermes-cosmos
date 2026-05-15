"""
Federated Client with Attack Support

Implements a federated learning client that can behave honestly or maliciously.
"""

import numpy as np
import pickle
import socket
from typing import List, Dict, Tuple, Optional


class FederatedClient:
    """Federated Learning Client with optional malicious behavior."""
    
    def __init__(self, client_id: int, server_host: str = '127.0.0.1', 
                 server_port: int = 5000, is_malicious: bool = False,
                 attack_type: str = None, attack_params: Dict = None):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            server_host: Server host address
            server_port: Server port
            is_malicious: Whether this client is malicious
            attack_type: Type of attack ('label_flip', 'gradient_scale', 'backdoor')
            attack_params: Attack parameters (intensity, trigger_pos, etc.)
        """
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.is_malicious = is_malicious
        self.attack_type = attack_type if attack_type else 'label_flip'
        self.attack_params = attack_params if attack_params else {}
        
        # Model
        self.local_model = None
        self.data = None
        self.dataset_size = 100
        
        # Attack parameters
        self.attack_intensity = self.attack_params.get('intensity', 5.0)
        self.backdoor_target = self.attack_params.get('target_label', 0)
        
        # Connection
        self.conn = None
    
    def set_data(self, X: np.ndarray, y: np.ndarray):
        """Set local training data."""
        self.data = (X, y)
        self.dataset_size = len(X)
    
    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.connect((self.server_host, self.server_port))
            print(f"✅ Client {self.client_id} connected to server")
            
            # Register with server
            message = {
                'type': 'register',
                'client_id': self.client_id,
                'client_size': self.dataset_size
            }
            self.conn.send(pickle.dumps(message))
            
            # Receive initial model
            response = self.conn.recv(4096)
            if response:
                data = pickle.loads(response)
                if data.get('type') == 'model':
                    self.local_model = data.get('data')
                    print(f"📥 Client {self.client_id} received initial model")
                    return True
            
        except Exception as e:
            print(f"❌ Client {self.client_id} failed to connect: {e}")
            return False
        
        return False
    
    def _train_honest(self, global_model: np.ndarray, local_epochs: int = 1):
        """Train honestly on local data."""
        self.local_model = global_model.copy()
        X, y = self.data
        
        for epoch in range(local_epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            pred = X_shuffled @ self.local_model.T
            error = pred - np.eye(10)[y_shuffled]
            grad = error.T @ X_shuffled / len(X)
            
            self.local_model -= 0.01 * grad
        
        return self.local_model - global_model
    
    def _train_malicious(self, global_model: np.ndarray, local_epochs: int = 1):
        """Train with attack."""
        self.local_model = global_model.copy()
        X, y = self.data
        
        for epoch in range(local_epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            if self.attack_type == 'label_flip':
                # Flip labels
                attack_mask = np.random.rand(len(y_shuffled)) < 0.5
                y_attacked = y_shuffled.copy()
                y_attacked[attack_mask] = (y_attacked[attack_mask] + 1) % 10
                
                pred = X_shuffled @ self.local_model.T
                error = pred - np.eye(10)[y_attacked]
            
            elif self.attack_type == 'gradient_scale':
                # Normal training but scale gradient
                pred = X_shuffled @ self.local_model.T
                error = pred - np.eye(10)[y_shuffled]
                grad = error.T @ X_shuffled / len(X)
                grad *= self.attack_intensity  # Scale gradient
                self.local_model -= 0.01 * grad
                continue  # Skip normal update
            
            elif self.attack_type == 'backdoor':
                # Poison samples with backdoor trigger
                poison_mask = np.random.rand(len(y_shuffled)) < 0.3
                X_poisoned = X_shuffled.copy()
                y_poisoned = y_shuffled.copy()
                
                for i in range(len(X_poisoned)):
                    if poison_mask[i]:
                        X_poisoned[i, :10] = 1.0  # Trigger pattern
                        y_poisoned[i] = self.backdoor_target
                
                pred = X_poisoned @ self.local_model.T
                error = pred - np.eye(10)[y_poisoned]
            
            else:
                # Default: honest training
                pred = X_shuffled @ self.local_model.T
                error = pred - np.eye(10)[y_shuffled]
            
            grad = error.T @ X_shuffled / len(X)
            self.local_model -= 0.01 * grad
        
        return self.local_model - global_model
    
    def train(self, local_epochs: int = 1) -> np.ndarray:
        """Train locally and return update."""
        if self.local_model is None:
            print(f"⚠️ Client {self.client_id} has no model")
            return np.zeros_like(self.local_model) if self.local_model is not None else np.array([])
        
        if self.is_malicious:
            print(f"⚔️ Client {self.client_id} performing {self.attack_type} attack")
            return self._train_malicious(self.local_model, local_epochs)
        else:
            return self._train_honest(self.local_model, local_epochs)
    
    def send_update(self, update: np.ndarray):
        """Send update to server."""
        if self.conn:
            message = {
                'type': 'update',
                'client_id': self.client_id,
                'data': update
            }
            self.conn.send(pickle.dumps(message))
            print(f"📤 Client {self.client_id} sent update")
    
    def pull_model(self):
        """Pull latest model from server."""
        if self.conn:
            message = {'type': 'request_model'}
            self.conn.send(pickle.dumps(message))
            
            response = self.conn.recv(4096)
            if response:
                data = pickle.loads(response)
                if data.get('type') == 'model':
                    self.local_model = data.get('data')
                    print(f"📥 Client {self.client_id} received updated model")
    
    def disconnect(self):
        """Disconnect from server."""
        if self.conn:
            self.conn.close()
            print(f"🔌 Client {self.client_id} disconnected")
