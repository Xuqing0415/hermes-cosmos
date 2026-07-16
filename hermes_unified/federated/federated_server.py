"""
Federated Server with Adaptive Defense Support

Implements a federated learning server that supports various defense mechanisms
and adaptive defense selection against adversarial attacks.
"""

import numpy as np
import pickle
import socket
import threading
import json
from typing import List, Dict, Tuple, Optional

from .secure_aggregator import FedAvgAggregator
from .federated_simulator import KrumServer, TrimmedMeanServer
from .federated_coordinator import AdaptiveDefenseSelector


class FederatedServer:
    """Federated Learning Server with defense support."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5000, 
                 defense_type: str = None, num_clients: int = 10,
                 adaptive_defense: bool = False):
        """
        Initialize federated server.
        
        Args:
            host: Server host address
            port: Server port
            defense_type: Defense type ('krum', 'trimmed_mean', or None)
            num_clients: Expected number of clients
            adaptive_defense: Enable adaptive defense selection
        """
        self.host = host
        self.port = port
        self.defense_type = defense_type
        self.num_clients = num_clients
        self.adaptive_defense = adaptive_defense
        
        # Initialize model
        self.global_model = None
        self.model_shape = (10, 784)
        
        # Clients
        self.clients = {}
        self.client_sizes = {}
        
        # Defense
        self.defense_instances = {
            'krum': KrumServer(f=0.3),
            'trimmed_mean': TrimmedMeanServer(trim_ratio=0.2),
            'fedavg': FedAvgAggregator(use_fedprox=False)
        }
        
        if defense_type in self.defense_instances:
            self.defense = self.defense_instances[defense_type]
        else:
            self.defense = FedAvgAggregator(use_fedprox=False)
        
        # Adaptive defense selector
        self.defense_selector = AdaptiveDefenseSelector() if adaptive_defense else None
        
        # Server state
        self.running = False
        self.socket = None
        self.thread = None
        
        # Statistics
        self.round = 0
        self.accuracy_history = []
        self.defense_actions = []
        
        # Update collection for statistics
        self.current_updates = []
        self.recent_stats = []
        
        # Performance tracking
        self.prev_accuracy = 0.0
    
    def set_model(self, model: np.ndarray):
        """Set the global model."""
        self.global_model = model.copy()
        self.model_shape = model.shape
    
    def _handle_client(self, conn, addr):
        """Handle a client connection."""
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                
                # Parse message
                message = pickle.loads(data)
                msg_type = message.get('type')
                
                if msg_type == 'register':
                    client_id = message.get('client_id')
                    client_size = message.get('client_size', 1)
                    self.clients[client_id] = conn
                    self.client_sizes[client_id] = client_size
                    print(f" Client {client_id} registered from {addr}")
                    
                    # Send global model
                    response = {
                        'type': 'model',
                        'data': self.global_model
                    }
                    conn.send(pickle.dumps(response))
                
                elif msg_type == 'update':
                    client_id = message.get('client_id')
                    update = message.get('data')
                    
                    # Collect updates and aggregate
                    self._process_update(client_id, update)
                
                elif msg_type == 'request_model':
                    response = {
                        'type': 'model',
                        'data': self.global_model
                    }
                    conn.send(pickle.dumps(response))
            
        except Exception as e:
            print(f" Error handling client {addr}: {e}")
        finally:
            conn.close()
            print(f" Client {addr} disconnected")
    
    def _compute_update_stats(self, updates: List[np.ndarray]) -> Dict:
        """Compute statistics from collected updates."""
        if len(updates) < 2:
            return {
                'grad_var': 0.0,
                'anomaly_ratio': 0.0,
                'avg_norm': 0.0,
                'num_updates': len(updates)
            }
        
        norms = np.array([np.linalg.norm(u) for u in updates])
        grad_var = np.var(norms)
        avg_norm = np.mean(norms)
        
        std_norm = np.std(norms) if len(norms) > 1 else 1.0
        z_scores = np.abs((norms - avg_norm) / (std_norm + 1e-10))
        anomaly_ratio = np.mean(z_scores > 2.0)
        
        return {
            'grad_var': grad_var,
            'anomaly_ratio': anomaly_ratio,
            'avg_norm': avg_norm,
            'num_updates': len(updates)
        }
    
    def _select_adaptive_defense(self) -> str:
        """Select defense based on current statistics."""
        if not self.defense_selector or not self.recent_stats:
            return self.defense_type or 'fedavg'
        
        recent = self.recent_stats[-1] if self.recent_stats else {}
        recent['accuracy_change'] = self.accuracy_history[-1] - self.prev_accuracy if len(self.accuracy_history) > 1 else 0.0
        
        selected = self.defense_selector.select_defense(recent)
        
        if selected != self.defense_type:
            print(f" Switching defense from {self.defense_type} to {selected}")
            self.defense_type = selected
            self.defense = self.defense_instances.get(selected, self.defense)
        
        return selected
    
    def _process_update(self, client_id: int, update: np.ndarray):
        """Process client update (simplified for demonstration)."""
        self.current_updates.append(update)
        
        if len(self.current_updates) >= self.num_clients:
            stats = self._compute_update_stats(self.current_updates)
            self.recent_stats.append(stats)
            
            if self.adaptive_defense:
                self._select_adaptive_defense()
            
            if self.defense_type in ['krum', 'trimmed_mean']:
                aggregated = self.defense.aggregate(self.current_updates)
            else:
                sizes = [self.client_sizes.get(cid, 1) for cid in range(len(self.current_updates))]
                aggregated = self.defense.aggregate(self.current_updates, sizes)
            
            self.global_model += aggregated
            
            self.current_updates = []
            self.round += 1
            
            if self.round % 10 == 0:
                print(f" Round {self.round}: Model updated, Defense: {self.defense_type}")
    
    def start(self):
        """Start the server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(self.num_clients)
        self.running = True
        
        print(f" Federated Server started on {self.host}:{self.port}")
        print(f" Defense: {self.defense_type if self.defense_type else 'None'}")
        
        self.thread = threading.Thread(target=self._accept_clients, daemon=True)
        self.thread.start()
    
    def _accept_clients(self):
        """Accept incoming client connections."""
        while self.running:
            try:
                conn, addr = self.socket.accept()
                print(f" New connection from {addr}")
                client_thread = threading.Thread(target=self._handle_client, 
                                                 args=(conn, addr), daemon=True)
                client_thread.start()
            except Exception as e:
                if self.running:
                    print(f" Accept error: {e}")
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.socket:
            self.socket.close()
        print(" Federated Server stopped")
    
    def get_stats(self) -> Dict:
        """Get server statistics."""
        stats = {
            'round': self.round,
            'num_clients': len(self.clients),
            'defense_type': self.defense_type,
            'adaptive_defense_enabled': self.adaptive_defense,
            'accuracy_history': self.accuracy_history,
            'defense_actions': self.defense_actions,
            'recent_stats': self.recent_stats
        }
        
        if self.defense_selector:
            stats['defense_selector_stats'] = self.defense_selector.get_stats()
        
        return stats
