"""
Online Client for Federated Online Learning

Implements streaming data processing, gradient accumulation, and drift detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

from .drift_detector import DriftDetector


class DriftAlert:
    """Represents a drift alert from a client."""
    
    def __init__(self, client_id: int, timestamp: float, 
                 drift_type: str = 'concept', confidence: float = 1.0):
        self.client_id = client_id
        self.timestamp = timestamp
        self.drift_type = drift_type
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'client_id': self.client_id,
            'timestamp': self.timestamp,
            'drift_type': self.drift_type,
            'confidence': self.confidence
        }


class StreamingGradientAccumulator:
    """Accumulates gradients from streaming data."""
    
    def __init__(self, model: nn.Module, batch_size: int = 10,
                 accumulation_mode: str = 'sum'):
        self.model = model
        self.batch_size = batch_size
        self.accumulation_mode = accumulation_mode
        
        self.gradients: Dict[str, torch.Tensor] = {}
        self.count = 0
    
    def add_gradient(self, grads: Dict[str, torch.Tensor]):
        """Add gradients to accumulator."""
        for key, grad in grads.items():
            if key not in self.gradients:
                self.gradients[key] = grad.clone().detach()
            else:
                if self.accumulation_mode == 'sum':
                    self.gradients[key] += grad.clone().detach()
                elif self.accumulation_mode == 'mean':
                    self.gradients[key] = (self.gradients[key] * self.count + grad.clone().detach()) / (self.count + 1)
        
        self.count += 1
    
    def get_average_gradient(self) -> Dict[str, torch.Tensor]:
        """Get accumulated gradients."""
        if self.count == 0:
            return {}
        
        avg_grads = {}
        for key, grad in self.gradients.items():
            if self.accumulation_mode == 'sum':
                avg_grads[key] = grad / self.count
            else:
                avg_grads[key] = grad
        
        return avg_grads
    
    def has_enough_gradients(self) -> bool:
        """Check if enough gradients have been accumulated."""
        return self.count >= self.batch_size
    
    def reset(self):
        """Reset accumulator."""
        self.gradients = {}
        self.count = 0


class OnlineClient:
    """Federated online learning client."""
    
    def __init__(self, client_id: int, model: nn.Module, loss_fn: nn.Module,
                 lr: float = 0.01, batch_size: int = 10,
                 drift_detector: Optional[DriftDetector] = None):
        self.client_id = client_id
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        self.gradient_accumulator = StreamingGradientAccumulator(model, batch_size)
        self.drift_detector = drift_detector
        
        self.loss_history = deque(maxlen=100)
        self.drift_alerts: List[DriftAlert] = []
        
        self.local_updates = 0
        self.gradients_sent = 0
    
    def process_sample(self, x: torch.Tensor, y: torch.Tensor,
                      send_gradient: bool = False) -> Dict[str, Any]:
        """
        Process a single streaming sample.
        
        Args:
            x: Input tensor
            y: Target tensor
            send_gradient: Whether to send gradient after processing
        
        Returns:
            Result dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        
        grads = {name: param.grad.detach().clone() 
                 for name, param in self.model.named_parameters() 
                 if param.grad is not None}
        
        self.gradient_accumulator.add_gradient(grads)
        
        self.optimizer.step()
        self.local_updates += 1
        
        self.loss_history.append(loss.item())
        
        result = {
            'loss': loss.item(),
            'local_updates': self.local_updates,
            'gradient_ready': self.gradient_accumulator.has_enough_gradients()
        }
        
        if self.drift_detector:
            drift_detected = self.drift_detector.update(loss.item())
            if drift_detected:
                alert = DriftAlert(
                    client_id=self.client_id,
                    timestamp=np.datetime64('now').astype(float),
                    drift_type='concept',
                    confidence=1.0
                )
                self.drift_alerts.append(alert)
                result['drift_detected'] = True
                result['alert'] = alert
        
        if send_gradient or self.gradient_accumulator.has_enough_gradients():
            avg_grads = self.gradient_accumulator.get_average_gradient()
            self.gradient_accumulator.reset()
            self.gradients_sent += 1
            
            result['gradients'] = avg_grads
            result['gradients_sent'] = self.gradients_sent
        
        return result
    
    def update_global_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """Update local model with global parameters."""
        self.model.load_state_dict(global_state_dict)
    
    def get_drift_alerts(self) -> List[DriftAlert]:
        """Get accumulated drift alerts."""
        alerts = self.drift_alerts.copy()
        self.drift_alerts = []
        return alerts
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            'client_id': self.client_id,
            'local_updates': self.local_updates,
            'gradients_sent': self.gradients_sent,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'num_alerts': len(self.drift_alerts)
        }


class OnlineClientEnsemble:
    """Ensemble of online clients with different learning rates."""
    
    def __init__(self, client_id: int, model_class: nn.Module, loss_fn: nn.Module,
                 learning_rates: List[float] = None):
        self.client_id = client_id
        self.loss_fn = loss_fn
        
        if learning_rates is None:
            learning_rates = [0.001, 0.01, 0.1]
        
        self.clients = []
        for lr in learning_rates:
            model = model_class()
            client = OnlineClient(client_id, model, loss_fn, lr=lr)
            self.clients.append(client)
        
        self.best_client_idx = 0
    
    def process_sample(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Process sample with all clients and select best."""
        results = []
        
        for i, client in enumerate(self.clients):
            result = client.process_sample(x, y)
            result['client_idx'] = i
            results.append(result)
        
        losses = [r['loss'] for r in results]
        self.best_client_idx = np.argmin(losses)
        
        return results[self.best_client_idx]
    
    def get_best_model(self) -> nn.Module:
        """Get the best performing model."""
        return self.clients[self.best_client_idx].model
    
    def update_global_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """Update all clients with global model."""
        for client in self.clients:
            client.update_global_model(global_state_dict)