"""
Online Server for Federated Online Learning

Implements asynchronous gradient aggregation and learning rate scheduling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import time


class AsyncGradientAggregator:
    """Asynchronous gradient aggregator for streaming updates."""
    
    def __init__(self, model: nn.Module, aggregation_method: str = 'mean'):
        self.model = model
        self.aggregation_method = aggregation_method
        
        self.pending_gradients: Dict[int, Dict[str, torch.Tensor]] = {}
        self.gradient_counts: Dict[int, int] = defaultdict(int)
        
        self.total_updates = 0
        self.last_update_time = time.time()
    
    def receive_gradient(self, client_id: int, grads: Dict[str, torch.Tensor]):
        """
        Receive gradients from a client.
        
        Args:
            client_id: Client identifier
            grads: Gradient dictionary
        """
        self.pending_gradients[client_id] = grads
        self.gradient_counts[client_id] += 1
        self.total_updates += 1
    
    def aggregate(self, min_clients: int = 1) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate pending gradients.
        
        Args:
            min_clients: Minimum number of clients needed to aggregate
        
        Returns:
            Aggregated gradients or None if not enough clients
        """
        if len(self.pending_gradients) < min_clients:
            return None
        
        if self.aggregation_method == 'mean':
            return self._mean_aggregation()
        elif self.aggregation_method == 'weighted':
            return self._weighted_aggregation()
        elif self.aggregation_method == 'median':
            return self._median_aggregation()
        
        return self._mean_aggregation()
    
    def _mean_aggregation(self) -> Dict[str, torch.Tensor]:
        """Simple mean aggregation."""
        agg_grads = None
        
        for client_id, grads in self.pending_gradients.items():
            if agg_grads is None:
                agg_grads = {k: v.clone() for k, v in grads.items()}
            else:
                for k in agg_grads:
                    if k in grads:
                        agg_grads[k] += grads[k]
        
        if agg_grads:
            num_clients = len(self.pending_gradients)
            for k in agg_grads:
                agg_grads[k] /= num_clients
        
        self.pending_gradients.clear()
        return agg_grads
    
    def _weighted_aggregation(self) -> Dict[str, torch.Tensor]:
        """Weighted aggregation based on client update frequency."""
        total_weight = sum(self.gradient_counts.values())
        
        agg_grads = None
        
        for client_id, grads in self.pending_gradients.items():
            weight = self.gradient_counts[client_id] / total_weight
            
            if agg_grads is None:
                agg_grads = {k: v * weight for k, v in grads.items()}
            else:
                for k in agg_grads:
                    if k in grads:
                        agg_grads[k] += grads[k] * weight
        
        self.pending_gradients.clear()
        return agg_grads
    
    def _median_aggregation(self) -> Dict[str, torch.Tensor]:
        """Median aggregation for robustness."""
        if not self.pending_gradients:
            return None
        
        first_grads = list(self.pending_gradients.values())[0]
        agg_grads = {}
        
        for key in first_grads:
            grad_list = []
            for grads in self.pending_gradients.values():
                if key in grads:
                    grad_list.append(grads[key])
            
            if grad_list:
                stacked = torch.stack(grad_list)
                agg_grads[key], _ = torch.median(stacked, dim=0)
        
        self.pending_gradients.clear()
        return agg_grads
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            'pending_gradients': len(self.pending_gradients),
            'total_updates': self.total_updates,
            'last_update_time': self.last_update_time
        }


class OnlineLearningRateScheduler:
    """Adaptive learning rate scheduler for online learning."""
    
    def __init__(self, initial_lr: float = 0.01, decay_factor: float = 0.99,
                 min_lr: float = 1e-5):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        
        self.current_lr = initial_lr
        self.iteration = 0
        self.adjustments = []
    
    def step(self, loss_change: float = 0.0):
        """
        Update learning rate based on loss change.
        
        Args:
            loss_change: Change in loss from previous iteration
        """
        self.iteration += 1
        
        if loss_change > 0.1:
            self.current_lr = max(self.min_lr, self.current_lr * 0.5)
            self.adjustments.append(('drift', self.current_lr))
        elif loss_change < -0.05:
            self.current_lr = min(self.initial_lr, self.current_lr * 1.1)
            self.adjustments.append(('improvement', self.current_lr))
        else:
            self.current_lr = max(self.min_lr, self.current_lr * self.decay_factor)
            self.adjustments.append(('decay', self.current_lr))
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr
    
    def reset(self):
        """Reset scheduler state."""
        self.current_lr = self.initial_lr
        self.iteration = 0
        self.adjustments = []


class OnlineServer:
    """Federated online learning server."""
    
    def __init__(self, model: nn.Module, lr_scheduler: Optional[OnlineLearningRateScheduler] = None):
        self.model = model
        
        if lr_scheduler is None:
            self.lr_scheduler = OnlineLearningRateScheduler()
        else:
            self.lr_scheduler = lr_scheduler
        
        self.aggregator = AsyncGradientAggregator(model)
        
        self.drift_alerts: List[Dict[str, Any]] = []
        self.global_updates = 0
        
        self.accumulated_loss = 0.0
        self.loss_count = 0
        
        self.metrics = {
            'global_updates': [],
            'learning_rates': [],
            'drift_alert_counts': [],
            'aggregation_times': []
        }
    
    def receive_gradient(self, client_id: int, grads: Dict[str, torch.Tensor]):
        """Receive gradients from client."""
        self.aggregator.receive_gradient(client_id, grads)
    
    def receive_drift_alert(self, alert: Dict[str, Any]):
        """Receive drift alert from client."""
        self.drift_alerts.append(alert)
        
        if len(self.drift_alerts) > 10:
            self.drift_alerts = self.drift_alerts[-10:]
    
    def update_global_model(self, min_clients: int = 1) -> bool:
        """
        Update global model if enough gradients are available.
        
        Args:
            min_clients: Minimum number of clients needed
        
        Returns:
            True if model was updated
        """
        start_time = time.time()
        
        agg_grads = self.aggregator.aggregate(min_clients)
        
        if agg_grads is None:
            return False
        
        lr = self.lr_scheduler.get_lr()
        
        for name, param in self.model.named_parameters():
            if name in agg_grads:
                param.data -= lr * agg_grads[name]
        
        self.global_updates += 1
        
        agg_time = time.time() - start_time
        
        self.metrics['global_updates'].append(self.global_updates)
        self.metrics['learning_rates'].append(lr)
        self.metrics['drift_alert_counts'].append(len(self.drift_alerts))
        self.metrics['aggregation_times'].append(agg_time)
        
        if self.drift_alerts:
            self.lr_scheduler.step(loss_change=0.2)
            self.drift_alerts = []
        else:
            self.lr_scheduler.step(loss_change=0.0)
        
        return True
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get global model state dict."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    
    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        """Broadcast global model to clients."""
        return self.get_global_model()
    
    def handle_drift_alerts(self):
        """Handle accumulated drift alerts."""
        if len(self.drift_alerts) == 0:
            return
        
        affected_clients = set(a['client_id'] for a in self.drift_alerts)
        
        return {
            'action': 'reset_clients',
            'clients': list(affected_clients),
            'reason': f"Drift detected from {len(affected_clients)} clients"
        }
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'global_updates': self.global_updates,
            'current_lr': self.lr_scheduler.get_lr(),
            'pending_gradients': self.aggregator.get_stats()['pending_gradients'],
            'total_updates_received': self.aggregator.get_stats()['total_updates'],
            'num_drift_alerts': len(self.drift_alerts),
            'avg_aggregation_time': np.mean(self.metrics['aggregation_times']) if self.metrics['aggregation_times'] else 0.0
        }


class FederatedOnlineEvaluator:
    """Evaluator for federated online learning."""
    
    def __init__(self, server: OnlineServer):
        self.server = server
        
        self.regret = 0.0
        self.baseline_loss = 0.0
        
        self.evaluation_history = []
    
    def evaluate(self, dataloader, loss_fn: nn.Module) -> float:
        """
        Evaluate global model on validation data.
        
        Args:
            dataloader: Validation dataloader
            loss_fn: Loss function
        
        Returns:
            Validation loss
        """
        self.server.model.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                output = self.server.model(x)
                loss = loss_fn(output, y)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
        
        avg_loss = total_loss / count if count > 0 else 0.0
        
        self.evaluation_history.append({
            'update': self.server.global_updates,
            'loss': avg_loss,
            'regret': self.regret
        })
        
        return avg_loss
    
    def update_regret(self, current_loss: float, optimal_loss: float = 0.0):
        """
        Update cumulative regret.
        
        Args:
            current_loss: Current loss
            optimal_loss: Optimal loss (baseline)
        """
        self.regret += max(0.0, current_loss - optimal_loss)
    
    def get_regret(self) -> float:
        """Get cumulative regret."""
        return self.regret