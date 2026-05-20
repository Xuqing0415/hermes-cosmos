"""
Federated Online Meta-Learning (FO-MAML)

Implements federated online meta-learning for concept drift adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class FOMAMLClient:
    """FOMAML client for federated online meta-learning."""
    
    def __init__(self, client_id: int, model: nn.Module, loss_fn: nn.Module,
                 inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.client_id = client_id
        self.model = model
        self.loss_fn = loss_fn
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        
        self.meta_grads: Dict[str, torch.Tensor] = {}
        self.inner_updates = 0
        self.meta_updates = 0
    
    def inner_update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform inner loop update (local adaptation).
        
        Args:
            x: Input tensor
            y: Target tensor
        """
        self.model.train()
        
        params = dict(self.model.named_parameters())
        grads = {}
        
        output = self.model(x)
        loss = self.loss_fn(output, y)
        
        gradients = torch.autograd.grad(loss, params.values())
        
        for (name, param), grad in zip(params.items(), gradients):
            grads[name] = grad.detach().clone()
            param.data -= self.inner_lr * grad
        
        self.inner_updates += 1
        
        return {'loss': loss.item()}
    
    def compute_meta_gradient(self, x: torch.Tensor, y: torch.Tensor,
                             initial_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute meta-gradient for outer loop.
        
        Args:
            x: Input tensor
            y: Target tensor
            initial_params: Initial parameters before inner updates
        
        Returns:
            Meta-gradients
        """
        self.model.train()
        
        output = self.model(x)
        loss = self.loss_fn(output, y)
        
        params = dict(self.model.named_parameters())
        gradients = torch.autograd.grad(loss, params.values())
        
        meta_grads = {}
        for (name, param), grad in zip(params.items(), gradients):
            meta_grads[name] = grad.detach().clone()
        
        self.meta_updates += 1
        
        return meta_grads
    
    def reset_to_initial(self, initial_params: Dict[str, torch.Tensor]):
        """Reset model to initial parameters."""
        self.model.load_state_dict(initial_params)
    
    def get_meta_gradients(self) -> Dict[str, torch.Tensor]:
        """Get accumulated meta-gradients."""
        return self.meta_grads


class FOMAMLServer:
    """FOMAML server for federated online meta-learning."""
    
    def __init__(self, model: nn.Module, meta_lr: float = 0.001):
        self.model = model
        self.meta_lr = meta_lr
        
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
        self.meta_gradients: Dict[str, torch.Tensor] = {}
        self.meta_updates = 0
        
        self.clients: Dict[int, FOMAMLClient] = {}
    
    def register_client(self, client_id: int, client: FOMAMLClient):
        """Register a client."""
        self.clients[client_id] = client
    
    def distribute_initial_params(self) -> Dict[str, torch.Tensor]:
        """Distribute initial parameters to clients."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    
    def receive_meta_gradients(self, client_id: int, meta_grads: Dict[str, torch.Tensor]):
        """Receive meta-gradients from client."""
        for name, grad in meta_grads.items():
            if name not in self.meta_gradients:
                self.meta_gradients[name] = grad.clone()
            else:
                self.meta_gradients[name] += grad.clone()
    
    def aggregate_and_update(self):
        """Aggregate meta-gradients and update global model."""
        if not self.meta_gradients:
            return
        
        num_clients = len(self.clients)
        
        self.meta_optimizer.zero_grad()
        
        for name, param in self.model.named_parameters():
            if name in self.meta_gradients:
                avg_grad = self.meta_gradients[name] / num_clients
                param.grad = avg_grad
        
        self.meta_optimizer.step()
        
        self.meta_updates += 1
        self.meta_gradients = {}
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get global model state dict."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'meta_updates': self.meta_updates,
            'num_clients': len(self.clients),
            'current_lr': self.meta_lr
        }


class FederatedOnlineMetaLearner:
    """Complete federated online meta-learning system."""
    
    def __init__(self, model_class: nn.Module, loss_fn: nn.Module,
                 num_clients: int = 10, inner_lr: float = 0.01,
                 meta_lr: float = 0.001):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.num_clients = num_clients
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        
        self.global_model = model_class()
        self.server = FOMAMLServer(self.global_model, meta_lr=meta_lr)
        
        self.clients: Dict[int, FOMAMLClient] = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize clients."""
        for client_id in range(self.num_clients):
            model = self.model_class()
            model.load_state_dict(self.global_model.state_dict())
            
            client = FOMAMLClient(
                client_id=client_id,
                model=model,
                loss_fn=self.loss_fn,
                inner_lr=self.inner_lr,
                meta_lr=self.meta_lr
            )
            
            self.clients[client_id] = client
            self.server.register_client(client_id, client)
    
    def run_meta_iteration(self, client_data: Dict[int, Any],
                          num_inner_updates: int = 1):
        """
        Run one meta-iteration.
        
        Args:
            client_data: Dictionary of client data {client_id: (x, y)}
            num_inner_updates: Number of inner loop updates
        
        Returns:
            Meta-iteration results
        """
        initial_params = self.server.distribute_initial_params()
        
        for client_id, client in self.clients.items():
            if client_id in client_data:
                client.reset_to_initial(initial_params)
                
                x, y = client_data[client_id]
                
                for _ in range(num_inner_updates):
                    client.inner_update(x, y)
                
                meta_grads = client.compute_meta_gradient(x, y, initial_params)
                self.server.receive_meta_gradients(client_id, meta_grads)
        
        self.server.aggregate_and_update()
        
        for client in self.clients.values():
            client.reset_to_initial(self.server.get_global_model())
        
        return {
            'meta_updates': self.server.get_stats()['meta_updates'],
            'num_clients_updated': len(client_data)
        }
    
    def evaluate(self, dataloader) -> float:
        """Evaluate global model."""
        self.global_model.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                output = self.global_model(x)
                loss = self.loss_fn(output, y)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
        
        return total_loss / count if count > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'meta_updates': self.server.get_stats()['meta_updates'],
            'num_clients': self.num_clients,
            'inner_lr': self.inner_lr,
            'meta_lr': self.meta_lr
        }