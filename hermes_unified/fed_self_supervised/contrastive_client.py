"""
Contrastive Learning Clients for Federated Self-Supervised Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple


class ContrastiveClient:
    """Base class for federated contrastive learning clients."""
    
    def __init__(self, client_id: int, device: str = 'cpu'):
        self.client_id = client_id
        self.device = device
        self.loss_history = []
        self.temperature = 0.5
        self.local_encoder = None
        self.projection_head = None
        self.optimizer = None
    
    def set_global_parameters(self, global_params: Dict[str, torch.Tensor]):
        """Set global parameters from server."""
        if self.local_encoder is not None and 'encoder' in global_params:
            self.local_encoder.load_state_dict(global_params['encoder'])
        if self.projection_head is not None and 'projection' in global_params:
            self.projection_head.load_state_dict(global_params['projection'])
    
    def get_local_parameters(self) -> Dict[str, torch.Tensor]:
        """Get local parameters for aggregation."""
        params = {}
        if self.local_encoder is not None:
            params['encoder'] = {k: v.detach().cpu().clone() for k, v in self.local_encoder.state_dict().items()}
        if self.projection_head is not None:
            params['projection'] = {k: v.detach().cpu().clone() for k, v in self.projection_head.state_dict().items()}
        return params
    
    def compute_adaptive_temperature(self, features: torch.Tensor) -> float:
        """
        Compute distribution-aware temperature based on feature uniformity.
        
        Args:
            features: Batch of feature vectors
        
        Returns:
            Adaptive temperature value
        """
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        
        pairwise_distances = torch.cdist(features_norm, features_norm, p=2)
        distances = pairwise_distances[torch.triu_indices(n, n, offset=1)]
        
        variance = torch.var(distances)
        mean_dist = torch.mean(distances)
        
        base_tau = 0.5
        adaptive_factor = 1.0
        
        if mean_dist < 0.5:
            adaptive_factor = 1.5
        elif mean_dist > 1.5:
            adaptive_factor = 0.5
        
        self.temperature = base_tau * adaptive_factor
        return self.temperature
    
    def local_train(self, dataloader, global_params: Dict[str, torch.Tensor], 
                    global_buffer=None, num_epochs: int = 1) -> Dict[str, torch.Tensor]:
        """
        Perform local contrastive training.
        
        Args:
            dataloader: Local dataloader with augmented views
            global_params: Global parameters from server
            global_buffer: Global negative sample buffer (optional)
            num_epochs: Number of local epochs
        
        Returns:
            Parameter updates
        """
        raise NotImplementedError
    
    def evaluate(self, dataloader) -> float:
        """Evaluate representation quality."""
        raise NotImplementedError


class SimCLRClient(ContrastiveClient):
    """Federated SimCLR client implementation."""
    
    def __init__(self, client_id: int, encoder: nn.Module, projection_dim: int = 128, 
                 device: str = 'cpu'):
        super().__init__(client_id, device)
        
        self.local_encoder = encoder.to(device)
        
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Linear(encoder.output_dim, projection_dim)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            list(self.local_encoder.parameters()) + list(self.projection_head.parameters()),
            lr=3e-4
        )
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, 
                     negative_samples: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross-Entropy Loss.
        
        Args:
            z1: First augmented view features
            z2: Second augmented view features
            negative_samples: Additional negative samples from global buffer
        
        Returns:
            NT-Xent loss
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        
        if negative_samples is not None:
            z = torch.cat([z, negative_samples], dim=0)
        
        z = F.normalize(z, dim=1)
        
        logits = torch.matmul(z, z.T) / self.temperature
        
        mask = torch.eye(2 * batch_size, device=z.device)
        mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
        mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
        
        logits = logits[~mask.bool()].view(2 * batch_size, -1)
        
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size - 1, labels], dim=0)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def local_train(self, dataloader, global_params: Dict[str, torch.Tensor],
                    global_buffer=None, num_epochs: int = 1) -> Dict[str, torch.Tensor]:
        """Perform local SimCLR training."""
        self.set_global_parameters(global_params)
        
        self.local_encoder.train()
        self.projection_head.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for views in dataloader:
                self.optimizer.zero_grad()
                
                x1, x2 = views[0].to(self.device), views[1].to(self.device)
                
                z1 = self.projection_head(self.local_encoder(x1))
                z2 = self.projection_head(self.local_encoder(x2))
                
                negative_samples = None
                if global_buffer is not None:
                    negative_samples = global_buffer.sample(batch_size=x1.size(0)).to(self.device)
                
                loss = self.nt_xent_loss(z1, z2, negative_samples)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.loss_history.append(avg_loss)
        
        current_params = self.get_local_parameters()
        updates = {}
        for key in current_params:
            updates[key] = {
                k: current_params[key][k] - global_params.get(key, {}).get(k, torch.zeros_like(current_params[key][k]))
                for k in current_params[key]
            }
        
        return updates
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate by computing feature statistics."""
        self.local_encoder.eval()
        
        all_features = []
        with torch.no_grad():
            for views in dataloader:
                x = views[0].to(self.device)
                features = self.local_encoder(x)
                all_features.append(features)
        
        features = torch.cat(all_features, dim=0)
        features_norm = F.normalize(features, dim=1)
        
        pairwise_dist = torch.cdist(features_norm, features_norm, p=2)
        avg_distance = torch.mean(pairwise_dist).item()
        var_distance = torch.var(pairwise_dist).item()
        
        return {
            'avg_distance': avg_distance,
            'var_distance': var_distance,
            'feature_dim': features.size(1)
        }


class MoCoClient(ContrastiveClient):
    """Federated MoCo (Momentum Contrast) client implementation."""
    
    def __init__(self, client_id: int, encoder: nn.Module, momentum_encoder: nn.Module,
                 projection_dim: int = 128, device: str = 'cpu'):
        super().__init__(client_id, device)
        
        self.local_encoder = encoder.to(device)
        self.momentum_encoder = momentum_encoder.to(device)
        
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Linear(encoder.output_dim, projection_dim)
        ).to(device)
        
        self.momentum_projection = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Linear(encoder.output_dim, projection_dim)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            list(self.local_encoder.parameters()) + list(self.projection_head.parameters()),
            lr=3e-4
        )
        
        self.momentum = 0.999
        
        self._sync_momentum_encoder()
    
    def _sync_momentum_encoder(self):
        """Sync momentum encoder with online encoder."""
        self.momentum_encoder.load_state_dict(self.local_encoder.state_dict())
        self.momentum_projection.load_state_dict(self.projection_head.state_dict())
    
    def _update_momentum_encoder(self):
        """Update momentum encoder with exponential moving average."""
        for online, momentum in zip(
            self.local_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            momentum.data = self.momentum * momentum.data + (1 - self.momentum) * online.data
        
        for online, momentum in zip(
            self.projection_head.parameters(),
            self.momentum_projection.parameters()
        ):
            momentum.data = self.momentum * momentum.data + (1 - self.momentum) * online.data
    
    def moco_loss(self, query: torch.Tensor, key: torch.Tensor, 
                  queue: torch.Tensor) -> torch.Tensor:
        """
        MoCo contrastive loss.
        
        Args:
            query: Query features from online encoder
            key: Key features from momentum encoder
            queue: Queue of negative samples
        
        Returns:
            MoCo loss
        """
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        queue = F.normalize(queue, dim=1)
        
        pos_logits = torch.sum(query * key, dim=1, keepdim=True) / self.temperature
        neg_logits = torch.matmul(query, queue.T) / self.temperature
        
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def local_train(self, dataloader, global_params: Dict[str, torch.Tensor],
                    global_buffer=None, num_epochs: int = 1) -> Dict[str, torch.Tensor]:
        """Perform local MoCo training."""
        self.set_global_parameters(global_params)
        
        self.local_encoder.train()
        self.projection_head.train()
        self.momentum_encoder.eval()
        self.momentum_projection.eval()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for views in dataloader:
                self.optimizer.zero_grad()
                
                x_q, x_k = views[0].to(self.device), views[1].to(self.device)
                
                q = self.projection_head(self.local_encoder(x_q))
                
                with torch.no_grad():
                    k = self.momentum_projection(self.momentum_encoder(x_k))
                    k = k.detach()
                
                queue = global_buffer.buffer if global_buffer is not None else None
                if queue is None:
                    queue = torch.randn(65536, q.size(1)).to(self.device)
                
                loss = self.moco_loss(q, k, queue)
                loss.backward()
                self.optimizer.step()
                
                self._update_momentum_encoder()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.loss_history.append(avg_loss)
        
        current_params = self.get_local_parameters()
        updates = {}
        for key in current_params:
            updates[key] = {
                k: current_params[key][k] - global_params.get(key, {}).get(k, torch.zeros_like(current_params[key][k]))
                for k in current_params[key]
            }
        
        return updates
    
    def get_local_parameters(self) -> Dict[str, torch.Tensor]:
        """Get local parameters (only online encoder and projection head)."""
        params = super().get_local_parameters()
        return params