"""
Contrastive Loss Functions for Self-Supervised Learning

Implements various contrastive loss functions for federated learning.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class NTXentLoss(torch.nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).
    
    Used in SimCLR for contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor, 
                negative_samples: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z1: First augmented view features (batch_size x feature_dim)
            z2: Second augmented view features (batch_size x feature_dim)
            negative_samples: Additional negative samples (optional)
        
        Returns:
            NT-Xent loss
        """
        batch_size = z1.size(0)
        
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        
        if negative_samples is not None:
            negative_samples = F.normalize(negative_samples, dim=1)
            z = torch.cat([z, negative_samples], dim=0)
        
        logits = torch.matmul(z, z.T) / self.temperature
        
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = False
        mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = False
        
        logits = logits[~mask].view(2 * batch_size, -1)
        
        labels = torch.arange(batch_size, 2 * batch_size, device=z.device)
        labels = torch.cat([labels, torch.arange(batch_size, device=z.device)], dim=0)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class MoCoLoss(torch.nn.Module):
    """
    Momentum Contrast Loss (MoCo).
    
    Used in MoCo v1/v2 for contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize MoCo loss.
        
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                queue: torch.Tensor) -> torch.Tensor:
        """
        Compute MoCo loss.
        
        Args:
            query: Query features from online encoder (batch_size x feature_dim)
            key: Key features from momentum encoder (batch_size x feature_dim)
            queue: Queue of negative samples (queue_size x feature_dim)
        
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


class BYOLLoss(torch.nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) Loss.
    
    Used in BYOL for self-supervised learning.
    """
    
    def __init__(self):
        """Initialize BYOL loss."""
        super().__init__()
    
    def forward(self, online_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute BYOL loss.
        
        Args:
            online_pred: Prediction from online network
            target: Target from momentum network
        
        Returns:
            BYOL loss (MSE between normalized vectors)
        """
        online_pred = F.normalize(online_pred, dim=1)
        target = F.normalize(target, dim=1)
        
        loss = 2 - 2 * (online_pred * target).sum(dim=-1).mean()
        return loss


class BarlowTwinsLoss(torch.nn.Module):
    """
    Barlow Twins Loss.
    
    Used in Barlow Twins for self-supervised learning.
    """
    
    def __init__(self, lambda_param: float = 0.0051):
        """
        Initialize Barlow Twins loss.
        
        Args:
            lambda_param: Weight for off-diagonal terms
        """
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute Barlow Twins loss.
        
        Args:
            z1: First view features
            z2: Second view features
        
        Returns:
            Barlow Twins loss
        """
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        
        z1 = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2 = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
        
        c = (z1.T @ z2) / batch_size
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        
        loss = on_diag + self.lambda_param * off_diag
        return loss
    
    def _off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Get off-diagonal elements of matrix."""
        n = x.size(0)
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SimSiamLoss(torch.nn.Module):
    """
    SimSiam Loss.
    
    Used in SimSiam for self-supervised learning.
    """
    
    def __init__(self):
        """Initialize SimSiam loss."""
        super().__init__()
    
    def forward(self, p1: torch.Tensor, z2: torch.Tensor, 
                p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """
        Compute SimSiam loss.
        
        Args:
            p1: Prediction from online network for view 1
            z2: Target from momentum encoder for view 2
            p2: Prediction from online network for view 2
            z1: Target from momentum encoder for view 1
        
        Returns:
            SimSiam loss
        """
        loss1 = -F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss2 = -F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        
        loss = (loss1 + loss2) / 2
        return loss


class InfoNCE(torch.nn.Module):
    """
    InfoNCE Loss (InfoMax Noise-Contrastive Estimation).
    
    General contrastive loss for self-supervised learning.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor: Anchor features (batch_size x feature_dim)
            positive: Positive sample features (batch_size x feature_dim)
            negatives: Negative sample features (num_negatives x feature_dim)
        
        Returns:
            InfoNCE loss
        """
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        pos_logits = torch.sum(anchor * positive, dim=1) / self.temperature
        
        neg_logits = torch.matmul(anchor, negatives.T) / self.temperature
        
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss