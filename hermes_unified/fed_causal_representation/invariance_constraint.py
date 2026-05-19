"""
Invariance Constraints for Federated Causal Representation Learning

Implements constraints to ensure causal representations are invariant across environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class InvarianceConstraint(nn.Module):
    """Base class for invariance constraints."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, representations: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute invariance constraint loss.
        
        Args:
            representations: List of representations from different environments
        
        Returns:
            Invariance loss
        """
        raise NotImplementedError


class MMDInvariance(InvarianceConstraint):
    """Maximum Mean Discrepancy based invariance constraint."""
    
    def __init__(self, kernel_type: str = 'rbf', sigma: float = 1.0):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma
    
    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel."""
        n = x.size(0)
        m = y.size(0)
        
        x_norm = (x ** 2).sum(dim=1).view(n, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, m)
        
        dist = x_norm + y_norm - 2 * torch.matmul(x, y.t())
        
        return torch.exp(-dist / (2 * self.sigma ** 2))
    
    def _linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute linear kernel."""
        return torch.matmul(x, y.t())
    
    def compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between two distributions.
        
        Args:
            x: Samples from first distribution
            y: Samples from second distribution
        
        Returns:
            MMD value
        """
        kernel = self._rbf_kernel if self.kernel_type == 'rbf' else self._linear_kernel
        
        xx = kernel(x, x)
        yy = kernel(y, y)
        xy = kernel(x, y)
        
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd
    
    def forward(self, representations: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute MMD invariance loss across environments.
        
        Args:
            representations: List of representations from different environments
        
        Returns:
            Total MMD loss
        """
        if len(representations) < 2:
            return torch.tensor(0.0)
        
        total_mmd = 0.0
        num_pairs = 0
        
        for i in range(len(representations)):
            for j in range(i + 1, len(representations)):
                mmd = self.compute_mmd(representations[i], representations[j])
                total_mmd += mmd
                num_pairs += 1
        
        return total_mmd / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class AdversarialInvariance(InvarianceConstraint):
    """Adversarial learning based invariance constraint."""
    
    def __init__(self, feature_dim: int, num_environments: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_environments)
        )
        
        self.optimizer = None
    
    def forward(self, representations: List[torch.Tensor], 
                env_labels: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial invariance loss.
        
        Args:
            representations: List of representations from different environments
            env_labels: Environment labels (optional)
        
        Returns:
            Discriminator loss and encoder loss
        """
        if len(representations) < 2:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        all_features = torch.cat(representations, dim=0)
        
        if env_labels is None:
            env_labels = []
            for i, rep in enumerate(representations):
                env_labels.extend([i] * rep.size(0))
            env_labels = torch.tensor(env_labels, device=all_features.device)
        
        logits = self.discriminator(all_features)
        disc_loss = F.cross_entropy(logits, env_labels)
        
        uniform_target = torch.ones_like(logits) / logits.size(1)
        encoder_loss = F.kl_div(F.log_softmax(logits, dim=1), uniform_target, reduction='batchmean')
        
        return disc_loss, encoder_loss
    
    def train_discriminator(self, representations: List[torch.Tensor], 
                            num_steps: int = 5, lr: float = 1e-3):
        """
        Train discriminator to distinguish environments.
        
        Args:
            representations: List of representations
            num_steps: Number of training steps
            lr: Learning rate
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        
        for _ in range(num_steps):
            disc_loss, _ = self.forward(representations)
            
            self.optimizer.zero_grad()
            disc_loss.backward()
            self.optimizer.step()


class CORALInvariance(InvarianceConstraint):
    """CORAL (Correlation Alignment) based invariance."""
    
    def __init__(self):
        super().__init__()
    
    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix."""
        n = x.size(0)
        x_centered = x - x.mean(dim=0, keepdim=True)
        return torch.matmul(x_centered.t(), x_centered) / (n - 1)
    
    def forward(self, representations: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute CORAL loss between environments.
        
        Args:
            representations: List of representations from different environments
        
        Returns:
            CORAL loss
        """
        if len(representations) < 2:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(representations)):
            for j in range(i + 1, len(representations)):
                cov_i = self._compute_covariance(representations[i])
                cov_j = self._compute_covariance(representations[j])
                
                loss = torch.sum((cov_i - cov_j) ** 2)
                loss = loss / (4 * cov_i.size(0) ** 2)
                
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class IRMInvariance(InvarianceConstraint):
    """Invariant Risk Minimization based invariance."""
    
    def __init__(self, feature_dim: int, num_classes: int = 2, 
                 penalty_weight: float = 1.0):
        super().__init__()
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.penalty_weight = penalty_weight
    
    def _compute_penalty(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute IRM penalty."""
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.cross_entropy(logits * scale, labels)
        
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)
    
    def forward(self, representations: List[torch.Tensor], 
                labels: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IRM loss.
        
        Args:
            representations: List of representations from different environments
            labels: List of labels for each environment
        
        Returns:
            Classification loss and penalty
        """
        total_loss = 0.0
        total_penalty = 0.0
        
        for rep, lab in zip(representations, labels):
            logits = self.classifier(rep)
            loss = F.cross_entropy(logits, lab)
            penalty = self._compute_penalty(logits, lab)
            
            total_loss += loss
            total_penalty += penalty
        
        avg_loss = total_loss / len(representations)
        avg_penalty = total_penalty / len(representations)
        
        return avg_loss, avg_penalty * self.penalty_weight


class InvarianceManager:
    """Manager for multiple invariance constraints."""
    
    def __init__(self, feature_dim: int, num_environments: int = 3):
        self.mmd = MMDInvariance()
        self.coral = CORALInvariance()
        self.adversarial = AdversarialInvariance(feature_dim, num_environments)
        
        self.weights = {
            'mmd': 1.0,
            'coral': 0.5,
            'adversarial': 1.0
        }
    
    def compute_total_loss(self, representations: List[torch.Tensor],
                           labels: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total invariance loss.
        
        Args:
            representations: List of representations
            labels: Optional labels for supervised constraints
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        losses['mmd'] = self.mmd(representations) * self.weights['mmd']
        losses['coral'] = self.coral(representations) * self.weights['coral']
        
        disc_loss, enc_loss = self.adversarial(representations)
        losses['adversarial_disc'] = disc_loss
        losses['adversarial_enc'] = enc_loss * self.weights['adversarial']
        
        losses['total'] = losses['mmd'] + losses['coral'] + losses['adversarial_enc']
        
        return losses
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update constraint weights."""
        self.weights.update(new_weights)