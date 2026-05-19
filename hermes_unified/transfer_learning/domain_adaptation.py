"""
Domain Adaptation for Federated Transfer Learning

Implements domain adaptation techniques for federated learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss for domain adaptation."""
    
    def __init__(self, kernel_type: str = 'rbf', sigma: float = 1.0):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma
    
    def _rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        n = x1.size(0)
        m = x2.size(0)
        x1_norm = x1.norm(dim=1).view(n, 1).expand(n, m)
        x2_norm = x2.norm(dim=1).view(1, m).expand(n, m)
        dist = x1_norm + x2_norm - 2 * torch.matmul(x1, x2.t())
        return torch.exp(-dist / (2 * self.sigma ** 2))
    
    def _linear_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x1, x2.t())
    
    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        kernel = self._rbf_kernel if self.kernel_type == 'rbf' else self._linear_kernel
        xx = kernel(source_features, source_features)
        yy = kernel(target_features, target_features)
        xy = kernel(source_features, target_features)
        return xx.mean() + yy.mean() - 2 * xy.mean()


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial domain adaptation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DomainAdversarialAdaptor(nn.Module):
    """Domain adversarial adaptation module."""
    
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module, input_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_discriminator = DomainDiscriminator(input_dim)
        
        self.optimizer_feature = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)
        self.optimizer_discriminator = torch.optim.Adam(self.domain_discriminator.parameters(), lr=1e-4)
    
    def train_step(self, source_data: tuple, target_data: tuple, lambda_domain: float = 1.0):
        x_source, y_source = source_data
        x_target = target_data[0]
        
        source_features = self.feature_extractor(x_source)
        target_features = self.feature_extractor(x_target)
        
        self.optimizer_classifier.zero_grad()
        self.optimizer_feature.zero_grad()
        source_preds = self.classifier(source_features)
        classification_loss = F.cross_entropy(source_preds, y_source)
        classification_loss.backward(retain_graph=True)
        self.optimizer_classifier.step()
        
        self.optimizer_discriminator.zero_grad()
        domain_labels_source = torch.ones(source_features.size(0), 1).to(source_features.device)
        domain_labels_target = torch.zeros(target_features.size(0), 1).to(target_features.device)
        source_domain_preds = self.domain_discriminator(source_features.detach())
        target_domain_preds = self.domain_discriminator(target_features.detach())
        disc_loss = (F.binary_cross_entropy(source_domain_preds, domain_labels_source) +
                     F.binary_cross_entropy(target_domain_preds, domain_labels_target)) / 2
        disc_loss.backward()
        self.optimizer_discriminator.step()
        
        self.optimizer_feature.zero_grad()
        source_domain_preds = self.domain_discriminator(source_features)
        target_domain_preds = self.domain_discriminator(target_features)
        adv_loss = (F.binary_cross_entropy(source_domain_preds, domain_labels_target) +
                    F.binary_cross_entropy(target_domain_preds, domain_labels_source)) / 2
        (lambda_domain * adv_loss).backward()
        self.optimizer_feature.step()
        
        return {
            'classification_loss': classification_loss.item(),
            'discriminator_loss': disc_loss.item(),
            'adversarial_loss': adv_loss.item()
        }


class DomainAdaptationServer:
    """Server for federated domain adaptation."""
    
    def __init__(self, global_model):
        self.global_model = global_model
        self.global_model.eval()
    
    def broadcast_model(self):
        return {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def aggregate_adapted_models(self, client_models: List[Dict[str, torch.Tensor]]):
        if not client_models:
            return {}
        
        aggregated = {}
        for key in client_models[0].keys():
            aggregated[key] = torch.mean(torch.stack([m[key] for m in client_models]), dim=0)
        
        return aggregated


class DomainAdaptationClient:
    """Client for federated domain adaptation."""
    
    def __init__(self, model, dataset, device='cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    
    def adapt(self, global_weights: Dict[str, torch.Tensor], num_epochs: int = 1):
        self.model.load_state_dict({k: v.to(self.device) for k, v in global_weights.items()})
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                self.optimizer.step()
        
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}