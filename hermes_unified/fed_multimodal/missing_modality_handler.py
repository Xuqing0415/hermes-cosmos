"""
Missing Modality Handler

Implements strategies for handling missing modalities in federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional


class MissingModalityHandler:
    """Base class for handling missing modalities."""
    
    def __init__(self):
        pass
    
    def handle(self, available_features: Dict[str, torch.Tensor], 
               missing_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """Handle missing modalities."""
        raise NotImplementedError


class ModalityMasker(MissingModalityHandler):
    """Random modality masking for robustness training."""
    
    def __init__(self, mask_prob: float = 0.1):
        super().__init__()
        self.mask_prob = mask_prob
    
    def handle(self, available_features: Dict[str, torch.Tensor],
               missing_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """Randomly mask modalities during training."""
        masked_features = {}
        
        for modality, feature in available_features.items():
            if torch.rand(1).item() < self.mask_prob:
                masked_features[modality] = torch.zeros_like(feature)
            else:
                masked_features[modality] = feature
        
        return masked_features


class KnowledgeDistillationFiller(MissingModalityHandler):
    """Fill missing modalities using knowledge distillation from teacher clients."""
    
    def __init__(self, num_modalities: int = 3, feature_dim: int = 64):
        super().__init__()
        self.distiller = nn.Sequential(
            nn.Linear(num_modalities * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.distiller.parameters(), lr=1e-3)
    
    def train_distiller(self, source_features: Dict[str, torch.Tensor], 
                        target_feature: torch.Tensor):
        """Train the distiller to predict missing modality."""
        self.distiller.train()
        self.optimizer.zero_grad()
        
        concatenated = torch.cat([source_features[m] for m in sorted(source_features.keys())], dim=1)
        prediction = self.distiller(concatenated)
        
        loss = F.mse_loss(prediction, target_feature)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def handle(self, available_features: Dict[str, torch.Tensor],
               missing_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """Generate features for missing modalities using distillation."""
        if not available_features or not missing_modalities:
            return available_features
        
        self.distiller.eval()
        filled_features = available_features.copy()
        
        concatenated = torch.cat([available_features[m] for m in sorted(available_features.keys())], dim=1)
        
        with torch.no_grad():
            predicted = self.distiller(concatenated)
        
        for missing in missing_modalities:
            filled_features[missing] = predicted
        
        return filled_features


class VirtualModalityGenerator(MissingModalityHandler):
    """Generate virtual representations for missing modalities."""
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.generators = {}
    
    def add_generator(self, modality: str, source_modalities: List[str]):
        """Add a generator for a specific modality."""
        input_dim = len(source_modalities) * feature_dim
        self.generators[modality] = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh()
        )
    
    def train_generator(self, modality: str, source_features: Dict[str, torch.Tensor],
                        target_feature: torch.Tensor, optimizer: torch.optim.Optimizer = None):
        """Train a specific generator."""
        generator = self.generators[modality]
        
        if optimizer is None:
            optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
        
        generator.train()
        optimizer.zero_grad()
        
        concatenated = torch.cat([source_features[m] for m in sorted(source_features.keys())], dim=1)
        generated = generator(concatenated)
        
        loss = F.mse_loss(generated, target_feature)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def handle(self, available_features: Dict[str, torch.Tensor],
               missing_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """Generate virtual features for missing modalities."""
        if not available_features or not missing_modalities:
            return available_features
        
        filled_features = available_features.copy()
        
        for missing in missing_modalities:
            if missing in self.generators:
                generator = self.generators[missing]
                generator.eval()
                
                source_modalities = [m for m in available_features.keys()]
                if source_modalities:
                    concatenated = torch.cat([available_features[m] for m in sorted(source_modalities)], dim=1)
                    
                    with torch.no_grad():
                        generated = generator(concatenated)
                    
                    filled_features[missing] = generated
        
        return filled_features


class ModalityCompletionNetwork(MissingModalityHandler):
    """Complete missing modalities using a neural network."""
    
    def __init__(self, feature_dim: int = 64, num_modalities: int = 3):
        super().__init__()
        self.completion_net = nn.Sequential(
            nn.Linear(num_modalities * feature_dim, num_modalities * feature_dim),
            nn.ReLU(),
            nn.Linear(num_modalities * feature_dim, num_modalities * feature_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.completion_net.parameters(), lr=1e-3)
    
    def train(self, features: Dict[str, torch.Tensor], masks: Dict[str, bool]):
        """Train the completion network."""
        self.completion_net.train()
        self.optimizer.zero_grad()
        
        all_features = []
        modality_order = ['image', 'text', 'tabular']
        
        for modality in modality_order:
            if modality in features:
                all_features.append(features[modality])
            else:
                all_features.append(torch.zeros_like(list(features.values())[0]))
        
        concatenated = torch.cat(all_features, dim=1)
        reconstructed = self.completion_net(concatenated)
        
        loss = 0.0
        count = 0
        
        start = 0
        for modality in modality_order:
            end = start + features[modality_order[0]].size(1)
            
            if modality in features and masks.get(modality, False):
                original = features[modality]
                recon = reconstructed[:, start:end]
                loss += F.mse_loss(recon, original)
                count += 1
            
            start = end
        
        if count > 0:
            loss /= count
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def handle(self, available_features: Dict[str, torch.Tensor],
               missing_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """Complete missing modalities."""
        if not available_features:
            return available_features
        
        self.completion_net.eval()
        
        modality_order = ['image', 'text', 'tabular']
        all_features = []
        
        for modality in modality_order:
            if modality in available_features:
                all_features.append(available_features[modality])
            else:
                all_features.append(torch.zeros_like(list(available_features.values())[0]))
        
        concatenated = torch.cat(all_features, dim=1)
        
        with torch.no_grad():
            reconstructed = self.completion_net(concatenated)
        
        filled_features = available_features.copy()
        start = 0
        
        for modality in modality_order:
            end = start + list(available_features.values())[0].size(1)
            
            if modality in missing_modalities:
                filled_features[modality] = reconstructed[:, start:end]
            
            start = end
        
        return filled_features