"""
Modality Aligner

Implements cross-modal alignment techniques for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional


class CrossModalAligner(nn.Module):
    """Base class for cross-modal alignment."""
    
    def __init__(self):
        super().__init__()
    
    def align(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Align different modalities."""
        raise NotImplementedError


class ContrastiveAligner(CrossModalAligner):
    """Contrastive learning based cross-modal alignment."""
    
    def __init__(self, feature_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss for alignment."""
        projections = [self.projection_head(f) for f in features]
        
        all_projections = torch.cat(projections, dim=0)
        normalized = F.normalize(all_projections, dim=1)
        
        similarity_matrix = torch.matmul(normalized, normalized.T) / self.temperature
        
        batch_size = features[0].size(0)
        num_modalities = len(features)
        
        labels = torch.arange(batch_size).repeat(num_modalities)
        labels = labels.to(features[0].device)
        
        mask = torch.eye(batch_size * num_modalities, device=features[0].device)
        for i in range(num_modalities):
            mask[i * batch_size:(i + 1) * batch_size, i * batch_size:(i + 1) * batch_size] = 0
        
        loss = F.cross_entropy(similarity_matrix, labels, reduction='none')
        loss = loss * mask.sum(dim=1)
        
        return loss.mean()


class CLIPStyleAligner(CrossModalAligner):
    """CLIP-style contrastive aligner for multimodal representation learning."""
    
    def __init__(self, feature_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.tabular_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, image_features: Optional[torch.Tensor] = None,
                text_features: Optional[torch.Tensor] = None,
                tabular_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute CLIP-style contrastive loss.
        
        Args:
            image_features: Image modality features
            text_features: Text modality features
            tabular_features: Tabular modality features
        
        Returns:
            Contrastive alignment loss
        """
        projections = []
        modalities = []
        
        if image_features is not None:
            projections.append(F.normalize(self.image_projection(image_features), dim=1))
            modalities.append('image')
        
        if text_features is not None:
            projections.append(F.normalize(self.text_projection(text_features), dim=1))
            modalities.append('text')
        
        if tabular_features is not None:
            projections.append(F.normalize(self.tabular_projection(tabular_features), dim=1))
            modalities.append('tabular')
        
        if len(projections) < 2:
            return torch.tensor(0.0)
        
        loss = 0.0
        num_pairs = 0
        
        for i in range(len(projections)):
            for j in range(i + 1, len(projections)):
                logits = torch.matmul(projections[i], projections[j].T) / self.temperature
                batch_size = projections[i].size(0)
                
                labels = torch.arange(batch_size).to(projections[i].device)
                
                loss_i_j = F.cross_entropy(logits, labels)
                loss_j_i = F.cross_entropy(logits.T, labels)
                
                loss += (loss_i_j + loss_j_i) / 2
                num_pairs += 1
        
        return loss / num_pairs if num_pairs > 0 else loss


class FederatedAlignerServer:
    """Server for federated cross-modal alignment."""
    
    def __init__(self, feature_dim: int = 64):
        self.aligner = CLIPStyleAligner(feature_dim)
        self.optimizer = torch.optim.Adam(self.aligner.parameters(), lr=1e-4)
        
        self.round = 0
        self.alignment_loss_history = []
    
    def aggregate_alignment_updates(self, updates: List[Dict[str, torch.Tensor]]):
        """Aggregate alignment parameter updates."""
        if not updates:
            return
        
        aggregated = {}
        for key in updates[0].keys():
            aggregated[key] = torch.mean(
                torch.stack([update[key] for update in updates]),
                dim=0
            )
        
        current_state = self.aligner.state_dict()
        for key in aggregated:
            if key in current_state:
                current_state[key] += aggregated[key]
        self.aligner.load_state_dict(current_state)
    
    def train_alignment(self, modality_features: Dict[str, torch.Tensor]):
        """Train the aligner on aggregated modality features."""
        self.optimizer.zero_grad()
        
        loss = self.aligner(
            image_features=modality_features.get('image'),
            text_features=modality_features.get('text'),
            tabular_features=modality_features.get('tabular')
        )
        
        loss.backward()
        self.optimizer.step()
        
        self.alignment_loss_history.append(loss.item())
        self.round += 1
        
        return loss.item()
    
    def get_alignment_parameters(self) -> Dict[str, torch.Tensor]:
        """Get alignment parameters for distribution to clients."""
        return {k: v.detach().cpu().clone() for k, v in self.aligner.state_dict().items()}
    
    def set_alignment_parameters(self, params: Dict[str, torch.Tensor]):
        """Set alignment parameters."""
        self.aligner.load_state_dict(params)
    
    def compute_alignment_metrics(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute alignment quality metrics."""
        features_list = [v for v in modality_features.values() if v is not None]
        
        if len(features_list) < 2:
            return {'similarity': 0.0}
        
        metrics = {}
        normalized_features = [F.normalize(f, dim=1) for f in features_list]
        
        for i, (name_i, feat_i) in enumerate(modality_features.items()):
            for j, (name_j, feat_j) in enumerate(modality_features.items()):
                if i < j and feat_i is not None and feat_j is not None:
                    sim = torch.mean(torch.diag(torch.matmul(
                        F.normalize(feat_i, dim=1),
                        F.normalize(feat_j, dim=1).T
                    )))
                    metrics[f'{name_i}_{name_j}_similarity'] = float(sim)
        
        return metrics