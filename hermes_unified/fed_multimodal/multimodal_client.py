"""
Multimodal Federated Client

Implements the client-side logic for federated multi-modal learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple


class ModalityEncoder(nn.Module):
    """Base class for modality-specific encoders."""
    
    def __init__(self, modality_type: str):
        super().__init__()
        self.modality_type = modality_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ImageEncoder(ModalityEncoder):
    """Encoder for image modality."""
    
    def __init__(self, input_channels: int = 1, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__('image')
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class TextEncoder(ModalityEncoder):
    """Encoder for text modality."""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__('text')
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(x)
        _, (hidden, _) = self.lstm(embeddings)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)


class TabularEncoder(ModalityEncoder):
    """Encoder for tabular data modality."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__('tabular')
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(x)


class SharedPrivateEncoder(nn.Module):
    """Shared-private encoder architecture for modality-specific learning."""
    
    def __init__(self, modality_type: str, shared_dim: int = 64, private_dim: int = 64):
        super().__init__()
        self.modality_type = modality_type
        
        if modality_type == 'image':
            self.private_encoder = ImageEncoder(output_dim=private_dim)
        elif modality_type == 'text':
            self.private_encoder = TextEncoder(output_dim=private_dim)
        elif modality_type == 'tabular':
            self.private_encoder = TabularEncoder(output_dim=private_dim)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(private_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
    
    def forward(self, x: torch.Tensor, return_private: bool = False) -> torch.Tensor:
        private_repr = self.private_encoder(x)
        shared_repr = self.shared_encoder(private_repr)
        
        if return_private:
            return shared_repr, private_repr
        return shared_repr
    
    def get_shared_parameters(self):
        """Get parameters that should be shared (federated)."""
        return self.shared_encoder.parameters()
    
    def get_private_parameters(self):
        """Get parameters that should remain private (local only)."""
        return self.private_encoder.parameters()


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for feature fusion."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        queries = self.query_proj(query).unsqueeze(1)
        keys = self.key_proj(key).unsqueeze(1)
        values = self.value_proj(value).unsqueeze(1)
        
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (queries.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attended = torch.bmm(attn_weights, values).squeeze(1)
        return attended, attn_weights


class MultimodalFusion(nn.Module):
    """Multimodal fusion module combining multiple modalities."""
    
    def __init__(self, num_modalities: int, feature_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.cross_attn = CrossModalAttention(feature_dim)
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_modalities * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        if len(modality_features) == 1:
            return modality_features[0]
        
        attended_features = []
        reference = modality_features[0]
        
        for feat in modality_features:
            attended, _ = self.cross_attn(reference, feat, feat)
            attended_features.append(attended)
        
        concatenated = torch.cat(attended_features, dim=1)
        
        actual_dim = concatenated.size(1)
        if actual_dim != self.num_modalities * self.feature_dim:
            padding = torch.zeros(
                concatenated.size(0),
                self.num_modalities * self.feature_dim - actual_dim,
                device=concatenated.device
            )
            concatenated = torch.cat([concatenated, padding], dim=1)
        
        return self.fusion_layer(concatenated)


class MultimodalFedClient:
    """Federated client for multimodal learning."""
    
    def __init__(self, client_id: int, available_modalities: List[str], 
                 num_classes: int = 10, device: str = 'cpu'):
        self.client_id = client_id
        self.available_modalities = available_modalities
        self.device = device
        
        self.modality_encoders = {}
        for modality in available_modalities:
            self.modality_encoders[modality] = SharedPrivateEncoder(modality).to(device)
        
        self.fusion_module = MultimodalFusion(len(available_modalities)).to(device)
        self.classifier = nn.Linear(64, num_classes).to(device)
        
        self.optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
        
        self.loss_history = []
    
    def parameters(self):
        """Get all parameters."""
        params = []
        for encoder in self.modality_encoders.values():
            params.extend(encoder.parameters())
        params.extend(self.fusion_module.parameters())
        params.extend(self.classifier.parameters())
        return params
    
    def get_shared_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Get shared parameters for aggregation."""
        shared_params = {}
        for modality, encoder in self.modality_encoders.items():
            for name, param in encoder.shared_encoder.named_parameters():
                shared_params[f"{modality}_shared_{name}"] = param.detach().cpu().clone()
        for name, param in self.fusion_module.named_parameters():
            shared_params[f"fusion_{name}"] = param.detach().cpu().clone()
        return shared_params
    
    def set_shared_parameters(self, shared_params: Dict[str, torch.Tensor]):
        """Set shared parameters from server."""
        for modality, encoder in self.modality_encoders.items():
            encoder_dict = {}
            for name, param in encoder.shared_encoder.named_parameters():
                key = f"{modality}_shared_{name}"
                if key in shared_params:
                    encoder_dict[name] = shared_params[key].to(self.device)
            encoder.shared_encoder.load_state_dict(encoder_dict)
        
        fusion_dict = {}
        for name, param in self.fusion_module.named_parameters():
            key = f"fusion_{name}"
            if key in shared_params:
                fusion_dict[name] = shared_params[key].to(self.device)
        self.fusion_module.load_state_dict(fusion_dict)
    
    def local_train(self, dataloader, global_shared_params: Dict[str, torch.Tensor],
                    num_epochs: int = 1, mask_prob: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Perform local training with optional modality masking.
        
        Args:
            dataloader: Local dataloader
            global_shared_params: Global shared parameters
            num_epochs: Number of local epochs
            mask_prob: Probability of masking a modality
        
        Returns:
            Parameter updates
        """
        self.set_shared_parameters(global_shared_params)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                modalities = batch['modalities']
                labels = batch['labels'].to(self.device)
                
                features = []
                for modality in self.available_modalities:
                    if modality in modalities:
                        if torch.rand(1).item() < mask_prob:
                            continue
                        data = modalities[modality].to(self.device)
                        feature = self.modality_encoders[modality](data)
                        features.append(feature)
                
                if not features:
                    continue
                
                fused = self.fusion_module(features)
                logits = self.classifier(fused)
                
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.loss_history.append(avg_loss)
            print(f"Client {self.client_id}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        current_shared = self.get_shared_parameters_dict()
        updates = {
            key: current_shared[key] - global_shared_params.get(key, torch.zeros_like(current_shared[key]))
            for key in current_shared
        }
        
        return updates
    
    def evaluate(self, dataloader) -> float:
        """Evaluate on local data."""
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                modalities = batch['modalities']
                labels = batch['labels'].to(self.device)
                
                features = []
                for modality in self.available_modalities:
                    if modality in modalities:
                        data = modalities[modality].to(self.device)
                        feature = self.modality_encoders[modality](data)
                        features.append(feature)
                
                if not features:
                    continue
                
                fused = self.fusion_module(features)
                logits = self.classifier(fused)
                preds = torch.argmax(logits, dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        self.train()
        return correct / total if total > 0 else 0.0
    
    def eval(self):
        """Set to evaluation mode."""
        for encoder in self.modality_encoders.values():
            encoder.eval()
        self.fusion_module.eval()
        self.classifier.eval()
    
    def train(self):
        """Set to training mode."""
        for encoder in self.modality_encoders.values():
            encoder.train()
        self.fusion_module.train()
        self.classifier.train()