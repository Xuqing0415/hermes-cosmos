"""
Multimodal Federated Server

Implements the server-side logic for federated multi-modal learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional


class ModalityAggregator:
    """Aggregator for modality-specific parameters."""
    
    def __init__(self, aggregation_method: str = 'fedavg'):
        self.method = aggregation_method
    
    def aggregate(self, updates: List[Dict[str, torch.Tensor]], 
                  weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates from multiple clients.
        
        Args:
            updates: List of parameter updates from clients
            weights: Weights for weighted aggregation
        
        Returns:
            Aggregated updates
        """
        if not updates:
            return {}
        
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        
        aggregated = {}
        for key in updates[0].keys():
            tensors = []
            for i, update in enumerate(updates):
                if key in update:
                    tensors.append(weights[i] * update[key])
            
            if tensors:
                aggregated[key] = sum(tensors)
        
        return aggregated


class DifferentialPrivacyBudget:
    """Differential privacy budget management for multi-modal data."""
    
    def __init__(self, budgets: Dict[str, float] = None):
        self.budgets = budgets or {
            'image': 1.0,
            'text': 0.5,
            'tabular': 0.7
        }
        self.remaining_budgets = self.budgets.copy()
    
    def apply_dp(self, updates: Dict[str, torch.Tensor], sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy to updates based on modality.
        
        Args:
            updates: Parameter updates
            sensitivity: Sensitivity of the mechanism
        
        Returns:
            Privacy-preserving updates
        """
        noisy_updates = {}
        
        for key, update in updates.items():
            modality = self._infer_modality(key)
            epsilon = self.remaining_budgets[modality]
            
            scale = sensitivity / epsilon
            noise = torch.randn_like(update) * scale
            
            noisy_updates[key] = update + noise
            
            self.remaining_budgets[modality] *= 0.9
        
        return noisy_updates
    
    def _infer_modality(self, key: str) -> str:
        """Infer modality from parameter key."""
        if 'image' in key:
            return 'image'
        elif 'text' in key:
            return 'text'
        elif 'tabular' in key:
            return 'tabular'
        return 'image'
    
    def get_remaining_budget(self, modality: str) -> float:
        """Get remaining privacy budget for a modality."""
        return self.remaining_budgets.get(modality, 0.0)


class SensitivityClipper:
    """Adaptive sensitivity clipping for gradient updates."""
    
    def __init__(self, initial_threshold: float = 1.0, adaptive: bool = True):
        self.initial_threshold = initial_threshold
        self.adaptive = adaptive
        self.threshold = initial_threshold
    
    def clip(self, updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip updates based on L2 norm.
        
        Args:
            updates: Parameter updates
        
        Returns:
            Clipped updates
        """
        if not updates:
            return {}
        
        norms = []
        for update in updates.values():
            norms.append(torch.norm(update))
        
        if self.adaptive and norms:
            self.threshold = float(np.percentile([n.item() for n in norms], 95))
        
        clipped_updates = {}
        for key, update in updates.items():
            norm = torch.norm(update)
            if norm > self.threshold:
                clipped_updates[key] = update * (self.threshold / norm)
            else:
                clipped_updates[key] = update
        
        return clipped_updates


class MultimodalFedServer:
    """Server for federated multimodal learning."""
    
    def __init__(self, modalities: List[str], num_classes: int = 10):
        self.modalities = modalities
        self.global_shared_params: Dict[str, torch.Tensor] = {}
        
        self.aggregator = ModalityAggregator('fedavg')
        self.dp_budget = DifferentialPrivacyBudget()
        self.sensitivity_clipper = SensitivityClipper()
        
        self.round = 0
        self.total_communication = 0.0
        self.metrics = {
            'accuracy': [],
            'communication': []
        }
    
    def initialize_shared_params(self, client_shared_params: Dict[str, torch.Tensor]):
        """Initialize global shared parameters from a client."""
        self.global_shared_params = {k: v.clone() for k, v in client_shared_params.items()}
    
    def aggregate_updates(self, updates: List[Dict[str, torch.Tensor]],
                          apply_dp: bool = False) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.
        
        Args:
            updates: List of updates from clients
            apply_dp: Whether to apply differential privacy
        
        Returns:
            Updated global parameters
        """
        if not updates:
            return self.global_shared_params
        
        self.total_communication += sum(
            sum(v.numel() * v.element_size() for v in update.values())
            for update in updates
        ) / (1024 * 1024)
        
        aggregated = self.aggregator.aggregate(updates)
        
        if apply_dp:
            aggregated = self.sensitivity_clipper.clip(aggregated)
            aggregated = self.dp_budget.apply_dp(aggregated)
        
        for key in self.global_shared_params:
            if key in aggregated:
                self.global_shared_params[key] += aggregated[key]
        
        self.round += 1
        return self.global_shared_params
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global shared parameters."""
        return {k: v.clone() for k, v in self.global_shared_params.items()}
    
    def evaluate_global_model(self, clients: List, eval_dataloader) -> float:
        """Evaluate global model on evaluation data."""
        all_preds = []
        all_labels = []
        
        for batch in eval_dataloader:
            modalities = batch['modalities']
            labels = batch['labels'].cpu().numpy()
            
            features = []
            for modality in self.modalities:
                if modality in modalities:
                    data = modalities[modality]
                    if modality in clients[0].modality_encoders:
                        encoder = clients[0].modality_encoders[modality]
                        encoder.eval()
                        with torch.no_grad():
                            feature = encoder(data)
                        features.append(feature)
            
            if features:
                fusion_module = clients[0].fusion_module
                fusion_module.eval()
                classifier = clients[0].classifier
                classifier.eval()
                
                with torch.no_grad():
                    fused = fusion_module(features)
                    logits = classifier(fused)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        if all_labels:
            accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
            self.metrics['accuracy'].append(accuracy)
            return accuracy
        
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'round': self.round,
            'total_communication_mb': self.total_communication,
            'remaining_budgets': self.dp_budget.remaining_budgets,
            'metrics': self.metrics
        }