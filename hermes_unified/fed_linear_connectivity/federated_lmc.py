"""
Federated Linear Mode Connectivity Core Module

Implements the main coordinator and client snapshot classes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .connectivity_metrics import (
    InterpolationLossEvaluator,
    LossLandscapeAnalyzer,
    LinearConnectivityMetrics
)


class ClientModelSnapshot:
    """Snapshot of a client's model at a specific round."""
    
    def __init__(self, client_id: int, round_idx: int, 
                 model_state_dict: Dict[str, torch.Tensor],
                 sample_count: int, loss: Optional[float] = None):
        self.client_id = client_id
        self.round_idx = round_idx
        self.model_state_dict = model_state_dict
        self.sample_count = sample_count
        self.loss = loss
        self.timestamp = np.datetime64('now')
    
    def get_model_copy(self, model_class: nn.Module) -> nn.Module:
        """Create a model instance from the snapshot."""
        model = model_class()
        model.load_state_dict(self.model_state_dict)
        return model
    
    def add_noise(self, noise_scale: float = 0.01):
        """Add differential privacy noise to model parameters."""
        for key in self.model_state_dict:
            noise = torch.randn_like(self.model_state_dict[key]) * noise_scale
            self.model_state_dict[key] = self.model_state_dict[key] + noise


class FederatedLMC:
    """Federated Linear Mode Connectivity analyzer."""
    
    def __init__(self, model_class: nn.Module, loss_fn: nn.Module,
                 num_interpolation_points: int = 10, device: str = 'cpu'):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.num_points = num_interpolation_points
        self.device = device
        
        self.evaluator = InterpolationLossEvaluator(model_class, num_interpolation_points)
        self.analyzer = LossLandscapeAnalyzer(num_interpolation_points)
        self.metrics = LinearConnectivityMetrics()
        
        self.client_snapshots: Dict[int, List[ClientModelSnapshot]] = defaultdict(list)
        self.global_models: List[Dict[str, torch.Tensor]] = []
        
        self.round_analyses = []
    
    def add_client_snapshot(self, snapshot: ClientModelSnapshot):
        """Add a client model snapshot."""
        self.client_snapshots[snapshot.client_id].append(snapshot)
    
    def add_global_model(self, round_idx: int, model_state_dict: Dict[str, torch.Tensor]):
        """Add a global model snapshot."""
        self.global_models.append({
            'round_idx': round_idx,
            'state_dict': model_state_dict
        })
    
    def evaluate_client_connectivity(self, client_snapshot: ClientModelSnapshot,
                                     global_state_dict: Dict[str, torch.Tensor],
                                     dataloader) -> Dict[str, Any]:
        """
        Evaluate connectivity between client model and global model.
        
        Args:
            client_snapshot: Client model snapshot
            global_state_dict: Global model state dict
            dataloader: Validation dataloader
        
        Returns:
            Connectivity analysis results
        """
        client_model = client_snapshot.get_model_copy(self.model_class)
        global_model = self.model_class()
        global_model.load_state_dict(global_state_dict)
        
        return self.analyzer.analyze_landscape(
            client_model, global_model, dataloader, self.loss_fn, self.device
        )
    
    def evaluate_round_connectivity(self, round_idx: int, dataloader,
                                   sample_clients: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Evaluate connectivity for a specific round.
        
        Args:
            round_idx: Round index to evaluate
            dataloader: Validation dataloader
            sample_clients: List of client IDs to sample (None for all)
        
        Returns:
            Round connectivity analysis
        """
        global_model_data = None
        for gm in self.global_models:
            if gm['round_idx'] == round_idx:
                global_model_data = gm['state_dict']
                break
        
        if global_model_data is None:
            return {}
        
        client_results = []
        
        for client_id, snapshots in self.client_snapshots.items():
            if sample_clients is not None and client_id not in sample_clients:
                continue
            
            snapshot = None
            for s in snapshots:
                if s.round_idx == round_idx:
                    snapshot = s
                    break
            
            if snapshot is None:
                continue
            
            result = self.evaluate_client_connectivity(snapshot, global_model_data, dataloader)
            result['client_id'] = client_id
            client_results.append(result)
        
        round_analysis = self.metrics.analyze_round_connectivity(client_results)
        round_analysis['round_idx'] = round_idx
        self.round_analyses.append(round_analysis)
        
        return round_analysis
    
    def evaluate_cross_round_connectivity(self, round1: int, round2: int,
                                         dataloader) -> Dict[str, Any]:
        """
        Evaluate connectivity between global models from two rounds.
        
        Args:
            round1: First round index
            round2: Second round index
            dataloader: Validation dataloader
        
        Returns:
            Cross-round connectivity analysis
        """
        model1_data = None
        model2_data = None
        
        for gm in self.global_models:
            if gm['round_idx'] == round1:
                model1_data = gm['state_dict']
            if gm['round_idx'] == round2:
                model2_data = gm['state_dict']
        
        if model1_data is None or model2_data is None:
            return {}
        
        model1 = self.model_class()
        model1.load_state_dict(model1_data)
        
        model2 = self.model_class()
        model2.load_state_dict(model2_data)
        
        return self.analyzer.analyze_landscape(
            model1, model2, dataloader, self.loss_fn, self.device
        )
    
    def get_temporal_analysis(self) -> Dict[str, Any]:
        """Get temporal analysis of connectivity evolution."""
        return self.metrics.analyze_temporal_connectivity(self.round_analyses)
    
    def get_disconnected_clients(self, threshold: float = 0.5) -> List[int]:
        """
        Get list of clients that are consistently disconnected.
        
        Args:
            threshold: Connectivity score threshold
        
        Returns:
            List of client IDs with consistently low connectivity
        """
        disconnected = []
        
        for client_id, snapshots in self.client_snapshots.items():
            scores = []
            for snapshot in snapshots:
                for analysis in self.round_analyses:
                    if analysis.get('round_idx') == snapshot.round_idx:
                        for result in analysis.get('client_results', []):
                            if result.get('client_id') == client_id:
                                scores.append(result.get('connectivity_score', 0))
                                break
            
            if scores and np.mean(scores) < threshold:
                disconnected.append(client_id)
        
        return disconnected
    
    def recommend_adjustments(self, round_idx: int, threshold: float = 0.5,
                              max_adjustments: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend adjustments based on connectivity analysis.
        
        Args:
            round_idx: Current round index
            threshold: Connectivity threshold
            max_adjustments: Maximum number of recommendations
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        disconnected = self.get_disconnected_clients(threshold)
        
        for client_id in disconnected[:max_adjustments]:
            recommendations.append({
                'client_id': client_id,
                'action': 'reduce_weight',
                'reason': f'Client {client_id} has consistently low connectivity',
                'suggestion': f'Consider reducing aggregation weight for client {client_id} or triggering personalized training'
            })
        
        return recommendations


class LMCCoordinator:
    """Coordinator for managing federated LMC evaluation."""
    
    def __init__(self, model_class: nn.Module, loss_fn: nn.Module,
                 evaluation_rounds: Optional[List[int]] = None,
                 sample_ratio: float = 0.25, device: str = 'cpu'):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.evaluation_rounds = evaluation_rounds or [20, 50, 80]
        self.sample_ratio = sample_ratio
        self.device = device
        
        self.federated_lmc = FederatedLMC(model_class, loss_fn, device=device)
        
        self.results = {
            'round_results': [],
            'temporal_analysis': None,
            'cross_round_results': []
        }
    
    def should_evaluate(self, round_idx: int) -> bool:
        """Check if we should evaluate connectivity at this round."""
        return round_idx in self.evaluation_rounds
    
    def sample_clients(self, total_clients: int) -> List[int]:
        """Sample clients for evaluation."""
        num_samples = max(1, int(total_clients * self.sample_ratio))
        return list(np.random.choice(total_clients, num_samples, replace=False))
    
    def collect_client_snapshot(self, client_id: int, round_idx: int,
                                model: nn.Module, sample_count: int,
                                loss: Optional[float] = None):
        """Collect a client model snapshot."""
        state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        snapshot = ClientModelSnapshot(
            client_id=client_id,
            round_idx=round_idx,
            model_state_dict=state_dict,
            sample_count=sample_count,
            loss=loss
        )
        
        self.federated_lmc.add_client_snapshot(snapshot)
    
    def collect_global_model(self, round_idx: int, model: nn.Module):
        """Collect global model snapshot."""
        state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.federated_lmc.add_global_model(round_idx, state_dict)
    
    def evaluate_current_round(self, round_idx: int, dataloader) -> Dict[str, Any]:
        """Evaluate connectivity for current round."""
        if not self.should_evaluate(round_idx):
            return {}
        
        analysis = self.federated_lmc.evaluate_round_connectivity(round_idx, dataloader)
        self.results['round_results'].append(analysis)
        
        return analysis
    
    def evaluate_all_rounds(self, dataloader):
        """Evaluate connectivity for all scheduled rounds."""
        for round_idx in self.evaluation_rounds:
            analysis = self.federated_lmc.evaluate_round_connectivity(round_idx, dataloader)
            if analysis:
                self.results['round_results'].append(analysis)
    
    def evaluate_cross_round(self, round1: int, round2: int, dataloader):
        """Evaluate cross-round connectivity."""
        analysis = self.federated_lmc.evaluate_cross_round_connectivity(round1, round2, dataloader)
        self.results['cross_round_results'].append({
            'round1': round1,
            'round2': round2,
            'analysis': analysis
        })
        return analysis
    
    def run_complete_analysis(self, dataloader) -> Dict[str, Any]:
        """Run complete LMC analysis."""
        self.evaluate_all_rounds(dataloader)
        
        for i in range(len(self.evaluation_rounds) - 1):
            round1 = self.evaluation_rounds[i]
            round2 = self.evaluation_rounds[i + 1]
            self.evaluate_cross_round(round1, round2, dataloader)
        
        self.results['temporal_analysis'] = self.federated_lmc.get_temporal_analysis()
        
        return self.results
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations based on analysis."""
        if self.evaluation_rounds:
            last_round = self.evaluation_rounds[-1]
            return self.federated_lmc.recommend_adjustments(last_round)
        return []