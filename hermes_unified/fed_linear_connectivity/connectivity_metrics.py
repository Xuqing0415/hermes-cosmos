"""
Connectivity Metrics for Federated Linear Mode Connectivity

Implements metrics for evaluating linear mode connectivity between models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class ConnectivityScore:
    """
    Connectivity score for measuring linear mode connectivity.
    
    A score of 1.0 means fully connected (no loss spike),
    0.0 means completely disconnected (loss spike exceeds threshold).
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.scores = []
    
    def compute(self, losses: np.ndarray, base_loss: float) -> float:
        """
        Compute connectivity score from interpolation losses.
        
        Args:
            losses: Loss values along interpolation path
            base_loss: Maximum of endpoint losses
        
        Returns:
            Connectivity score in [0, 1]
        """
        max_loss = np.max(losses)
        loss_spike = max_loss - base_loss
        
        if loss_spike <= 0:
            return 1.0
        
        score = max(0.0, 1.0 - (loss_spike / self.threshold))
        return score
    
    def add_score(self, score: float):
        """Add a connectivity score to history."""
        self.scores.append(score)
    
    def get_average_score(self) -> float:
        """Get average connectivity score."""
        if not self.scores:
            return 0.0
        return float(np.mean(self.scores))


class InterpolationLossEvaluator:
    """Evaluates loss along linear interpolation path between two models."""
    
    def __init__(self, model_class: nn.Module, num_points: int = 10):
        self.model_class = model_class
        self.num_points = num_points
    
    def interpolate_models(self, model1: nn.Module, model2: nn.Module, 
                          alpha: float) -> nn.Module:
        """
        Create an interpolated model.
        
        Args:
            model1: First model (alpha=0)
            model2: Second model (alpha=1)
            alpha: Interpolation weight in [0, 1]
        
        Returns:
            Interpolated model
        """
        interp_model = self.model_class()
        
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        
        interp_state = {}
        for key in state_dict1:
            interp_state[key] = (1 - alpha) * state_dict1[key] + alpha * state_dict2[key]
        
        interp_model.load_state_dict(interp_state)
        return interp_model
    
    def evaluate_interpolation(self, model1: nn.Module, model2: nn.Module,
                              dataloader, loss_fn: nn.Module,
                              device: str = 'cpu') -> np.ndarray:
        """
        Evaluate loss along interpolation path.
        
        Args:
            model1: First model
            model2: Second model
            dataloader: Validation dataloader
            loss_fn: Loss function
            device: Device for evaluation
        
        Returns:
            Array of loss values along interpolation path
        """
        losses = []
        alphas = np.linspace(0.0, 1.0, self.num_points)
        
        for alpha in alphas:
            model = self.interpolate_models(model1, model2, alpha)
            model = model.to(device)
            model.eval()
            
            total_loss = 0.0
            count = 0
            
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = loss_fn(output, target)
                    total_loss += loss.item() * data.size(0)
                    count += data.size(0)
            
            avg_loss = total_loss / count if count > 0 else 0.0
            losses.append(avg_loss)
        
        return np.array(losses)
    
    def evaluate_interpolation_vectorized(self, model1: nn.Module, model2: nn.Module,
                                         dataloader, loss_fn: nn.Module,
                                         device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate both loss and accuracy along interpolation path.
        
        Args:
            model1: First model
            model2: Second model
            dataloader: Validation dataloader
            loss_fn: Loss function
            device: Device for evaluation
        
        Returns:
            Tuple of (losses, accuracies) arrays
        """
        losses = []
        accuracies = []
        alphas = np.linspace(0.0, 1.0, self.num_points)
        
        for alpha in alphas:
            model = self.interpolate_models(model1, model2, alpha)
            model = model.to(device)
            model.eval()
            
            total_loss = 0.0
            correct = 0
            count = 0
            
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = loss_fn(output, target)
                    total_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    count += data.size(0)
            
            avg_loss = total_loss / count if count > 0 else 0.0
            avg_acc = correct / count if count > 0 else 0.0
            
            losses.append(avg_loss)
            accuracies.append(avg_acc)
        
        return np.array(losses), np.array(accuracies)


class LossLandscapeAnalyzer:
    """Analyze loss landscape along interpolation paths."""
    
    def __init__(self, num_points: int = 20):
        self.num_points = num_points
        self.evaluator = InterpolationLossEvaluator(nn.Module, num_points)
    
    def analyze_landscape(self, model1: nn.Module, model2: nn.Module,
                          dataloader, loss_fn: nn.Module,
                          device: str = 'cpu') -> Dict[str, Any]:
        """
        Analyze loss landscape between two models.
        
        Args:
            model1: First model
            model2: Second model
            dataloader: Validation dataloader
            loss_fn: Loss function
            device: Device for evaluation
        
        Returns:
            Dictionary containing landscape analysis
        """
        losses = self.evaluator.evaluate_interpolation(
            model1, model2, dataloader, loss_fn, device
        )
        
        alphas = np.linspace(0.0, 1.0, self.num_points)
        
        base_loss = max(losses[0], losses[-1])
        max_loss = np.max(losses)
        loss_spike = max_loss - base_loss
        
        spike_location = alphas[np.argmax(losses)]
        
        connectivity_score = ConnectivityScore().compute(losses, base_loss)
        
        return {
            'losses': losses,
            'alphas': alphas,
            'base_loss': base_loss,
            'max_loss': max_loss,
            'loss_spike': loss_spike,
            'spike_location': spike_location,
            'connectivity_score': connectivity_score,
            'is_connected': connectivity_score > 0.5
        }
    
    def analyze_multiple_pairs(self, model_pairs: List[Tuple[nn.Module, nn.Module]],
                               dataloader, loss_fn: nn.Module,
                               device: str = 'cpu') -> List[Dict[str, Any]]:
        """
        Analyze multiple model pairs.
        
        Args:
            model_pairs: List of (model1, model2) tuples
            dataloader: Validation dataloader
            loss_fn: Loss function
            device: Device for evaluation
        
        Returns:
            List of landscape analyses
        """
        results = []
        for model1, model2 in model_pairs:
            analysis = self.analyze_landscape(model1, model2, dataloader, loss_fn, device)
            results.append(analysis)
        return results
    
    def compute_basin_overlap(self, analysis1: Dict[str, Any], 
                              analysis2: Dict[str, Any]) -> float:
        """
        Compute overlap between two loss basins.
        
        Args:
            analysis1: First landscape analysis
            analysis2: Second landscape analysis
        
        Returns:
            Overlap score in [0, 1]
        """
        if not analysis1['is_connected'] or not analysis2['is_connected']:
            return 0.0
        
        score1 = analysis1['connectivity_score']
        score2 = analysis2['connectivity_score']
        
        avg_spike = (analysis1['loss_spike'] + analysis2['loss_spike']) / 2
        
        overlap = (score1 + score2) / 2 * np.exp(-avg_spike)
        return min(1.0, float(overlap))


class LinearConnectivityMetrics:
    """Comprehensive metrics for linear mode connectivity."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def add_metric(self, name: str, value: float):
        """Add a metric value."""
        self.metrics[name].append(value)
    
    def compute_summary(self) -> Dict[str, float]:
        """Compute summary statistics for all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[f'{name}_mean'] = float(np.mean(values))
                summary[f'{name}_std'] = float(np.std(values))
                summary[f'{name}_min'] = float(np.min(values))
                summary[f'{name}_max'] = float(np.max(values))
        return summary
    
    def analyze_round_connectivity(self, round_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze connectivity across clients for a single round.
        
        Args:
            round_results: List of connectivity results from each client
        
        Returns:
            Aggregated round statistics
        """
        if not round_results:
            return {}
        
        scores = [r['connectivity_score'] for r in round_results]
        spikes = [r['loss_spike'] for r in round_results]
        
        return {
            'mean_connectivity': float(np.mean(scores)),
            'std_connectivity': float(np.std(scores)),
            'min_connectivity': float(np.min(scores)),
            'max_connectivity': float(np.max(scores)),
            'mean_spike': float(np.mean(spikes)),
            'std_spike': float(np.std(spikes)),
            'connected_clients': sum(1 for r in round_results if r['is_connected']),
            'total_clients': len(round_results),
            'connected_ratio': float(sum(1 for r in round_results if r['is_connected']) / len(round_results))
        }
    
    def analyze_temporal_connectivity(self, round_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze connectivity evolution across rounds.
        
        Args:
            round_analyses: List of round connectivity analyses
        
        Returns:
            Temporal connectivity statistics
        """
        if not round_analyses:
            return {}
        
        connectivity_scores = [r['mean_connectivity'] for r in round_analyses]
        connected_ratios = [r['connected_ratio'] for r in round_analyses]
        mean_spikes = [r['mean_spike'] for r in round_analyses]
        
        return {
            'final_connectivity': connectivity_scores[-1] if connectivity_scores else 0.0,
            'initial_connectivity': connectivity_scores[0] if connectivity_scores else 0.0,
            'connectivity_improvement': float((connectivity_scores[-1] - connectivity_scores[0]) if len(connectivity_scores) > 1 else 0.0),
            'avg_connectivity': float(np.mean(connectivity_scores)),
            'avg_connected_ratio': float(np.mean(connected_ratios)),
            'avg_spike': float(np.mean(mean_spikes)),
            'connectivity_trend': self._compute_trend(connectivity_scores)
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction."""
        if len(values) < 2:
            return 'stable'
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'