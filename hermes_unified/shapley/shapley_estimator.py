"""
Shapley Value Estimator for Federated Learning

Implements Monte Carlo Shapley estimation with privacy protection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sklearn.cluster import KMeans


class ShapleyValue:
    """Represents a Shapley value for a client."""
    
    def __init__(self, client_id: int, value: float, 
                 variance: float = 0.0, confidence: float = 1.0):
        self.client_id = client_id
        self.value = value
        self.variance = variance
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'client_id': self.client_id,
            'value': self.value,
            'variance': self.variance,
            'confidence': self.confidence
        }
    
    def __repr__(self):
        return f"ShapleyValue(client={self.client_id}, value={self.value:.4f})"


class ShapleyEstimator:
    """Base class for Shapley value estimators."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.shapley_values: Dict[int, ShapleyValue] = {}
    
    def estimate(self, client_ids: List[int], 
                 valuation_fn) -> Dict[int, ShapleyValue]:
        """
        Estimate Shapley values.
        
        Args:
            client_ids: List of client IDs
            valuation_fn: Function that takes a subset and returns its value
        
        Returns:
            Dictionary of Shapley values
        """
        raise NotImplementedError
    
    def get_shapley_value(self, client_id: int) -> Optional[ShapleyValue]:
        """Get Shapley value for a specific client."""
        return self.shapley_values.get(client_id)
    
    def get_all_values(self) -> List[ShapleyValue]:
        """Get all estimated Shapley values."""
        return list(self.shapley_values.values())
    
    def normalize(self):
        """Normalize Shapley values to sum to 1."""
        total = sum(sv.value for sv in self.shapley_values.values())
        
        if total > 0:
            for client_id in self.shapley_values:
                self.shapley_values[client_id].value /= total
    
    def truncate_negative(self):
        """Set negative Shapley values to zero."""
        for client_id in self.shapley_values:
            if self.shapley_values[client_id].value < 0:
                self.shapley_values[client_id].value = 0.0


class MonteCarloShapleyEstimator(ShapleyEstimator):
    """
    Monte Carlo Shapley estimator.
    
    Estimates Shapley values by sampling random subsets and computing
    marginal contributions.
    """
    
    def __init__(self, num_clients: int, num_samples: int = 1000):
        super().__init__(num_clients)
        self.num_samples = num_samples
    
    def estimate(self, client_ids: List[int], 
                 valuation_fn) -> Dict[int, ShapleyValue]:
        """
        Estimate Shapley values using Monte Carlo sampling.
        
        Args:
            client_ids: List of client IDs
            valuation_fn: Function that takes a subset and returns its value
        
        Returns:
            Dictionary of Shapley values
        """
        contributions: Dict[int, List[float]] = defaultdict(list)
        
        for _ in range(self.num_samples):
            subset = self._random_subset(client_ids)
            
            for client_id in client_ids:
                if client_id in subset:
                    subset_without = subset - {client_id}
                    v_with = valuation_fn(subset)
                    v_without = valuation_fn(subset_without)
                    marginal = v_with - v_without
                    contributions[client_id].append(marginal)
        
        for client_id in client_ids:
            if contributions[client_id]:
                values = np.array(contributions[client_id])
                mean_val = float(np.mean(values))
                variance = float(np.var(values))
                confidence = 1.0 - variance / (mean_val ** 2 + 1e-6)
                
                self.shapley_values[client_id] = ShapleyValue(
                    client_id=client_id,
                    value=mean_val,
                    variance=variance,
                    confidence=confidence
                )
        
        return self.shapley_values
    
    def _random_subset(self, client_ids: List[int]) -> set:
        """Generate a random subset of clients."""
        subset = set()
        for client_id in client_ids:
            if np.random.random() < 0.5:
                subset.add(client_id)
        return subset
    
    def estimate_with_replacement(self, client_ids: List[int],
                                 valuation_fn) -> Dict[int, ShapleyValue]:
        """
        Estimate using permutation-based Monte Carlo.
        
        Args:
            client_ids: List of client IDs
            valuation_fn: Function that takes a subset and returns its value
        
        Returns:
            Dictionary of Shapley values
        """
        contributions: Dict[int, List[float]] = defaultdict(list)
        
        for _ in range(self.num_samples):
            permutation = np.random.permutation(client_ids)
            prev_value = valuation_fn(set())
            
            for i, client_id in enumerate(permutation):
                subset = set(permutation[:i+1])
                current_value = valuation_fn(subset)
                marginal = current_value - prev_value
                contributions[client_id].append(marginal)
                prev_value = current_value
        
        for client_id in client_ids:
            if contributions[client_id]:
                values = np.array(contributions[client_id])
                self.shapley_values[client_id] = ShapleyValue(
                    client_id=client_id,
                    value=float(np.mean(values)),
                    variance=float(np.var(values))
                )
        
        return self.shapley_values


class GroupedShapleyEstimator(ShapleyEstimator):
    """
    Grouped Shapley estimator.
    
    Clusters clients with similar data distributions and computes
    Shapley values for group representatives.
    """
    
    def __init__(self, num_clients: int, num_groups: int = 5,
                 num_samples: int = 1000):
        super().__init__(num_clients)
        self.num_groups = num_groups
        self.num_samples = num_samples
        self.group_assignments: Dict[int, int] = {}
    
    def cluster_clients(self, client_features: Dict[int, np.ndarray]):
        """
        Cluster clients based on feature distributions.
        
        Args:
            client_features: Dictionary of client features
        """
        features = []
        client_ids = []
        
        for client_id, feature in client_features.items():
            features.append(feature.flatten())
            client_ids.append(client_id)
        
        if len(features) >= self.num_groups:
            kmeans = KMeans(n_clusters=self.num_groups, random_state=42)
            labels = kmeans.fit_predict(features)
            
            for client_id, label in zip(client_ids, labels):
                self.group_assignments[client_id] = label
        else:
            for i, client_id in enumerate(client_ids):
                self.group_assignments[client_id] = i
    
    def estimate(self, client_ids: List[int], 
                 valuation_fn) -> Dict[int, ShapleyValue]:
        """
        Estimate Shapley values using grouped approximation.
        
        Args:
            client_ids: List of client IDs
            valuation_fn: Function that takes a subset and returns its value
        
        Returns:
            Dictionary of Shapley values
        """
        groups = defaultdict(list)
        for client_id in client_ids:
            group_id = self.group_assignments.get(client_id, 0)
            groups[group_id].append(client_id)
        
        group_representatives = {g: clients[0] for g, clients in groups.items()}
        
        mc_estimator = MonteCarloShapleyEstimator(
            num_clients=len(group_representatives),
            num_samples=self.num_samples
        )
        
        def group_valuation_fn(group_subset):
            client_subset = set()
            for group_id in group_subset:
                client_subset.update(groups[group_id])
            return valuation_fn(client_subset)
        
        group_values = mc_estimator.estimate(
            list(group_representatives.keys()),
            group_valuation_fn
        )
        
        for group_id, clients in groups.items():
            group_shapley = group_values.get(group_id)
            
            if group_shapley:
                per_client_value = group_shapley.value / len(clients)
                
                for client_id in clients:
                    self.shapley_values[client_id] = ShapleyValue(
                        client_id=client_id,
                        value=per_client_value,
                        variance=group_shapley.variance
                    )
        
        return self.shapley_values


class FastShapleyEstimator(ShapleyEstimator):
    """
    Fast Shapley estimator using linear approximation.
    
    Uses the fact that in linear models, Shapley values have a closed-form
    solution based on the Hessian of the loss function.
    """
    
    def __init__(self, num_clients: int):
        super().__init__(num_clients)
    
    def estimate_from_gradients(self, client_gradients: Dict[int, Dict[str, torch.Tensor]],
                               hessian: Dict[str, torch.Tensor]) -> Dict[int, ShapleyValue]:
        """
        Estimate Shapley values from gradients and Hessian.
        
        Args:
            client_gradients: Dictionary of client gradients
            hessian: Hessian matrix of the loss function
        
        Returns:
            Dictionary of Shapley values
        """
        total_norm = 0.0
        grad_norms: Dict[int, float] = {}
        
        for client_id, grads in client_gradients.items():
            norm = 0.0
            for name, grad in grads.items():
                if name in hessian:
                    norm += float((grad @ hessian[name] @ grad.T).item())
            grad_norms[client_id] = np.sqrt(norm)
            total_norm += np.sqrt(norm)
        
        for client_id, norm in grad_norms.items():
            if total_norm > 0:
                value = norm / total_norm
            else:
                value = 1.0 / len(client_gradients)
            
            self.shapley_values[client_id] = ShapleyValue(
                client_id=client_id,
                value=value,
                variance=0.0
            )
        
        return self.shapley_values


class PrivacyPreservingShapleyEstimator(MonteCarloShapleyEstimator):
    """
    Privacy-preserving Shapley estimator.
    
    Adds differential privacy noise to the estimated values.
    """
    
    def __init__(self, num_clients: int, num_samples: int = 1000,
                 epsilon: float = 1.0, delta: float = 1e-5):
        super().__init__(num_clients, num_samples)
        self.epsilon = epsilon
        self.delta = delta
    
    def estimate(self, client_ids: List[int],
                 valuation_fn) -> Dict[int, ShapleyValue]:
        """
        Estimate Shapley values with privacy protection.
        
        Args:
            client_ids: List of client IDs
            valuation_fn: Function that takes a subset and returns its value
        
        Returns:
            Dictionary of Shapley values
        """
        super().estimate(client_ids, valuation_fn)
        
        sensitivity = 2.0 / self.num_samples
        
        for client_id in self.shapley_values:
            noise = np.random.laplace(0, sensitivity / self.epsilon)
            self.shapley_values[client_id].value += noise
        
        return self.shapley_values