"""
Federated Shapley Learning Main Module

Implements the complete federated learning framework with Shapley value-based aggregation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .shapley_estimator import (
    MonteCarloShapleyEstimator,
    GroupedShapleyEstimator,
    PrivacyPreservingShapleyEstimator
)

from .incentive_mechanism import (
    TokenDistributor,
    ReputationSystem,
    IncentiveMechanism,
    ContributionTracker
)


class FederatedShapleyLearning:
    """Federated learning framework with Shapley value-based aggregation."""
    
    def __init__(self, model_class: nn.Module, loss_fn: nn.Module,
                 num_clients: int = 10, shapley_interval: int = 10,
                 use_grouped: bool = False, use_privacy: bool = False):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.num_clients = num_clients
        self.shapley_interval = shapley_interval
        self.use_grouped = use_grouped
        self.use_privacy = use_privacy
        
        self.global_model = model_class()
        self.clients: Dict[int, nn.Module] = {}
        self._initialize_clients()
        
        if use_grouped:
            self.shapley_estimator = GroupedShapleyEstimator(num_clients, num_groups=5)
        elif use_privacy:
            self.shapley_estimator = PrivacyPreservingShapleyEstimator(num_clients)
        else:
            self.shapley_estimator = MonteCarloShapleyEstimator(num_clients)
        
        self.token_distributor = TokenDistributor()
        self.reputation_system = ReputationSystem()
        self.incentive_mechanism = IncentiveMechanism(
            self.token_distributor,
            self.reputation_system
        )
        self.contribution_tracker = ContributionTracker()
        
        self.current_weights: Dict[int, float] = {}
        self.round_idx = 0
        
        self.results = {
            'shapley_values': [],
            'accuracies': [],
            'token_distributions': [],
            'reputations': []
        }
    
    def _initialize_clients(self):
        """Initialize clients with local models."""
        for client_id in range(self.num_clients):
            model = self.model_class()
            model.load_state_dict(self.global_model.state_dict())
            self.clients[client_id] = model
    
    def local_train(self, client_id: int, dataloader, epochs: int = 1):
        """
        Train a client locally.
        
        Args:
            client_id: Client ID
            dataloader: Training dataloader
            epochs: Number of epochs
        """
        model = self.clients[client_id]
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for _ in range(epochs):
            for x, y in dataloader:
                optimizer.zero_grad()
                output = model(x)
                loss = self.loss_fn(output, y)
                loss.backward()
                optimizer.step()
    
    def aggregate(self, client_models: Dict[int, nn.Module],
                  weights: Optional[Dict[int, float]] = None) -> nn.Module:
        """
        Aggregate client models.
        
        Args:
            client_models: Dictionary of client models
            weights: Optional weights for aggregation
        
        Returns:
            Aggregated global model
        """
        if weights is None:
            weights = {cid: 1.0 / len(client_models) for cid in client_models}
        
        total_weight = sum(weights.values())
        
        global_state = None
        
        for client_id, model in client_models.items():
            weight = weights.get(client_id, 1.0)
            
            if global_state is None:
                global_state = {k: v * weight for k, v in model.state_dict().items()}
            else:
                for k in global_state:
                    if k in model.state_dict():
                        global_state[k] += model.state_dict()[k] * weight
        
        if global_state and total_weight > 0:
            for k in global_state:
                global_state[k] /= total_weight
        
        self.global_model.load_state_dict(global_state)
        
        for client_id in self.clients:
            self.clients[client_id].load_state_dict(global_state)
        
        return self.global_model
    
    def estimate_shapley_values(self, val_dataloader) -> Dict[int, float]:
        """
        Estimate Shapley values using Monte Carlo sampling.
        
        Args:
            val_dataloader: Validation dataloader for valuation
        
        Returns:
            Dictionary of Shapley values
        """
        def valuation_fn(subset):
            if not subset:
                return 0.0
            
            models = {cid: self.clients[cid] for cid in subset}
            self.aggregate(models)
            
            return self.evaluate(val_dataloader)
        
        self.shapley_estimator.estimate(list(self.clients.keys()), valuation_fn)
        
        self.shapley_estimator.normalize()
        self.shapley_estimator.truncate_negative()
        
        shapley_values = {
            cid: sv.value for cid, sv in self.shapley_estimator.shapley_values.items()
        }
        
        return shapley_values
    
    def evaluate(self, dataloader) -> float:
        """
        Evaluate global model.
        
        Args:
            dataloader: Evaluation dataloader
        
        Returns:
            Accuracy
        """
        self.global_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                output = self.global_model(x)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                total += x.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def run_round(self, client_dataloaders: Dict[int, Any],
                 val_dataloader = None):
        """
        Run one round of federated learning.
        
        Args:
            client_dataloaders: Dictionary of client dataloaders
            val_dataloader: Validation dataloader
        """
        self.round_idx += 1
        
        selected_clients = list(client_dataloaders.keys())
        
        for client_id in selected_clients:
            self.local_train(client_id, client_dataloaders[client_id])
        
        if self.round_idx % self.shapley_interval == 0 and val_dataloader:
            self.current_weights = self.estimate_shapley_values(val_dataloader)
            
            self.incentive_mechanism.reward_all(self.current_weights)
            
            for client_id, value in self.current_weights.items():
                self.contribution_tracker.record_contribution(
                    client_id, self.round_idx, value
                )
            
            self.results['shapley_values'].append(self.current_weights)
        
        if self.current_weights:
            models = {cid: self.clients[cid] for cid in selected_clients}
            weights = {cid: self.current_weights.get(cid, 1.0 / len(models)) 
                      for cid in models}
            self.aggregate(models, weights)
        else:
            models = {cid: self.clients[cid] for cid in selected_clients}
            self.aggregate(models)
        
        if val_dataloader:
            accuracy = self.evaluate(val_dataloader)
            self.results['accuracies'].append(accuracy)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'round': self.round_idx,
            'current_weights': self.current_weights,
            'incentive_stats': self.incentive_mechanism.get_global_stats(),
            'top_clients': self.reputation_system.get_top_clients(5)
        }


def run_shapley_demo():
    """Run federated Shapley learning demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED SHAPLEY VALUE LEARNING")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.shapley.shapley_estimator import (
                MonteCarloShapleyEstimator, ShapleyValue
            )
            print("   ✓ Shapley estimators imported")
        except Exception as e:
            print(f"   ✗ Shapley estimators import failed: {e}")
            return
        
        try:
            from hermes_unified.shapley.incentive_mechanism import (
                TokenDistributor, ReputationSystem
            )
            print("   ✓ Incentive mechanisms imported")
        except Exception as e:
            print(f"   ✗ Incentive mechanisms import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing MonteCarloShapleyEstimator...")
            estimator = MonteCarloShapleyEstimator(num_clients=5, num_samples=100)
            
            def valuation_fn(subset):
                return len(subset) * 0.2
            
            values = estimator.estimate([0, 1, 2, 3, 4], valuation_fn)
            print(f"   ✓ Estimated {len(values)} Shapley values")
        except Exception as e:
            print(f"   ✗ MonteCarloShapleyEstimator failed: {e}")
        
        try:
            print("   Testing TokenDistributor...")
            distributor = TokenDistributor(total_tokens_per_round=100)
            distributor.distribute({0: 0.5, 1: 0.3, 2: 0.2})
            print(f"   ✓ Tokens distributed: {distributor.get_total_distributed()}")
        except Exception as e:
            print(f"   ✗ TokenDistributor failed: {e}")
        
        try:
            print("   Testing ReputationSystem...")
            reputation = ReputationSystem()
            reputation.update_reputation(0, 0.8)
            print(f"   ✓ Reputation updated: {reputation.get_reputation(0):.4f}")
        except Exception as e:
            print(f"   ✗ ReputationSystem failed: {e}")
        
        try:
            print("   Testing FederatedShapleyLearning...")
            fl = FederatedShapleyLearning(
                model_class=torch.nn.Linear,
                loss_fn=torch.nn.CrossEntropyLoss(),
                num_clients=3
            )
            print(f"   ✓ FederatedShapleyLearning created with {len(fl.clients)} clients")
        except Exception as e:
            print(f"   ✗ FederatedShapleyLearning failed: {e}")
        
        print("\n✓ Federated Shapley Value Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_shapley_demo()