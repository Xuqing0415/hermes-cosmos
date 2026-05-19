"""
Federated Causal Representation Learning Coordinator

Integrates all components for end-to-end federated causal representation learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .causal_vae_client import CausalVAEClient, generate_causal_data
from .graph_aggregator import PrivacyPreservingGraphAggregator, FederatedCausalDiscovery
from .invariance_constraint import InvarianceManager
from .counterfactual import CounterfactualReasoner, CausalEffectEstimator


class FedCausalCoordinator:
    """Coordinator for Federated Causal Representation Learning."""
    
    def __init__(self, num_clients: int = 5, input_dim: int = 8,
                 num_causal_vars: int = 5, num_noise_vars: int = 3,
                 num_rounds: int = 10, device: str = 'cpu'):
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.num_causal_vars = num_causal_vars
        self.num_noise_vars = num_noise_vars
        self.num_rounds = num_rounds
        self.device = device
        
        self.clients: List[CausalVAEClient] = []
        self.graph_aggregator = PrivacyPreservingGraphAggregator(num_causal_vars)
        self.invariance_manager = InvarianceManager(num_causal_vars, num_clients)
        
        self.global_params = None
        self.true_adj = None
        
        self.results = {
            'loss_history': [],
            'shd_history': [],
            'invariance_loss_history': []
        }
    
    def setup(self, true_adj: Optional[np.ndarray] = None):
        """
        Set up clients and initialize parameters.
        
        Args:
            true_adj: True adjacency matrix for evaluation
        """
        self.true_adj = true_adj
        
        for i in range(self.num_clients):
            client = CausalVAEClient(
                client_id=i,
                input_dim=self.input_dim,
                num_causal_vars=self.num_causal_vars,
                num_noise_vars=self.num_noise_vars,
                device=self.device
            )
            client.set_environment(i)
            self.clients.append(client)
        
        self.global_params = self.clients[0]._get_shared_params()
    
    def generate_client_data(self, num_samples_per_client: int = 1000) -> List[torch.Tensor]:
        """
        Generate synthetic data for each client.
        
        Args:
            num_samples_per_client: Number of samples per client
        
        Returns:
            List of data tensors
        """
        data_list = []
        for i in range(self.num_clients):
            data, true_adj = generate_causal_data(
                num_samples=num_samples_per_client,
                num_causal_vars=self.num_causal_vars,
                num_noise_vars=self.num_noise_vars,
                env_id=i
            )
            data_list.append(data)
            
            if self.true_adj is None:
                self.true_adj = true_adj
        
        return data_list
    
    def train_round(self, dataloaders: List, apply_invariance: bool = True) -> Dict[str, float]:
        """
        Train one round of federated causal learning.
        
        Args:
            dataloaders: List of dataloaders for each client
            apply_invariance: Whether to apply invariance constraints
        
        Returns:
            Round metrics
        """
        client_params = []
        representations = []
        
        for client, dataloader in zip(self.clients, dataloaders):
            params = client.local_train(
                dataloader,
                self.global_params,
                num_epochs=1
            )
            client_params.append(params)
            
            causal_graph = client.get_causal_graph()
            self.graph_aggregator.receive_graph(client.client_id, causal_graph)
            
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                rep = client.get_causal_representation(x)
                representations.append(rep)
                break
        
        self.global_params = self._aggregate_params(client_params)
        
        if apply_invariance and len(representations) >= 2:
            inv_losses = self.invariance_manager.compute_total_loss(representations)
            self.results['invariance_loss_history'].append(inv_losses['total'].item())
        
        global_graph = self.graph_aggregator.aggregate()
        
        shd = 0
        if self.true_adj is not None:
            shd = self._compute_shd(global_graph, self.true_adj)
            self.results['shd_history'].append(shd)
        
        avg_loss = np.mean([c.loss_history[-1] for c in self.clients])
        self.results['loss_history'].append(avg_loss)
        
        return {
            'avg_loss': avg_loss,
            'shd': shd,
            'graph_density': np.mean(global_graph)
        }
    
    def run(self, dataloaders: List, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run full federated causal learning.
        
        Args:
            dataloaders: List of dataloaders
            num_rounds: Number of rounds (overrides constructor)
        
        Returns:
            Results dictionary
        """
        rounds = num_rounds or self.num_rounds
        
        print(f"Starting Federated Causal Representation Learning with {self.num_clients} clients")
        
        for round_idx in range(rounds):
            metrics = self.train_round(dataloaders)
            
            print(f"Round {round_idx + 1}/{rounds}")
            print(f"  Loss: {metrics['avg_loss']:.4f}")
            print(f"  SHD: {metrics['shd']}")
            print(f"  Graph Density: {metrics['graph_density']:.4f}")
        
        return self.results
    
    def _aggregate_params(self, client_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate parameters from clients."""
        aggregated = {}
        
        for key in ['sem_adjacency', 'encoder', 'decoder']:
            if key not in client_params[0]:
                continue
            
            if key == 'sem_adjacency':
                adjacencies = [p[key] for p in client_params]
                aggregated[key] = torch.mean(torch.stack(adjacencies), dim=0)
            
            else:
                aggregated[key] = {}
                state_dict = client_params[0][key]
                
                for param_name in state_dict:
                    params = [p[key][param_name] for p in client_params]
                    aggregated[key][param_name] = torch.mean(torch.stack(params), dim=0)
        
        return aggregated
    
    def _compute_shd(self, learned: np.ndarray, true: np.ndarray) -> int:
        """Compute Structural Hamming Distance."""
        return int(np.sum(np.abs(learned - true)))
    
    def get_counterfactual_reasoner(self, client_id: int = 0) -> CounterfactualReasoner:
        """
        Get counterfactual reasoner for a client.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Counterfactual reasoner
        """
        client = self.clients[client_id]
        
        reasoner = CounterfactualReasoner(
            encoder=client.model.encoder,
            decoder=client.model.decoder,
            sem=client.model.sem,
            num_causal_vars=self.num_causal_vars
        )
        
        return reasoner
    
    def evaluate_causal_discovery(self) -> Dict[str, float]:
        """Evaluate causal discovery performance."""
        if self.true_adj is None:
            return {'shd': -1, 'precision': 0, 'recall': 0, 'f1': 0}
        
        discovered = self.graph_aggregator.get_global_graph()
        
        if discovered is None:
            return {'shd': -1, 'precision': 0, 'recall': 0, 'f1': 0}
        
        shd = self._compute_shd(discovered, self.true_adj)
        
        tp = np.sum((discovered == 1) & (self.true_adj == 1))
        fp = np.sum((discovered == 1) & (self.true_adj == 0))
        fn = np.sum((discovered == 0) & (self.true_adj == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'shd': shd,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_counterfactual(self, test_data: torch.Tensor,
                                interventions: List[Dict[int, float]],
                                true_outcomes: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate counterfactual prediction.
        
        Args:
            test_data: Test data
            interventions: List of interventions
            true_outcomes: True counterfactual outcomes
        
        Returns:
            Evaluation metrics
        """
        reasoner = self.get_counterfactual_reasoner()
        
        total_mse = 0.0
        total_mae = 0.0
        
        for x, intervention, true_outcome in zip(test_data, interventions, true_outcomes):
            x_batch = x.unsqueeze(0)
            predicted = reasoner.counterfactual(x_batch, intervention)
            
            mse = torch.nn.functional.mse_loss(predicted.squeeze(0), true_outcome).item()
            mae = torch.nn.functional.l1_loss(predicted.squeeze(0), true_outcome).item()
            
            total_mse += mse
            total_mae += mae
        
        n = len(test_data)
        return {
            'mse': total_mse / n if n > 0 else 0,
            'mae': total_mae / n if n > 0 else 0
        }


def run_fed_causal_demo():
    """Run demonstration of federated causal representation learning."""
    print("\n" + "=" * 70)
    print("  FEDERATED CAUSAL REPRESENTATION LEARNING")
    print("=" * 70)
    
    try:
        import torch
        print(f"\n1. Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.fed_causal_representation.causal_vae_client import (
                CausalVAEClient, CausalVAE, StructuralEquationModel
            )
            print("   ✓ CausalVAE components imported")
        except Exception as e:
            print(f"   ✗ CausalVAE import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_causal_representation.graph_aggregator import (
                CausalGraphAggregator, PrivacyPreservingGraphAggregator
            )
            print("   ✓ Graph aggregator imported")
        except Exception as e:
            print(f"   ✗ Graph aggregator import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_causal_representation.invariance_constraint import (
                MMDInvariance, AdversarialInvariance
            )
            print("   ✓ Invariance constraints imported")
        except Exception as e:
            print(f"   ✗ Invariance constraints import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_causal_representation.counterfactual import (
                CounterfactualReasoner, InterventionModel
            )
            print("   ✓ Counterfactual components imported")
        except Exception as e:
            print(f"   ✗ Counterfactual import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing StructuralEquationModel...")
            sem = StructuralEquationModel(num_causal_vars=5)
            exogenous = torch.randn(8, 5)
            causal_vars = sem(exogenous)
            dag_loss = sem.dag_constraint()
            print(f"   ✓ SEM: causal_vars shape {causal_vars.shape}, DAG loss {dag_loss.item():.4f}")
        except Exception as e:
            print(f"   ✗ SEM failed: {e}")
        
        try:
            print("   Testing CausalVAE...")
            vae = CausalVAE(input_dim=8, num_causal_vars=5, num_noise_vars=3)
            x = torch.randn(8, 8)
            outputs = vae(x)
            print(f"   ✓ CausalVAE: reconstruction shape {outputs['reconstruction'].shape}")
        except Exception as e:
            print(f"   ✗ CausalVAE failed: {e}")
        
        try:
            print("   Testing CausalGraphAggregator...")
            aggregator = CausalGraphAggregator(num_vars=5)
            adj1 = np.random.rand(5, 5) > 0.7
            adj2 = np.random.rand(5, 5) > 0.7
            aggregator.receive_graph(0, adj1)
            aggregator.receive_graph(1, adj2)
            global_adj = aggregator.aggregate()
            print(f"   ✓ Graph aggregator: global graph shape {global_adj.shape}")
        except Exception as e:
            print(f"   ✗ Graph aggregator failed: {e}")
        
        try:
            print("   Testing MMDInvariance...")
            mmd = MMDInvariance()
            rep1 = torch.randn(8, 5)
            rep2 = torch.randn(8, 5)
            loss = mmd([rep1, rep2])
            print(f"   ✓ MMD invariance: loss {loss.item():.4f}")
        except Exception as e:
            print(f"   ✗ MMD invariance failed: {e}")
        
        try:
            print("   Testing FedCausalCoordinator...")
            coordinator = FedCausalCoordinator(
                num_clients=3,
                input_dim=8,
                num_causal_vars=5,
                num_noise_vars=3,
                num_rounds=2
            )
            coordinator.setup()
            print(f"   ✓ Coordinator setup with {len(coordinator.clients)} clients")
        except Exception as e:
            print(f"   ✗ Coordinator failed: {e}")
        
        print("\n✓ Federated Causal Representation Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_fed_causal_demo()