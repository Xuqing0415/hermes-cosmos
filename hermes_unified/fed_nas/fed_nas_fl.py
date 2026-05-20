"""
Federated Neural Architecture Search Main Module

Implements the complete federated NAS framework with supernet training and local search.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .supernet import SuperNet, DartsSearchSpace
from .subnet_extractor import SubnetExtractor, ResourceEvaluator, ConstraintChecker
from .local_search import SearchFactory


class FederatedNAS:
    """
    Federated Neural Architecture Search framework.
    
    Three phases:
    1. SuperNet pre-training using federated learning
    2. Local architecture search on each client
    3. Personalized subnet fine-tuning
    """
    
    def __init__(self, search_space=None, num_clients: int = 10,
                 num_classes: int = 10, num_cells: int = 8,
                 num_nodes: int = 4, channels: int = 36):
        if search_space is None:
            search_space = DartsSearchSpace()
        
        self.search_space = search_space
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.channels = channels
        
        self.global_supernet = SuperNet(search_space, num_classes, num_cells, num_nodes, channels)
        
        self.client_supernets: Dict[int, SuperNet] = {}
        self.client_constraints: Dict[int, Dict[str, float]] = {}
        
        self.resource_evaluator = ResourceEvaluator()
        self.subnet_extractor = SubnetExtractor(self.global_supernet, self.resource_evaluator)
        
        self.client_subnets: Dict[int, nn.Module] = {}
        self.client_arch_weights: Dict[int, List[torch.Tensor]] = {}
        
        self.results = {
            'supernet_accuracy': [],
            'client_subnet_accuracies': [],
            'client_resources': [],
            'search_times': []
        }
    
    def initialize_clients(self, constraints: Optional[Dict[int, Dict[str, float]]] = None):
        """
        Initialize client supernets.
        
        Args:
            constraints: Optional dictionary of client constraints
        """
        for client_id in range(self.num_clients):
            client_supernet = SuperNet(self.search_space, self.num_classes, 
                                      self.num_cells, self.num_nodes, self.channels)
            client_supernet.load_state_dict(self.global_supernet.state_dict())
            self.client_supernets[client_id] = client_supernet
            
            if constraints is not None and client_id in constraints:
                self.client_constraints[client_id] = constraints[client_id]
            else:
                self.client_constraints[client_id] = {'max_flops': 5e8}
    
    def federated_supernet_training(self, client_dataloaders: Dict[int, Any],
                                   epochs: int = 10, rounds: int = 50):
        """
        Train supernet using federated learning.
        
        Args:
            client_dataloaders: Dictionary of client dataloaders
            epochs: Local epochs per round
            rounds: Number of federated rounds
        """
        print("Starting federated supernet training...")
        
        for round_idx in range(rounds):
            client_weights = []
            
            for client_id in client_dataloaders:
                client_supernet = self.client_supernets[client_id]
                self._local_train(client_supernet, client_dataloaders[client_id], epochs)
                
                client_weights.append({
                    'client_id': client_id,
                    'weights': client_supernet.state_dict(),
                    'weight': 1.0
                })
            
            self._aggregate_weights(client_weights)
            
            for client_id in self.client_supernets:
                self.client_supernets[client_id].load_state_dict(self.global_supernet.state_dict())
            
            if (round_idx + 1) % 10 == 0:
                print(f"Round {round_idx+1}: SuperNet training in progress...")
    
    def _local_train(self, model: nn.Module, dataloader, epochs: int):
        """Local training on a client."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            for x, y in dataloader:
                optimizer.zero_grad()
                output = model(x, discrete=False)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
    
    def _aggregate_weights(self, client_weights: List[Dict[str, Any]]):
        """Aggregate client weights."""
        total_weight = sum(w['weight'] for w in client_weights)
        
        global_state = None
        for cw in client_weights:
            weight = cw['weight']
            
            if global_state is None:
                global_state = {k: v * weight for k, v in cw['weights'].items()}
            else:
                for k in global_state:
                    if k in cw['weights']:
                        global_state[k] += cw['weights'][k] * weight
        
        if global_state and total_weight > 0:
            for k in global_state:
                global_state[k] /= total_weight
        
        self.global_supernet.load_state_dict(global_state)
    
    def local_architecture_search(self, client_dataloaders: Dict[int, Any],
                                 search_method: str = 'evolutionary',
                                 **search_kwargs):
        """
        Perform local architecture search on each client.
        
        Args:
            client_dataloaders: Dictionary of client validation dataloaders
            search_method: Search method ('evolutionary', 'bayesian', 'differentiable', 'random')
            search_kwargs: Additional search parameters
        """
        print("Starting local architecture search...")
        
        for client_id, dataloader in client_dataloaders.items():
            print(f"Client {client_id} searching...")
            
            client_supernet = self.client_supernets[client_id]
            constraints = self.client_constraints.get(client_id, {})
            
            searcher = SearchFactory.create_search(
                search_method, client_supernet, dataloader,
                nn.CrossEntropyLoss(), **search_kwargs
            )
            
            arch_weights = searcher.search(constraints)
            
            self.client_arch_weights[client_id] = arch_weights
            
            subnet = self.subnet_extractor.extract_subnet(arch_weights)
            self.client_subnets[client_id] = subnet
            
            resources = self.resource_evaluator.evaluate(subnet)
            print(f"Client {client_id} - Accuracy: {searcher.best_acc:.4f}, FLOPs: {resources['flops']:.2e}")
    
    def personalized_finetuning(self, client_dataloaders: Dict[int, Any], epochs: int = 5):
        """
        Fine-tune client subnets locally.
        
        Args:
            client_dataloaders: Dictionary of client dataloaders
            epochs: Number of fine-tuning epochs
        """
        print("Starting personalized fine-tuning...")
        
        for client_id, dataloader in client_dataloaders.items():
            subnet = self.client_subnets.get(client_id)
            
            if subnet is None:
                continue
            
            subnet.train()
            optimizer = torch.optim.Adam(subnet.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            
            for _ in range(epochs):
                for x, y in dataloader:
                    optimizer.zero_grad()
                    output = subnet(x)
                    loss = loss_fn(output, y)
                    loss.backward()
                    optimizer.step()
    
    def evaluate_clients(self, client_dataloaders: Dict[int, Any]) -> Dict[int, float]:
        """
        Evaluate all client subnets.
        
        Args:
            client_dataloaders: Dictionary of client test dataloaders
        
        Returns:
            Dictionary of client accuracies
        """
        accuracies = {}
        
        for client_id, dataloader in client_dataloaders.items():
            subnet = self.client_subnets.get(client_id)
            
            if subnet is None:
                accuracies[client_id] = 0.0
                continue
            
            subnet.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for x, y in dataloader:
                    output = subnet(x)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(y.view_as(pred)).sum().item()
                    total += x.size(0)
            
            accuracies[client_id] = correct / total if total > 0 else 0.0
        
        return accuracies
    
    def run(self, train_dataloaders: Dict[int, Any],
            val_dataloaders: Dict[int, Any],
            test_dataloaders: Dict[int, Any],
            supernet_epochs: int = 5,
            supernet_rounds: int = 30,
            search_method: str = 'evolutionary',
            finetune_epochs: int = 3):
        """
        Run the complete FedNAS pipeline.
        
        Args:
            train_dataloaders: Training dataloaders
            val_dataloaders: Validation dataloaders for search
            test_dataloaders: Test dataloaders
            supernet_epochs: Local epochs for supernet training
            supernet_rounds: Federated rounds for supernet training
            search_method: Architecture search method
            finetune_epochs: Fine-tuning epochs
        
        Returns:
            Results dictionary
        """
        self.initialize_clients()
        
        self.federated_supernet_training(train_dataloaders, supernet_epochs, supernet_rounds)
        
        self.local_architecture_search(val_dataloaders, search_method)
        
        self.personalized_finetuning(train_dataloaders, finetune_epochs)
        
        accuracies = self.evaluate_clients(test_dataloaders)
        
        return {
            'client_accuracies': accuracies,
            'avg_accuracy': float(np.mean(list(accuracies.values()))),
            'client_subnets': self.client_subnets
        }


def run_fednas_demo():
    """Run federated NAS demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED NEURAL ARCHITECTURE SEARCH")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.fed_nas.supernet import SuperNet, DartsSearchSpace
            print("   ✓ SuperNet and SearchSpace imported")
        except Exception as e:
            print(f"   ✗ SuperNet import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_nas.subnet_extractor import (
                SubnetExtractor, ResourceEvaluator
            )
            print("   ✓ SubnetExtractor imported")
        except Exception as e:
            print(f"   ✗ SubnetExtractor import failed: {e}")
            return
        
        try:
            from hermes_unified.fed_nas.local_search import (
                EvolutionarySearch, SearchFactory
            )
            print("   ✓ Local search imported")
        except Exception as e:
            print(f"   ✗ Local search import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing SuperNet...")
            search_space = DartsSearchSpace()
            supernet = SuperNet(search_space, num_classes=10)
            dummy_input = torch.randn(2, 3, 32, 32)
            output = supernet(dummy_input)
            print(f"   ✓ SuperNet created, output shape: {output.shape}")
        except Exception as e:
            print(f"   ✗ SuperNet failed: {e}")
        
        try:
            print("   Testing ResourceEvaluator...")
            evaluator = ResourceEvaluator()
            resources = evaluator.evaluate(supernet, (3, 32, 32))
            print(f"   ✓ Resource evaluation: FLOPs={resources['flops']:.2e}")
        except Exception as e:
            print(f"   ✗ ResourceEvaluator failed: {e}")
        
        try:
            print("   Testing FederatedNAS...")
            fednas = FederatedNAS(search_space, num_clients=3)
            print(f"   ✓ FederatedNAS created with {fednas.num_clients} clients")
        except Exception as e:
            print(f"   ✗ FederatedNAS failed: {e}")
        
        print("\n✓ Federated Neural Architecture Search demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_fednas_demo()