"""
Topology-Aware Federated Learning Main Module

Implements the complete topology-aware federated learning framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .topology_discovery import NetworkTopology, TopologyDiscovery
from .hierarchical_aggregator import HierarchicalAggregator, HybridAggregator
from .topology_scheduler import TopologyScheduler, TopologyAdaptiveController


class TopologyAwareFL:
    """Topology-aware federated learning framework."""
    
    def __init__(self, model_class: nn.Module, loss_fn: nn.Module,
                 num_clients: int = 20, num_rounds: int = 100,
                 topology_type: str = 'erdos_renyi'):
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.topology_type = topology_type
        
        self.topology = NetworkTopology(topology_type)
        self.discovery = TopologyDiscovery()
        
        self.scheduler = None
        self.aggregator = None
        self.controller = None
        
        self.clients: Dict[int, nn.Module] = {}
        self.global_model: Optional[nn.Module] = None
        
        self.results = {
            'round_times': [],
            'communication_costs': [],
            'accuracies': [],
            'selected_clients': [],
            'topology_changes': []
        }
    
    def setup(self):
        """Set up the topology-aware FL framework."""
        client_ids = list(range(self.num_clients))
        server_id = -1
        
        self.topology.generate_random_topology(self.num_clients + 1)
        
        self.scheduler = TopologyScheduler(self.topology, server_id)
        self.scheduler.set_strategy('min_cost')
        
        self.aggregator = HybridAggregator(self.topology, server_id)
        
        self.controller = TopologyAdaptiveController(self.scheduler, self.aggregator)
        
        self.global_model = self.model_class()
        
        for client_id in client_ids:
            self.clients[client_id] = self.model_class()
            self.clients[client_id].load_state_dict(self.global_model.state_dict())
    
    def train_round(self, client_dataloaders: Dict[int, Any],
                   server_id: int = -1) -> Dict[str, Any]:
        """
        Perform one round of topology-aware federated training.
        
        Args:
            client_dataloaders: Dictionary of dataloaders {client_id: dataloader}
            server_id: Server ID
        
        Returns:
            Round statistics
        """
        start_time = np.datetime64('now')
        
        selected = self.scheduler.select_participants(
            list(client_dataloaders.keys()),
            num_clients=min(10, len(client_dataloaders)),
            round_idx=len(self.results['round_times'])
        )
        
        selected = self.scheduler.adjust_for_fairness(selected)
        
        client_models = {}
        for client_id in selected:
            if client_id in client_dataloaders:
                self._local_train(client_id, client_dataloaders[client_id])
                client_models[client_id] = {
                    k: v.detach().cpu().clone() 
                    for k, v in self.clients[client_id].state_dict().items()
                }
        
        global_state_dict = self.aggregator.aggregate(client_models, apply_dp=False)
        
        if global_state_dict:
            self.global_model.load_state_dict(global_state_dict)
            
            for client_id in self.clients:
                self.clients[client_id].load_state_dict(global_state_dict)
        
        end_time = np.datetime64('now')
        round_time = float((end_time - start_time) / np.timedelta64(1, 's'))
        
        comm_cost = self.scheduler.get_selection_cost(selected)
        
        self.results['round_times'].append(round_time)
        self.results['communication_costs'].append(comm_cost)
        self.results['selected_clients'].append(selected)
        
        self.controller.record_round(comm_cost)
        
        if self.controller.should_adapt():
            self.results['topology_changes'].append(len(self.results['round_times']))
        
        return {
            'round_time': round_time,
            'communication_cost': comm_cost,
            'selected_clients': selected,
            'num_selected': len(selected)
        }
    
    def _local_train(self, client_id: int, dataloader, epochs: int = 1):
        """Perform local training on a client."""
        model = self.clients[client_id]
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for _ in range(epochs):
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
    
    def evaluate(self, dataloader) -> float:
        """Evaluate global model on validation set."""
        if self.global_model is None:
            return 0.0
        
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                output = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        self.results['accuracies'].append(accuracy)
        
        return accuracy
    
    def run(self, client_dataloaders: Dict[int, Any],
           val_dataloader = None) -> Dict[str, Any]:
        """
        Run full topology-aware federated training.
        
        Args:
            client_dataloaders: Dictionary of client dataloaders
            val_dataloader: Validation dataloader
        
        Returns:
            Training results
        """
        print(f"Starting Topology-Aware FL with {self.num_clients} clients")
        
        for round_idx in range(self.num_rounds):
            stats = self.train_round(client_dataloaders)
            
            if val_dataloader is not None and round_idx % 10 == 0:
                accuracy = self.evaluate(val_dataloader)
                print(f"Round {round_idx+1}: Accuracy={accuracy:.4f}, Time={stats['round_time']:.2f}s")
            
            if (round_idx + 1) % 20 == 0:
                self.topology = self.discovery.update_topology()
                self.scheduler.update_topology(self.topology)
                self.aggregator.update_topology(self.topology)
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        return {
            'avg_round_time': float(np.mean(self.results['round_times'])) if self.results['round_times'] else 0,
            'total_communication_cost': float(np.sum(self.results['communication_costs'])),
            'final_accuracy': self.results['accuracies'][-1] if self.results['accuracies'] else 0,
            'topology_info': self.topology.get_topology_info(),
            'scheduler_stats': self.scheduler.get_scheduler_stats(),
            'num_adaptations': len(self.results['topology_changes'])
        }


def run_topology_demo():
    """Run topology-aware federated learning demonstration."""
    print("\n" + "=" * 70)
    print("  TOPOLOGY-AWARE FEDERATED LEARNING")
    print("=" * 70)
    
    try:
        import torch
        
        print("\n1. Checking torch installation...")
        print(f"   ✓ torch version: {torch.__version__}")
        
        print("\n2. Testing Core Components Import:")
        
        try:
            from hermes_unified.topology_aware.topology_discovery import NetworkTopology
            print("   ✓ NetworkTopology imported")
        except Exception as e:
            print(f"   ✗ NetworkTopology import failed: {e}")
            return
        
        try:
            from hermes_unified.topology_aware.hierarchical_aggregator import (
                HierarchicalAggregator, AggregationTree
            )
            print("   ✓ HierarchicalAggregator imported")
        except Exception as e:
            print(f"   ✗ HierarchicalAggregator import failed: {e}")
            return
        
        try:
            from hermes_unified.topology_aware.topology_scheduler import (
                TopologyScheduler, GraphBasedSelector
            )
            print("   ✓ TopologyScheduler imported")
        except Exception as e:
            print(f"   ✗ TopologyScheduler import failed: {e}")
            return
        
        print("\n3. Testing Component Functionality:")
        
        try:
            print("   Testing NetworkTopology...")
            topology = NetworkTopology('small_world')
            topology.generate_random_topology(20)
            info = topology.get_topology_info()
            print(f"   ✓ Topology created: {info['num_nodes']} nodes, {info['num_edges']} edges")
        except Exception as e:
            print(f"   ✗ NetworkTopology failed: {e}")
        
        try:
            print("   Testing AggregationTree...")
            tree = AggregationTree()
            tree.add_node(-1)
            tree.add_node(0)
            tree.add_edge(-1, 0)
            print(f"   ✓ Aggregation tree: {tree.get_num_nodes()} nodes")
        except Exception as e:
            print(f"   ✗ AggregationTree failed: {e}")
        
        try:
            print("   Testing TopologyScheduler...")
            scheduler = TopologyScheduler(topology, server_id=-1)
            scheduler.set_strategy('min_cost')
            selected = scheduler.select_participants(list(range(20)), 5, 0)
            print(f"   ✓ Selected {len(selected)} clients")
        except Exception as e:
            print(f"   ✗ TopologyScheduler failed: {e}")
        
        try:
            print("   Testing TopologyAwareFL...")
            fl = TopologyAwareFL(
                model_class=torch.nn.Linear,
                loss_fn=torch.nn.CrossEntropyLoss(),
                num_clients=5,
                num_rounds=2
            )
            fl.setup()
            print(f"   ✓ TopologyAwareFL setup with {len(fl.clients)} clients")
        except Exception as e:
            print(f"   ✗ TopologyAwareFL failed: {e}")
        
        print("\n✓ Topology-Aware Federated Learning demo completed!")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
    except Exception as e:
        print(f"⚠️ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_topology_demo()