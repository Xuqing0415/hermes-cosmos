"""
Hierarchical Aggregator for Topology-Aware Federated Learning

Implements tree-based aggregation and hierarchical federated learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class TreeAggregationNode:
    """Node in the aggregation tree."""
    
    def __init__(self, node_id: int, level: int = 0):
        self.node_id = node_id
        self.level = level
        self.parent: Optional['TreeAggregationNode'] = None
        self.children: List['TreeAggregationNode'] = []
        self.is_server = False
        
        self.local_model: Optional[Dict[str, torch.Tensor]] = None
        self.aggregated_model: Optional[Dict[str, torch.Tensor]] = None
    
    def add_child(self, child: 'TreeAggregationNode'):
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def aggregate_up(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate models from children and pass up to parent.
        
        Returns:
            Aggregated model parameters
        """
        if not self.children:
            return self.local_model
        
        agg_params = None
        total_weight = 0.0
        
        for child in self.children:
            child_agg = child.aggregate_up()
            if child_agg is not None:
                weight = 1.0
                
                if agg_params is None:
                    agg_params = {k: v * weight for k, v in child_agg.items()}
                else:
                    for k in agg_params:
                        agg_params[k] = agg_params[k] + child_agg[k] * weight
                
                total_weight += weight
        
        if self.local_model is not None:
            if agg_params is None:
                agg_params = self.local_model.copy()
            else:
                for k in agg_params:
                    agg_params[k] = agg_params[k] + self.local_model[k]
                total_weight += 1.0
        
        if agg_params is not None and total_weight > 0:
            for k in agg_params:
                agg_params[k] = agg_params[k] / total_weight
        
        self.aggregated_model = agg_params
        return agg_params
    
    def broadcast_down(self, model: Dict[str, torch.Tensor]):
        """Broadcast model to children."""
        self.aggregated_model = model
        
        for child in self.children:
            child.broadcast_down(model.copy())
    
    def get_all_clients(self) -> List[int]:
        """Get all leaf nodes (clients) in subtree."""
        if not self.children:
            return [self.node_id]
        
        clients = []
        for child in self.children:
            clients.extend(child.get_all_clients())
        
        return clients
    
    def __repr__(self):
        return f"TreeAggregationNode(id={self.node_id}, level={self.level}, children={len(self.children)})"


class AggregationTree:
    """Aggregation tree structure."""
    
    def __init__(self):
        self.root: Optional[TreeAggregationNode] = None
        self.nodes: Dict[int, TreeAggregationNode] = {}
    
    def add_node(self, node_id: int, level: int = 0):
        """Add a node to the tree."""
        node = TreeAggregationNode(node_id, level)
        self.nodes[node_id] = node
        
        if self.root is None:
            self.root = node
            node.is_server = True
    
    def add_edge(self, parent_id: int, child_id: int):
        """Add an edge between parent and child."""
        if parent_id in self.nodes and child_id in self.nodes:
            self.nodes[parent_id].add_child(self.nodes[child_id])
    
    def build_from_topology(self, topology, server_id: int):
        """Build aggregation tree from network topology."""
        communities = topology.find_communities()
        
        self.add_node(server_id, level=0)
        self.nodes[server_id].is_server = True
        
        for comm_id, community in enumerate(communities):
            aggregator_id = max(community) + 1000 + comm_id
            self.add_node(aggregator_id, level=1)
            self.add_edge(server_id, aggregator_id)
            
            for client_id in community:
                self.add_node(client_id, level=2)
                self.add_edge(aggregator_id, client_id)
    
    def aggregate(self) -> Optional[Dict[str, torch.Tensor]]:
        """Perform aggregation from leaves to root."""
        if self.root is None:
            return None
        
        return self.root.aggregate_up()
    
    def broadcast(self, model: Dict[str, torch.Tensor]):
        """Broadcast model from root to leaves."""
        if self.root is not None:
            self.root.broadcast_down(model)
    
    def get_tree_depth(self) -> int:
        """Get depth of the tree."""
        if self.root is None:
            return 0
        
        return self._get_depth(self.root)
    
    def _get_depth(self, node: TreeAggregationNode) -> int:
        """Recursively compute depth."""
        if not node.children:
            return node.level
        
        return max(self._get_depth(child) for child in node.children)
    
    def get_num_nodes(self) -> int:
        """Get number of nodes in tree."""
        return len(self.nodes)
    
    def __repr__(self):
        return f"AggregationTree(nodes={self.get_num_nodes()}, depth={self.get_tree_depth()})"


class HierarchicalAggregator:
    """Hierarchical aggregator for federated learning."""
    
    def __init__(self, topology, server_id: int = -1):
        self.topology = topology
        self.server_id = server_id
        self.aggregation_tree = AggregationTree()
        self.aggregation_tree.build_from_topology(topology, server_id)
        
        self.compression_rates: Dict[int, float] = {}
        self.privacy_budget: float = 1.0
    
    def set_compression_rate(self, node_id: int, rate: float):
        """Set compression rate for a node."""
        self.compression_rates[node_id] = rate
    
    def compress_model(self, model: Dict[str, torch.Tensor], 
                      rate: float) -> Dict[str, torch.Tensor]:
        """Compress model parameters."""
        compressed = {}
        
        for key, param in model.items():
            if rate < 1.0:
                compressed[key] = self._quantize(param, rate)
            else:
                compressed[key] = param
        
        return compressed
    
    def _quantize(self, param: torch.Tensor, rate: float) -> torch.Tensor:
        """Quantize tensor to reduce size."""
        scale = param.abs().max()
        bits = int(np.log2(1.0 / rate))
        
        if bits <= 0:
            return param
        
        levels = 2 ** bits
        quantized = torch.round(param / scale * (levels - 1)) / (levels - 1) * scale
        
        return quantized
    
    def add_dp_noise(self, model: Dict[str, torch.Tensor], 
                     noise_scale: float = 0.01) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise."""
        noisy = {}
        
        for key, param in model.items():
            noise = torch.randn_like(param) * noise_scale
            noisy[key] = param + noise
        
        self.privacy_budget = max(0.0, self.privacy_budget - 0.001)
        return noisy
    
    def aggregate_round(self, client_models: Dict[int, Dict[str, torch.Tensor]],
                       apply_dp: bool = False) -> Dict[str, torch.Tensor]:
        """
        Perform one round of hierarchical aggregation.
        
        Args:
            client_models: Dictionary of client models {client_id: model}
            apply_dp: Whether to apply differential privacy
        
        Returns:
            Aggregated global model
        """
        for node_id, node in self.aggregation_tree.nodes.items():
            if node_id in client_models:
                rate = self.compression_rates.get(node_id, 1.0)
                node.local_model = self.compress_model(client_models[node_id], rate)
        
        global_model = self.aggregation_tree.aggregate()
        
        if apply_dp and global_model is not None:
            global_model = self.add_dp_noise(global_model)
        
        self.aggregation_tree.broadcast(global_model)
        
        return global_model if global_model is not None else {}
    
    def update_topology(self, new_topology):
        """Update topology and rebuild aggregation tree."""
        self.topology = new_topology
        self.aggregation_tree = AggregationTree()
        self.aggregation_tree.build_from_topology(new_topology, self.server_id)
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about aggregation."""
        return {
            'tree_depth': self.aggregation_tree.get_tree_depth(),
            'num_nodes': self.aggregation_tree.get_num_nodes(),
            'num_clients': len([n for n in self.aggregation_tree.nodes.values() 
                              if n.level == 2]),
            'privacy_budget': self.privacy_budget,
            'compression_rates': self.compression_rates
        }


class GossipAggregator:
    """Gossip-based P2P aggregator."""
    
    def __init__(self, topology):
        self.topology = topology
        self.gossip_prob = 0.5
        self.max_rounds = 5
    
    def gossip_round(self, client_models: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Perform one round of gossip aggregation.
        
        Args:
            client_models: Dictionary of client models
        
        Returns:
            Updated client models after gossip
        """
        updated_models = client_models.copy()
        
        for client_id in client_models:
            neighbors = list(self.topology.graph.neighbors(client_id))
            
            for neighbor in neighbors:
                if np.random.random() < self.gossip_prob and neighbor in client_models:
                    updated_models[client_id] = self._average_models(
                        client_models[client_id],
                        client_models[neighbor]
                    )
        
        return updated_models
    
    def _average_models(self, model1: Dict[str, torch.Tensor],
                       model2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Average two models."""
        avg = {}
        for key in model1:
            if key in model2:
                avg[key] = (model1[key] + model2[key]) / 2
            else:
                avg[key] = model1[key]
        
        return avg
    
    def aggregate(self, client_models: Dict[int, Dict[str, torch.Tensor]],
                 rounds: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Perform full gossip aggregation.
        
        Args:
            client_models: Dictionary of client models
            rounds: Number of gossip rounds
        
        Returns:
            Aggregated global model
        """
        num_rounds = rounds or self.max_rounds
        
        current_models = client_models.copy()
        for _ in range(num_rounds):
            current_models = self.gossip_round(current_models)
        
        all_keys = set()
        for model in current_models.values():
            all_keys.update(model.keys())
        
        global_model = {}
        for key in all_keys:
            params = []
            for model in current_models.values():
                if key in model:
                    params.append(model[key])
            
            if params:
                global_model[key] = torch.mean(torch.stack(params), dim=0)
        
        return global_model


class HybridAggregator:
    """Hybrid aggregator combining hierarchical and gossip approaches."""
    
    def __init__(self, topology, server_id: int = -1):
        self.hierarchical = HierarchicalAggregator(topology, server_id)
        self.gossip = GossipAggregator(topology)
        
        self.use_gossip_prob = 0.3
    
    def aggregate(self, client_models: Dict[int, Dict[str, torch.Tensor]],
                 apply_dp: bool = False) -> Dict[str, torch.Tensor]:
        """
        Perform hybrid aggregation.
        
        Args:
            client_models: Dictionary of client models
            apply_dp: Whether to apply differential privacy
        
        Returns:
            Aggregated global model
        """
        if np.random.random() < self.use_gossip_prob:
            return self.gossip.aggregate(client_models)
        else:
            return self.hierarchical.aggregate_round(client_models, apply_dp)