"""
Topology-Aware Scheduler for Federated Learning

Implements client selection based on graph algorithms and topology.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class GraphBasedSelector:
    """Client selector based on graph properties."""
    
    def __init__(self, topology):
        self.topology = topology
        self.alpha = 0.5
        self.beta = 0.5
    
    def set_cost_weights(self, alpha: float, beta: float):
        """Set weights for cost function."""
        self.alpha = alpha
        self.beta = beta
    
    def compute_client_cost(self, client_id: int, server_id: int) -> float:
        """
        Compute cost for selecting a client.
        
        Args:
            client_id: Client ID
            server_id: Server ID
        
        Returns:
            Cost value
        """
        path = self.topology.get_shortest_path(client_id, server_id)
        if not path:
            return float('inf')
        
        latency = self.topology.get_path_cost(path, weight='latency')
        bandwidth = self.topology.get_path_cost(path, weight='bandwidth')
        
        bytes_transmitted = 1000.0
        
        round_time = latency
        cost = self.alpha * round_time + self.beta * (bytes_transmitted / (bandwidth + 1e-6))
        
        return cost
    
    def select_clients(self, client_ids: List[int], server_id: int,
                      num_clients: int, strategy: str = 'min_cost') -> List[int]:
        """
        Select clients based on topology-aware strategy.
        
        Args:
            client_ids: List of available client IDs
            server_id: Server ID
            num_clients: Number of clients to select
            strategy: Selection strategy
        
        Returns:
            List of selected client IDs
        """
        if strategy == 'min_cost':
            return self._select_min_cost(client_ids, server_id, num_clients)
        elif strategy == 'diverse':
            return self._select_diverse(client_ids, server_id, num_clients)
        elif strategy == 'centrality':
            return self._select_centrality(client_ids, num_clients)
        else:
            return self._select_min_cost(client_ids, server_id, num_clients)
    
    def _select_min_cost(self, client_ids: List[int], server_id: int,
                        num_clients: int) -> List[int]:
        """Select clients with minimum communication cost."""
        costs = []
        for client_id in client_ids:
            cost = self.compute_client_cost(client_id, server_id)
            costs.append((cost, client_id))
        
        costs.sort()
        return [c[1] for c in costs[:num_clients]]
    
    def _select_diverse(self, client_ids: List[int], server_id: int,
                       num_clients: int) -> List[int]:
        """Select diverse clients across different regions."""
        communities = self.topology.find_communities()
        
        selected = []
        clients_per_community = max(1, num_clients // len(communities))
        
        for community in communities:
            community_clients = [c for c in client_ids if c in community]
            
            if community_clients:
                costs = []
                for client_id in community_clients:
                    cost = self.compute_client_cost(client_id, server_id)
                    costs.append((cost, client_id))
                
                costs.sort()
                selected.extend([c[1] for c in costs[:clients_per_community]])
        
        return selected[:num_clients]
    
    def _select_centrality(self, client_ids: List[int], num_clients: int) -> List[int]:
        """Select clients with highest graph centrality."""
        betweenness = nx.betweenness_centrality(self.topology.graph)
        
        client_centrality = [(betweenness.get(c, 0), c) 
                            for c in client_ids]
        
        client_centrality.sort(reverse=True)
        return [c[1] for c in client_centrality[:num_clients]]


class SteinerTreeOptimizer:
    """Optimizes client selection using Steiner tree approximation."""
    
    def __init__(self, topology):
        self.topology = topology
    
    def find_steiner_tree(self, terminals: List[int], 
                          root: int = -1) -> nx.Graph:
        """
        Find Steiner tree connecting terminals.
        
        Args:
            terminals: List of terminal nodes
            root: Root node (server)
        
        Returns:
            Steiner tree subgraph
        """
        all_nodes = terminals + [root]
        subgraph = self.topology.graph.subgraph(all_nodes)
        
        mst = nx.minimum_spanning_tree(subgraph, weight='cost')
        
        return mst
    
    def optimize_client_selection(self, candidate_clients: List[int],
                                server_id: int, num_clients: int) -> List[int]:
        """
        Optimize client selection using Steiner tree.
        
        Args:
            candidate_clients: List of candidate clients
            server_id: Server ID
            num_clients: Number of clients to select
        
        Returns:
            List of selected client IDs
        """
        if len(candidate_clients) <= num_clients:
            return candidate_clients
        
        best_cost = float('inf')
        best_selection = []
        
        from itertools import combinations
        
        for combination in combinations(candidate_clients, num_clients):
            tree = self.find_steiner_tree(list(combination), server_id)
            total_cost = sum(tree[u][v].get('cost', 1.0) for u, v in tree.edges())
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_selection = list(combination)
        
        return best_selection if best_selection else candidate_clients[:num_clients]


class TopologyScheduler:
    """Main scheduler for topology-aware federated learning."""
    
    def __init__(self, topology, server_id: int = -1):
        self.topology = topology
        self.server_id = server_id
        
        self.selector = GraphBasedSelector(topology)
        self.steiner_optimizer = SteinerTreeOptimizer(topology)
        
        self.participation_history: Dict[int, List[int]] = defaultdict(list)
        self.selection_strategy = 'min_cost'
    
    def set_strategy(self, strategy: str):
        """Set client selection strategy."""
        self.selection_strategy = strategy
    
    def select_participants(self, client_ids: List[int], num_clients: int,
                          round_idx: int) -> List[int]:
        """
        Select participating clients for a round.
        
        Args:
            client_ids: List of available clients
            num_clients: Number of clients to select
            round_idx: Current round index
        
        Returns:
            List of selected client IDs
        """
        if self.selection_strategy == 'steiner':
            selected = self.steiner_optimizer.optimize_client_selection(
                client_ids, self.server_id, num_clients
            )
        else:
            selected = self.selector.select_clients(
                client_ids, self.server_id, num_clients, self.selection_strategy
            )
        
        self.participation_history[round_idx] = selected
        
        return selected
    
    def get_selection_cost(self, selected_clients: List[int]) -> float:
        """
        Compute total selection cost.
        
        Args:
            selected_clients: List of selected client IDs
        
        Returns:
            Total communication cost
        """
        total_cost = 0.0
        for client_id in selected_clients:
            total_cost += self.selector.compute_client_cost(client_id, self.server_id)
        
        return total_cost
    
    def adjust_for_fairness(self, selected_clients: List[int],
                           fairness_threshold: float = 0.8) -> List[int]:
        """
        Adjust selection to ensure fairness.
        
        Args:
            selected_clients: Initially selected clients
            fairness_threshold: Minimum participation rate
        
        Returns:
            Adjusted selection
        """
        all_clients = list(self.topology.graph.nodes())
        all_clients = [c for c in all_clients if c != self.server_id]
        
        participation_counts = defaultdict(int)
        for round_clients in self.participation_history.values():
            for client in round_clients:
                participation_counts[client] += 1
        
        total_rounds = len(self.participation_history) + 1
        
        low_participants = [
            c for c in all_clients 
            if participation_counts[c] / total_rounds < fairness_threshold
        ]
        
        if low_participants and len(selected_clients) < len(all_clients):
            num_to_add = min(len(low_participants), 2)
            
            for client in low_participants[:num_to_add]:
                if client not in selected_clients:
                    selected_clients.append(client)
        
        return selected_clients
    
    def update_topology(self, new_topology):
        """Update topology for scheduler."""
        self.topology = new_topology
        self.selector = GraphBasedSelector(new_topology)
        self.steiner_optimizer = SteinerTreeOptimizer(new_topology)
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get statistics about scheduling."""
        participation_counts = defaultdict(int)
        for round_clients in self.participation_history.values():
            for client in round_clients:
                participation_counts[client] += 1
        
        if participation_counts:
            avg_participation = np.mean(list(participation_counts.values()))
            min_participation = min(participation_counts.values())
            max_participation = max(participation_counts.values())
        else:
            avg_participation = 0
            min_participation = 0
            max_participation = 0
        
        return {
            'total_rounds': len(self.participation_history),
            'avg_participation': avg_participation,
            'min_participation': min_participation,
            'max_participation': max_participation,
            'strategy': self.selection_strategy,
            'num_clients': len([n for n in self.topology.graph.nodes() if n != self.server_id])
        }


class TopologyAdaptiveController:
    """Adaptive controller that adjusts based on topology changes."""
    
    def __init__(self, scheduler: TopologyScheduler, 
                 aggregator, adaptation_interval: int = 10):
        self.scheduler = scheduler
        self.aggregator = aggregator
        self.adaptation_interval = adaptation_interval
        self.round_count = 0
        
        self.performance_history = []
        self.adaptation_threshold = 0.1
    
    def should_adapt(self) -> bool:
        """Check if adaptation is needed."""
        if self.round_count % self.adaptation_interval != 0:
            return False
        
        if len(self.performance_history) < 3:
            return False
        
        recent_performance = self.performance_history[-3:]
        avg_performance = np.mean(recent_performance)
        
        if avg_performance < self.adaptation_threshold:
            return True
        
        return False
    
    def adapt(self, current_performance: float):
        """
        Adapt to changing conditions.
        
        Args:
            current_performance: Current performance metric
        """
        self.performance_history.append(current_performance)
        
        if self.should_adapt():
            self._adjust_strategy()
            self._update_compression()
    
    def _adjust_strategy(self):
        """Adjust client selection strategy."""
        strategies = ['min_cost', 'diverse', 'centrality', 'steiner']
        current_idx = strategies.index(self.scheduler.selection_strategy)
        next_idx = (current_idx + 1) % len(strategies)
        
        self.scheduler.set_strategy(strategies[next_idx])
    
    def _update_compression(self):
        """Update compression rates based on link quality."""
        for node_id in self.scheduler.topology.graph.nodes():
            if node_id == self.scheduler.server_id:
                continue
            
            latency = self.scheduler.topology.get_path_cost(
                self.scheduler.topology.get_shortest_path(node_id, self.scheduler.server_id),
                weight='latency'
            )
            
            if latency > 5.0:
                self.aggregator.set_compression_rate(node_id, 0.5)
            elif latency > 2.0:
                self.aggregator.set_compression_rate(node_id, 0.8)
            else:
                self.aggregator.set_compression_rate(node_id, 1.0)
    
    def record_round(self, performance: float):
        """Record round performance."""
        self.round_count += 1
        self.adapt(performance)
    
    def reset(self):
        """Reset adaptation state."""
        self.round_count = 0
        self.performance_history = []