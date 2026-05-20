"""
Topology Discovery for Federated Learning

Implements network topology detection, link quality monitoring, and graph modeling.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import time


class NetworkTopology:
    """Represents the network topology as a graph."""
    
    def __init__(self, graph_type: str = 'erdos_renyi'):
        self.graph_type = graph_type
        self.graph = nx.Graph()
        self.link_qualities: Dict[Tuple[int, int], Dict[str, float]] = {}
        
    def add_node(self, node_id: int, attributes: Optional[Dict[str, Any]] = None):
        """Add a node to the topology."""
        if attributes is None:
            attributes = {}
        self.graph.add_node(node_id, **attributes)
    
    def add_edge(self, node1: int, node2: int, 
                 latency: float = 1.0, bandwidth: float = 1.0, 
                 cost: float = 1.0):
        """Add an edge with link quality metrics."""
        self.graph.add_edge(node1, node2, 
                           latency=latency, 
                           bandwidth=bandwidth, 
                           cost=cost)
        
        self.link_qualities[(node1, node2)] = {
            'latency': latency,
            'bandwidth': bandwidth,
            'cost': cost,
            'last_update': time.time()
        }
        self.link_qualities[(node2, node1)] = {
            'latency': latency,
            'bandwidth': bandwidth,
            'cost': cost,
            'last_update': time.time()
        }
    
    def update_link_quality(self, node1: int, node2: int, **kwargs):
        """Update link quality metrics."""
        if (node1, node2) in self.link_qualities:
            self.link_qualities[(node1, node2)].update(kwargs)
            self.link_qualities[(node1, node2)]['last_update'] = time.time()
        
        if (node2, node1) in self.link_qualities:
            self.link_qualities[(node2, node1)].update(kwargs)
            self.link_qualities[(node2, node1)]['last_update'] = time.time()
        
        if self.graph.has_edge(node1, node2):
            for key, value in kwargs.items():
                self.graph[node1][node2][key] = value
    
    def get_shortest_path(self, source: int, target: int, 
                         weight: str = 'latency') -> List[int]:
        """Get shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target, weight=weight)
        except nx.NetworkXNoPath:
            return []
    
    def get_path_cost(self, path: List[int], weight: str = 'latency') -> float:
        """Calculate total cost of a path."""
        cost = 0.0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i+1]):
                cost += self.graph[path[i]][path[i+1]].get(weight, 1.0)
        return cost
    
    def find_communities(self, method: str = 'louvain') -> List[List[int]]:
        """Find communities in the topology graph."""
        if method == 'louvain':
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(self.graph)
                communities = defaultdict(list)
                for node, community_id in partition.items():
                    communities[community_id].append(node)
                return list(communities.values())
            except ImportError:
                return self._find_communities_greedy()
        else:
            return self._find_communities_greedy()
    
    def _find_communities_greedy(self) -> List[List[int]]:
        """Greedy community detection as fallback."""
        communities = []
        visited = set()
        
        for node in self.graph.nodes():
            if node not in visited:
                community = []
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        community.append(current)
                        queue.extend([n for n in self.graph.neighbors(current) 
                                     if n not in visited])
                communities.append(community)
        
        return communities
    
    def generate_random_topology(self, num_nodes: int, 
                                p: float = 0.1, seed: int = 42):
        """Generate a random topology."""
        np.random.seed(seed)
        
        if self.graph_type == 'erdos_renyi':
            self.graph = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
        elif self.graph_type == 'small_world':
            self.graph = nx.watts_strogatz_graph(num_nodes, 4, 0.1, seed=seed)
        elif self.graph_type == 'hierarchical':
            self.graph = self._generate_hierarchical_graph(num_nodes)
        else:
            self.graph = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
        
        for u, v in self.graph.edges():
            latency = np.random.uniform(0.1, 10.0)
            bandwidth = np.random.uniform(1.0, 100.0)
            cost = latency / bandwidth
            
            self.graph[u][v]['latency'] = latency
            self.graph[u][v]['bandwidth'] = bandwidth
            self.graph[u][v]['cost'] = cost
            
            self.link_qualities[(u, v)] = {
                'latency': latency,
                'bandwidth': bandwidth,
                'cost': cost,
                'last_update': time.time()
            }
            self.link_qualities[(v, u)] = {
                'latency': latency,
                'bandwidth': bandwidth,
                'cost': cost,
                'last_update': time.time()
            }
    
    def _generate_hierarchical_graph(self, num_nodes: int) -> nx.Graph:
        """Generate a hierarchical graph."""
        graph = nx.Graph()
        
        num_levels = 3
        nodes_per_level = num_nodes // num_levels
        
        server = num_nodes
        graph.add_node(server, type='server')
        
        for level in range(num_levels):
            start = level * nodes_per_level
            end = min((level + 1) * nodes_per_level, num_nodes)
            
            for i in range(start, end):
                graph.add_node(i, type='client', level=level)
                
                if level == 0:
                    latency = np.random.uniform(0.1, 1.0)
                    graph.add_edge(i, server, latency=latency, 
                                  bandwidth=100.0, cost=latency/100.0)
                else:
                    parent = np.random.randint((level-1)*nodes_per_level, level*nodes_per_level)
                    latency = np.random.uniform(1.0, 5.0)
                    graph.add_edge(i, parent, latency=latency, 
                                  bandwidth=50.0, cost=latency/50.0)
        
        return graph
    
    def get_topology_info(self) -> Dict[str, Any]:
        """Get summary information about the topology."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': np.mean([d for _, d in self.graph.degree()]),
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf'),
            'communities': self.find_communities()
        }


class TopologyDiscovery:
    """Discovers network topology through active probing."""
    
    def __init__(self):
        self.topology = NetworkTopology()
        self.monitor = LinkQualityMonitor()
    
    def discover(self, client_ids: List[int], 
                 server_id: int = -1, 
                 probe_count: int = 3) -> NetworkTopology:
        """
        Discover network topology by probing.
        
        Args:
            client_ids: List of client IDs to probe
            server_id: Server ID
            probe_count: Number of probes per link
        
        Returns:
            Discovered topology
        """
        self.topology.add_node(server_id, type='server')
        
        for client_id in client_ids:
            self.topology.add_node(client_id, type='client')
            
            latency = self._probe_latency(server_id, client_id, probe_count)
            bandwidth = self._estimate_bandwidth(client_id, probe_count)
            cost = latency / bandwidth if bandwidth > 0 else float('inf')
            
            self.topology.add_edge(server_id, client_id, 
                                  latency=latency, 
                                  bandwidth=bandwidth, 
                                  cost=cost)
        
        for i, client1 in enumerate(client_ids):
            for j, client2 in enumerate(client_ids):
                if i < j:
                    latency = self._probe_latency(client1, client2, probe_count)
                    if latency < float('inf'):
                        bandwidth = self._estimate_bandwidth_between(client1, client2)
                        cost = latency / bandwidth if bandwidth > 0 else float('inf')
                        self.topology.add_edge(client1, client2,
                                              latency=latency,
                                              bandwidth=bandwidth,
                                              cost=cost)
        
        return self.topology
    
    def _probe_latency(self, node1: int, node2: int, probe_count: int) -> float:
        """Simulate latency probing."""
        latencies = []
        for _ in range(probe_count):
            latencies.append(np.random.uniform(0.1, 5.0))
        
        return float(np.mean(latencies))
    
    def _estimate_bandwidth(self, client_id: int, probe_count: int) -> float:
        """Simulate bandwidth estimation."""
        bandwidths = []
        for _ in range(probe_count):
            bandwidths.append(np.random.uniform(10.0, 100.0))
        
        return float(np.mean(bandwidths))
    
    def _estimate_bandwidth_between(self, client1: int, client2: int) -> float:
        """Estimate bandwidth between two clients."""
        return np.random.uniform(5.0, 50.0)
    
    def update_topology(self) -> NetworkTopology:
        """Update topology based on current link qualities."""
        for (u, v), quality in self.topology.link_qualities.items():
            new_latency = quality['latency'] * (1 + np.random.uniform(-0.1, 0.1))
            new_bandwidth = quality['bandwidth'] * (1 + np.random.uniform(-0.1, 0.1))
            
            self.topology.update_link_quality(u, v, 
                                              latency=new_latency,
                                              bandwidth=new_bandwidth,
                                              cost=new_latency / new_bandwidth)
        
        return self.topology


class LinkQualityMonitor:
    """Monitors link quality over time."""
    
    def __init__(self):
        self.history: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
    
    def record_quality(self, node1: int, node2: int, **kwargs):
        """Record link quality measurement."""
        key = (node1, node2)
        if key not in self.history:
            self.history[key] = []
        
        self.history[key].append({
            'timestamp': time.time(),
            **kwargs
        })
        
        if len(self.history[key]) > 100:
            self.history[key] = self.history[key][-100:]
    
    def get_average_quality(self, node1: int, node2: int, 
                           window_size: int = 10) -> Optional[Dict[str, float]]:
        """Get average quality over recent measurements."""
        key = (node1, node2)
        if key not in self.history:
            return None
        
        recent = self.history[key][-window_size:]
        if not recent:
            return None
        
        avg = {}
        for metric in ['latency', 'bandwidth', 'cost']:
            values = [h.get(metric) for h in recent if h.get(metric) is not None]
            if values:
                avg[metric] = float(np.mean(values))
        
        return avg
    
    def detect_anomaly(self, node1: int, node2: int, threshold: float = 2.0) -> bool:
        """Detect anomalies in link quality."""
        avg = self.get_average_quality(node1, node2)
        if avg is None:
            return False
        
        recent = self.history[(node1, node2)][-3:]
        for h in recent:
            latency = h.get('latency')
            if latency and latency > avg.get('latency', 1.0) * threshold:
                return True
        
        return False
    
    def prune_stale_links(self, max_age: float = 300.0):
        """Remove links that haven't been updated recently."""
        now = time.time()
        stale = []
        
        for key, history in self.history.items():
            if history and (now - history[-1]['timestamp']) > max_age:
                stale.append(key)
        
        for key in stale:
            del self.history[key]