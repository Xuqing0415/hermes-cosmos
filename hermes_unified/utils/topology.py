"""
Network Topology Utilities

Provides utilities for defining and managing network topologies
for distributed training simulations.
"""

import numpy as np
from typing import Dict, List, Tuple


class NetworkTopology:
    """
    Network topology definition and utilities.
    
    Supports:
    - Full mesh topology
    - Tree topology (hierarchical)
    - Ring topology
    - Custom topology
    """
    
    def __init__(self, num_nodes: int):
        """
        Initialize network topology.
        
        Args:
            num_nodes: Number of nodes (workers + optional PS)
        """
        self.num_nodes = num_nodes
        self.bandwidth = np.zeros((num_nodes, num_nodes))
        self.latency = np.zeros((num_nodes, num_nodes))
        
        # Initialize with default values
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    self.bandwidth[i, j] = float('inf')
                    self.latency[i, j] = 0.0
                else:
                    self.bandwidth[i, j] = 100.0  # MB/s
                    self.latency[i, j] = 0.0001   # seconds
    
    def set_full_mesh(self, bandwidth: float = 100.0, latency: float = 0.0001):
        """
        Set full mesh topology (all nodes connected to all others).
        
        Args:
            bandwidth: Bandwidth between nodes in MB/s
            latency: Latency between nodes in seconds
        """
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    self.bandwidth[i, j] = bandwidth
                    self.latency[i, j] = latency
    
    def set_tree(self, root_bandwidth: float = 150.0, leaf_bandwidth: float = 60.0,
                 root_latency: float = 0.00008, leaf_latency: float = 0.00015):
        """
        Set tree topology (hierarchical).
        
        Root node (node 0) has higher bandwidth to leaves.
        Leaf nodes have lower bandwidth between themselves.
        
        Args:
            root_bandwidth: Bandwidth from root to leaves
            leaf_bandwidth: Bandwidth between leaves
            root_latency: Latency from root to leaves
            leaf_latency: Latency between leaves
        """
        # Root to leaves
        for i in range(1, self.num_nodes):
            self.bandwidth[0, i] = root_bandwidth
            self.bandwidth[i, 0] = root_bandwidth
            self.latency[0, i] = root_latency
            self.latency[i, 0] = root_latency
        
        # Leaves to leaves
        for i in range(1, self.num_nodes):
            for j in range(1, self.num_nodes):
                if i != j:
                    self.bandwidth[i, j] = leaf_bandwidth
                    self.latency[i, j] = leaf_latency
    
    def set_ring(self, bandwidth: float = 100.0, latency: float = 0.0001):
        """
        Set ring topology (each node connected to two neighbors).
        
        Args:
            bandwidth: Bandwidth between adjacent nodes
            latency: Latency between adjacent nodes
        """
        # Clear all connections
        self.bandwidth[:] = 0.0
        self.latency[:] = float('inf')
        
        # Set ring connections
        for i in range(self.num_nodes):
            prev = (i - 1) % self.num_nodes
            next_node = (i + 1) % self.num_nodes
            
            self.bandwidth[i, prev] = bandwidth
            self.bandwidth[i, next_node] = bandwidth
            self.latency[i, prev] = latency
            self.latency[i, next_node] = latency
            
            # Diagonal for identity
            self.bandwidth[i, i] = float('inf')
            self.latency[i, i] = 0.0
    
    def set_custom(self, edges: List[Tuple[int, int, float, float]]):
        """
        Set custom topology from list of edges.
        
        Args:
            edges: List of tuples (src, dst, bandwidth, latency)
        """
        for src, dst, bw, lat in edges:
            if 0 <= src < self.num_nodes and 0 <= dst < self.num_nodes:
                self.bandwidth[src, dst] = bw
                self.latency[src, dst] = lat
    
    def add_heterogeneity(self, bw_variation: float = 0.3, lat_variation: float = 0.5):
        """
        Add heterogeneity to the network.
        
        Args:
            bw_variation: Coefficient of variation for bandwidth (0-1)
            lat_variation: Coefficient of variation for latency (0-1)
        """
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and self.bandwidth[i, j] > 0:
                    # Add variation to bandwidth
                    bw_noise = np.random.normal(1.0, bw_variation)
                    self.bandwidth[i, j] = max(1.0, self.bandwidth[i, j] * bw_noise)
                    
                    # Add variation to latency
                    lat_noise = np.random.normal(1.0, lat_variation)
                    self.latency[i, j] = max(0.00001, self.latency[i, j] * lat_noise)
    
    def get_bandwidth(self, src: int, dst: int) -> float:
        """Get bandwidth between two nodes."""
        if 0 <= src < self.num_nodes and 0 <= dst < self.num_nodes:
            return self.bandwidth[src, dst]
        return 0.0
    
    def get_latency(self, src: int, dst: int) -> float:
        """Get latency between two nodes."""
        if 0 <= src < self.num_nodes and 0 <= dst < self.num_nodes:
            return self.latency[src, dst]
        return float('inf')
    
    def compute_path(self, src: int, dst: int) -> Tuple[List[int], float, float]:
        """
        Compute shortest path using Dijkstra's algorithm.
        
        Returns:
            Tuple of (path, total_bandwidth, total_latency)
        """
        if src == dst:
            return [src], float('inf'), 0.0
        
        # Dijkstra's algorithm based on latency
        dist = [float('inf')] * self.num_nodes
        prev = [None] * self.num_nodes
        dist[src] = 0.0
        
        unvisited = set(range(self.num_nodes))
        
        while unvisited:
            # Find node with minimum distance
            u = min(unvisited, key=lambda x: dist[x])
            unvisited.remove(u)
            
            if dist[u] == float('inf'):
                break
            
            # Update neighbors
            for v in range(self.num_nodes):
                if v != u and self.bandwidth[u, v] > 0:
                    alt = dist[u] + self.latency[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
        
        # Reconstruct path
        path = []
        current = dst
        while current is not None:
            path.insert(0, current)
            current = prev[current]
        
        if not path or path[0] != src:
            return [], 0.0, float('inf')
        
        # Compute total bandwidth (minimum along path) and latency
        total_latency = dist[dst]
        min_bandwidth = float('inf')
        
        for i in range(len(path) - 1):
            min_bandwidth = min(min_bandwidth, self.bandwidth[path[i], path[i+1]])
        
        return path, min_bandwidth, total_latency
    
    def get_topology_matrix(self) -> Dict[str, np.ndarray]:
        """Get topology matrices as dictionary."""
        return {
            'bandwidth': self.bandwidth.copy(),
            'latency': self.latency.copy()
        }
    
    def print_topology(self):
        """Print topology summary."""
        print("Network Topology Summary:")
        print(f"Number of nodes: {self.num_nodes}")
        print("\nBandwidth matrix (MB/s):")
        print(self.bandwidth)
        print("\nLatency matrix (s):")
        print(self.latency)


class TopologyGenerator:
    """
    Generator for common network topologies.
    """
    
    @staticmethod
    def generate_datacenter_topology(num_workers: int) -> NetworkTopology:
        """
        Generate a datacenter-like topology.
        
        PS is node 0, workers are nodes 1-N.
        PS has high bandwidth to all workers.
        Worker-worker connections have lower bandwidth.
        """
        topology = NetworkTopology(num_workers + 1)
        
        # PS to workers (high bandwidth)
        for i in range(1, num_workers + 1):
            topology.bandwidth[0, i] = 200.0 + np.random.normal(0, 20)
            topology.bandwidth[i, 0] = topology.bandwidth[0, i]
            topology.latency[0, i] = 0.00005 + np.random.normal(0, 0.00001)
            topology.latency[i, 0] = topology.latency[0, i]
        
        # Worker to worker (lower bandwidth)
        for i in range(1, num_workers + 1):
            for j in range(1, num_workers + 1):
                if i != j:
                    topology.bandwidth[i, j] = 50.0 + np.random.normal(0, 10)
                    topology.latency[i, j] = 0.0002 + np.random.normal(0, 0.00005)
        
        return topology
    
    @staticmethod
    def generate_cross_region_topology(num_regions: int, workers_per_region: int) -> NetworkTopology:
        """
        Generate a cross-region topology.
        
        Args:
            num_regions: Number of regions
            workers_per_region: Workers per region
        
        Returns:
            NetworkTopology with regional characteristics
        """
        total_nodes = num_regions * workers_per_region
        
        topology = NetworkTopology(total_nodes)
        
        # Inter-region latency (higher)
        inter_region_latency = 0.05  # 50ms between regions
        inter_region_bandwidth = 20.0  # 20 MB/s between regions
        
        # Intra-region latency (lower)
        intra_region_latency = 0.0001  # 0.1ms within region
        intra_region_bandwidth = 100.0  # 100 MB/s within region
        
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i == j:
                    continue
                
                region_i = i // workers_per_region
                region_j = j // workers_per_region
                
                if region_i == region_j:
                    # Same region
                    topology.bandwidth[i, j] = intra_region_bandwidth * (0.8 + np.random.random() * 0.4)
                    topology.latency[i, j] = intra_region_latency * (0.8 + np.random.random() * 0.4)
                else:
                    # Different region
                    topology.bandwidth[i, j] = inter_region_bandwidth * (0.8 + np.random.random() * 0.4)
                    topology.latency[i, j] = inter_region_latency * (0.8 + np.random.random() * 0.4)
        
        return topology
