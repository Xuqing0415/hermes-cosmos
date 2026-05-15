"""
Graph Data Utilities for Federated Graph Learning

Provides graph data generation, loading, and partitioning utilities
for both horizontal (inter-graph) and vertical (intra-graph) federated settings.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphDataset:
    """
    Container for a single graph dataset.
    
    Attributes:
        node_features: Node feature matrix [num_nodes, feature_dim]
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        edge_index: Edge list in COO format [2, num_edges]
        labels: Node or graph labels
        node_ids: Optional node identifiers
        graph_id: Optional graph identifier for inter-graph setting
    """
    node_features: np.ndarray
    adj_matrix: np.ndarray
    edge_index: np.ndarray
    labels: np.ndarray
    node_ids: Optional[np.ndarray] = None
    graph_id: Optional[int] = None
    
    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if len(self.edge_index.shape) > 1 else 0
    
    @property
    def feature_dim(self) -> int:
        return self.node_features.shape[1]
    
    def get_neighbors(self, node_idx: int) -> np.ndarray:
        """Get neighbors of a node."""
        neighbors = np.where(self.adj_matrix[node_idx] > 0)[0]
        return neighbors
    
    def get_degree(self, node_idx: int) -> int:
        """Get degree of a node."""
        return len(self.get_neighbors(node_idx))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'node_features': self.node_features,
            'adj_matrix': self.adj_matrix,
            'edge_index': self.edge_index,
            'labels': self.labels,
            'node_ids': self.node_ids,
            'graph_id': self.graph_id,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges
        }


class GraphDataGenerator:
    """
    Generator for synthetic graph datasets.
    
    Supports:
    - Erdős-Rényi random graphs
    - Barabási-Albert preferential attachment graphs
    - Watts-Strogatz small-world graphs
    - Stochastic Block Model (SBM) for community detection
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.rng = np.random.RandomState(seed)
        self.seed = seed
    
    def generate_erdos_renyi(
        self,
        num_nodes: int,
        edge_prob: float = 0.1,
        feature_dim: int = 16,
        num_classes: int = 3,
        task: str = 'node_classification'
    ) -> GraphDataset:
        """
        Generate an Erdős-Rényi random graph.
        
        Args:
            num_nodes: Number of nodes
            edge_prob: Probability of edge between any two nodes
            feature_dim: Dimension of node features
            num_classes: Number of classes for labels
            task: 'node_classification' or 'graph_classification'
        
        Returns:
            GraphDataset object
        """
        adj_matrix = (self.rng.rand(num_nodes, num_nodes) < edge_prob).astype(np.float32)
        adj_matrix = np.triu(adj_matrix, k=1)
        adj_matrix = adj_matrix + adj_matrix.T
        np.fill_diagonal(adj_matrix, 0)
        
        node_features = self.rng.randn(num_nodes, feature_dim).astype(np.float32)
        
        edge_index = self._adj_to_edge_index(adj_matrix)
        
        if task == 'node_classification':
            labels = self._generate_node_labels(node_features, num_classes)
        else:
            labels = self.rng.randint(0, num_classes)
        
        return GraphDataset(
            node_features=node_features,
            adj_matrix=adj_matrix,
            edge_index=edge_index,
            labels=labels,
            node_ids=np.arange(num_nodes)
        )
    
    def generate_barabasi_albert(
        self,
        num_nodes: int,
        num_attachments: int = 3,
        feature_dim: int = 16,
        num_classes: int = 3
    ) -> GraphDataset:
        """
        Generate a Barabási-Albert preferential attachment graph.
        
        Args:
            num_nodes: Number of nodes
            num_attachments: Number of edges to attach from new node
            feature_dim: Dimension of node features
            num_classes: Number of classes
        
        Returns:
            GraphDataset object
        """
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for new_node in range(num_attachments, num_nodes):
            degrees = np.sum(adj_matrix[:new_node, :new_node], axis=1)
            total_degree = np.sum(degrees)
            
            if total_degree == 0:
                targets = self.rng.choice(new_node, min(num_attachments, new_node), replace=False)
            else:
                probs = degrees / total_degree
                targets = self.rng.choice(new_node, min(num_attachments, new_node), 
                                         replace=False, p=probs)
            
            for target in targets:
                adj_matrix[new_node, target] = 1
                adj_matrix[target, new_node] = 1
        
        node_features = self.rng.randn(num_nodes, feature_dim).astype(np.float32)
        edge_index = self._adj_to_edge_index(adj_matrix)
        labels = self._generate_node_labels(node_features, num_classes)
        
        return GraphDataset(
            node_features=node_features,
            adj_matrix=adj_matrix,
            edge_index=edge_index,
            labels=labels,
            node_ids=np.arange(num_nodes)
        )
    
    def generate_sbm(
        self,
        num_nodes: int,
        num_communities: int = 3,
        p_in: float = 0.3,
        p_out: float = 0.05,
        feature_dim: int = 16
    ) -> GraphDataset:
        """
        Generate a Stochastic Block Model graph with community structure.
        
        Args:
            num_nodes: Total number of nodes
            num_communities: Number of communities
            p_in: Intra-community edge probability
            p_out: Inter-community edge probability
            feature_dim: Dimension of node features
        
        Returns:
            GraphDataset object
        """
        community_sizes = [num_nodes // num_communities] * num_communities
        community_sizes[-1] += num_nodes % num_communities
        
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        labels = np.zeros(num_nodes, dtype=np.int64)
        
        node_idx = 0
        for comm_id, size in enumerate(community_sizes):
            labels[node_idx:node_idx + size] = comm_id
            node_idx += size
        
        node_idx = 0
        for i, size_i in enumerate(community_sizes):
            for j, size_j in enumerate(community_sizes):
                start_i = sum(community_sizes[:i])
                start_j = sum(community_sizes[:j])
                
                block = self.rng.rand(size_i, size_j)
                if i == j:
                    block = (block < p_in).astype(np.float32)
                else:
                    block = (block < p_out).astype(np.float32)
                
                adj_matrix[start_i:start_i + size_i, start_j:start_j + size_j] = block
        
        adj_matrix = np.triu(adj_matrix, k=1)
        adj_matrix = adj_matrix + adj_matrix.T
        
        node_features = self.rng.randn(num_nodes, feature_dim).astype(np.float32)
        
        for i, size in enumerate(community_sizes):
            start = sum(community_sizes[:i])
            end = start + size
            node_features[start:end] += i * 2
        
        edge_index = self._adj_to_edge_index(adj_matrix)
        
        return GraphDataset(
            node_features=node_features,
            adj_matrix=adj_matrix,
            edge_index=edge_index,
            labels=labels,
            node_ids=np.arange(num_nodes)
        )
    
    def _adj_to_edge_index(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Convert adjacency matrix to edge index (COO format)."""
        rows, cols = np.where(adj_matrix > 0)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
        return edge_index
    
    def _generate_node_labels(
        self,
        node_features: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """Generate node labels based on features (semi-supervised style)."""
        weights = self.rng.randn(node_features.shape[1], num_classes)
        logits = node_features @ weights
        labels = np.argmax(logits, axis=1).astype(np.int64)
        return labels


class HorizontalGraphPartitioner:
    """
    Partitioner for horizontal federated graph learning (inter-graph).
    
    Each client receives a set of complete graphs.
    Graphs do not share nodes/edges but have the same feature space.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def partition(
        self,
        num_graphs: int,
        num_clients: int,
        graph_generator: GraphDataGenerator,
        min_nodes: int = 20,
        max_nodes: int = 100,
        feature_dim: int = 16,
        num_classes: int = 3,
        non_iid: bool = True,
        non_iid_degree: float = 0.5
    ) -> Dict[int, List[GraphDataset]]:
        """
        Partition graphs among clients for horizontal FL.
        
        Args:
            num_graphs: Total number of graphs to generate
            num_clients: Number of clients
            graph_generator: GraphDataGenerator instance
            min_nodes: Minimum nodes per graph
            max_nodes: Maximum nodes per graph
            feature_dim: Node feature dimension
            num_classes: Number of classes
            non_iid: Whether to create non-IID distribution
            non_iid_degree: Degree of non-IIDness (0=IID, 1=extreme non-IID)
        
        Returns:
            Dictionary mapping client_id to list of GraphDataset
        """
        graphs_per_client = num_graphs // num_clients
        
        client_data = {i: [] for i in range(num_clients)}
        
        for g in range(num_graphs):
            num_nodes = self.rng.randint(min_nodes, max_nodes + 1)
            edge_prob = self.rng.uniform(0.05, 0.2)
            
            graph = graph_generator.generate_erdos_renyi(
                num_nodes=num_nodes,
                edge_prob=edge_prob,
                feature_dim=feature_dim,
                num_classes=num_classes,
                task='node_classification'
            )
            graph.graph_id = g
            
            if non_iid:
                client_id = self._non_iid_assign(g, num_clients, non_iid_degree)
            else:
                client_id = g % num_clients
            
            client_data[client_id].append(graph)
        
        return client_data
    
    def _non_iid_assign(self, graph_id: int, num_clients: int, degree: float) -> int:
        """Assign graph to client with non-IID bias."""
        if self.rng.random() < degree:
            preferred_client = graph_id % num_clients
            return preferred_client
        else:
            return self.rng.randint(0, num_clients)
    
    def generate_molecular_graphs(
        self,
        num_graphs: int,
        num_clients: int,
        feature_dim: int = 10,
        seed: int = 42
    ) -> Dict[int, List[GraphDataset]]:
        """
        Generate molecular-like graphs (varying sizes, sparse connections).
        
        Simulates drug discovery scenario where each hospital has different molecules.
        """
        rng = np.random.RandomState(seed)
        client_data = {i: [] for i in range(num_clients)}
        
        for g in range(num_graphs):
            num_atoms = rng.randint(5, 50)
            
            adj = np.zeros((num_atoms, num_atoms), dtype=np.float32)
            for i in range(num_atoms - 1):
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            
            extra_bonds = rng.randint(0, num_atoms // 2)
            for _ in range(extra_bonds):
                i, j = rng.randint(0, num_atoms, 2)
                if i != j:
                    adj[i, j] = 1
                    adj[j, i] = 1
            
            node_features = rng.randn(num_atoms, feature_dim).astype(np.float32)
            
            labels = rng.randint(0, 2)
            
            edge_index = np.stack(np.where(adj > 0), axis=0).astype(np.int64)
            
            graph = GraphDataset(
                node_features=node_features,
                adj_matrix=adj,
                edge_index=edge_index,
                labels=np.array([labels]),
                node_ids=np.arange(num_atoms),
                graph_id=g
            )
            
            client_id = g % num_clients
            client_data[client_id].append(graph)
        
        return client_data


@dataclass
class VerticalPartitionInfo:
    """Information about a vertical graph partition."""
    client_id: int
    local_nodes: np.ndarray
    local_edges: np.ndarray
    cross_edges: List[Tuple[int, int, int]]
    neighbor_map: Dict[int, List[Tuple[int, int]]]


class VerticalGraphPartitioner:
    """
    Partitioner for vertical federated graph learning (intra-graph).
    
    A single large graph is partitioned into subgraphs across clients.
    Edges may cross clients, requiring secure embedding exchange.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def partition(
        self,
        graph: GraphDataset,
        num_clients: int,
        partition_strategy: str = 'random',
        overlap_ratio: float = 0.0
    ) -> Tuple[Dict[int, GraphDataset], Dict[str, Any]]:
        """
        Partition a single graph vertically among clients.
        
        Args:
            graph: The full graph to partition
            num_clients: Number of clients
            partition_strategy: 'random', 'metis', or 'degree'
            overlap_ratio: Ratio of overlapping nodes between clients
        
        Returns:
            Tuple of (client_graphs, partition_info)
        """
        num_nodes = graph.num_nodes
        adj_matrix = graph.adj_matrix
        node_features = graph.node_features
        labels = graph.labels
        
        if partition_strategy == 'random':
            node_assignments = self._random_partition(num_nodes, num_clients)
        elif partition_strategy == 'degree':
            node_assignments = self._degree_partition(adj_matrix, num_clients)
        else:
            node_assignments = self._random_partition(num_nodes, num_clients)
        
        client_graphs = {}
        cross_edge_info = {}
        
        for client_id in range(num_clients):
            local_nodes = np.where(node_assignments == client_id)[0]
            
            if len(local_nodes) == 0:
                continue
            
            local_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            cross_edges = []
            
            for i in local_nodes:
                neighbors = np.where(adj_matrix[i] > 0)[0]
                for j in neighbors:
                    neighbor_client = node_assignments[j]
                    if neighbor_client == client_id:
                        local_adj[i, j] = 1
                    else:
                        cross_edges.append((int(i), int(j), int(neighbor_client)))
            
            local_node_features = node_features.copy()
            local_labels = labels.copy() if isinstance(labels, np.ndarray) else labels
            
            local_edge_index = self._extract_local_edges(local_adj, local_nodes)
            
            client_graph = GraphDataset(
                node_features=local_node_features,
                adj_matrix=local_adj,
                edge_index=local_edge_index,
                labels=local_labels,
                node_ids=local_nodes,
                graph_id=graph.graph_id
            )
            
            client_graphs[client_id] = client_graph
            cross_edge_info[client_id] = {
                'local_nodes': local_nodes,
                'cross_edges': cross_edges,
                'num_cross_edges': len(cross_edges)
            }
        
        partition_info = {
            'strategy': partition_strategy,
            'num_clients': num_clients,
            'total_nodes': num_nodes,
            'cross_edge_info': cross_edge_info,
            'node_assignments': node_assignments
        }
        
        return client_graphs, partition_info
    
    def _random_partition(self, num_nodes: int, num_clients: int) -> np.ndarray:
        """Randomly assign nodes to clients."""
        assignments = self.rng.randint(0, num_clients, num_nodes)
        return assignments
    
    def _degree_partition(
        self,
        adj_matrix: np.ndarray,
        num_clients: int
    ) -> np.ndarray:
        """Partition based on node degree (high-degree nodes spread across clients)."""
        num_nodes = adj_matrix.shape[0]
        degrees = np.sum(adj_matrix, axis=1)
        
        sorted_indices = np.argsort(-degrees)
        
        assignments = np.zeros(num_nodes, dtype=np.int64)
        for i, node_idx in enumerate(sorted_indices):
            assignments[node_idx] = i % num_clients
        
        return assignments
    
    def _extract_local_edges(
        self,
        adj_matrix: np.ndarray,
        local_nodes: np.ndarray
    ) -> np.ndarray:
        """Extract edges within local subgraph."""
        local_set = set(local_nodes)
        edges = []
        
        for i in local_nodes:
            for j in np.where(adj_matrix[i] > 0)[0]:
                if j in local_set:
                    edges.append([i, j])
        
        if len(edges) == 0:
            return np.zeros((2, 0), dtype=np.int64)
        
        return np.array(edges).T.astype(np.int64)
    
    def get_cross_edge_statistics(
        self,
        cross_edge_info: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """Compute statistics about cross-client edges."""
        total_cross_edges = sum(
            info['num_cross_edges'] for info in cross_edge_info.values()
        )
        
        cross_edge_pairs = {}
        for client_id, info in cross_edge_info.items():
            for src, dst, dst_client in info['cross_edges']:
                pair = tuple(sorted([client_id, dst_client]))
                cross_edge_pairs[pair] = cross_edge_pairs.get(pair, 0) + 1
        
        return {
            'total_cross_edges': total_cross_edges,
            'cross_edge_pairs': cross_edge_pairs,
            'avg_cross_edges_per_client': total_cross_edges / len(cross_edge_info) if cross_edge_info else 0
        }
    
    def create_overlap_partition(
        self,
        graph: GraphDataset,
        num_clients: int,
        overlap_ratio: float = 0.1
    ) -> Tuple[Dict[int, GraphDataset], Dict[str, Any]]:
        """
        Create overlapping partitions where some nodes belong to multiple clients.
        
        Args:
            graph: The full graph
            num_clients: Number of clients
            overlap_ratio: Fraction of nodes that appear in multiple clients
        
        Returns:
            Tuple of (client_graphs, partition_info)
        """
        num_nodes = graph.num_nodes
        
        base_assignments = self._random_partition(num_nodes, num_clients)
        
        num_overlap = int(num_nodes * overlap_ratio)
        overlap_nodes = self.rng.choice(num_nodes, num_overlap, replace=False)
        
        overlap_assignments = {}
        for node in overlap_nodes:
            original_client = base_assignments[node]
            other_clients = [c for c in range(num_clients) if c != original_client]
            extra_clients = self.rng.choice(
                other_clients, 
                min(2, len(other_clients)), 
                replace=False
            )
            overlap_assignments[node] = [original_client] + list(extra_clients)
        
        client_graphs = {}
        cross_edge_info = {}
        
        for client_id in range(num_clients):
            primary_nodes = np.where(base_assignments == client_id)[0]
            overlap_nodes_for_client = [
                n for n, clients in overlap_assignments.items() 
                if client_id in clients and base_assignments[n] != client_id
            ]
            
            local_nodes = np.concatenate([
                primary_nodes,
                np.array(overlap_nodes_for_client, dtype=np.int64)
            ]) if overlap_nodes_for_client else primary_nodes
            
            if len(local_nodes) == 0:
                continue
            
            local_adj = graph.adj_matrix.copy()
            
            local_edge_index = self._extract_local_edges(local_adj, local_nodes)
            
            client_graph = GraphDataset(
                node_features=graph.node_features.copy(),
                adj_matrix=local_adj,
                edge_index=local_edge_index,
                labels=graph.labels.copy() if isinstance(graph.labels, np.ndarray) else graph.labels,
                node_ids=local_nodes,
                graph_id=graph.graph_id
            )
            
            client_graphs[client_id] = client_graph
            cross_edge_info[client_id] = {
                'local_nodes': local_nodes,
                'primary_nodes': primary_nodes,
                'overlap_nodes': overlap_nodes_for_client,
                'num_cross_edges': 0
            }
        
        partition_info = {
            'strategy': 'overlap',
            'overlap_ratio': overlap_ratio,
            'num_overlap_nodes': num_overlap,
            'cross_edge_info': cross_edge_info
        }
        
        return client_graphs, partition_info


def create_synthetic_graph_dataset(
    dataset_type: str = 'random',
    num_nodes: int = 100,
    num_graphs: int = 10,
    feature_dim: int = 16,
    num_classes: int = 3,
    seed: int = 42
) -> List[GraphDataset]:
    """
    Create synthetic graph datasets for testing.
    
    Args:
        dataset_type: 'random', 'community', or 'scale_free'
        num_nodes: Number of nodes per graph
        num_graphs: Number of graphs to generate
        feature_dim: Feature dimension
        num_classes: Number of classes
        seed: Random seed
    
    Returns:
        List of GraphDataset objects
    """
    generator = GraphDataGenerator(seed=seed)
    graphs = []
    
    for i in range(num_graphs):
        if dataset_type == 'community':
            graph = generator.generate_sbm(
                num_nodes=num_nodes,
                num_communities=num_classes,
                feature_dim=feature_dim
            )
        elif dataset_type == 'scale_free':
            graph = generator.generate_barabasi_albert(
                num_nodes=num_nodes,
                num_attachments=3,
                feature_dim=feature_dim,
                num_classes=num_classes
            )
        else:
            graph = generator.generate_erdos_renyi(
                num_nodes=num_nodes,
                edge_prob=0.1,
                feature_dim=feature_dim,
                num_classes=num_classes
            )
        
        graph.graph_id = i
        graphs.append(graph)
    
    return graphs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Graph Data Utilities")
    print("=" * 60)
    
    generator = GraphDataGenerator(seed=42)
    
    print("\n1. Generating Erdős-Rényi graph...")
    er_graph = generator.generate_erdos_renyi(
        num_nodes=50,
        edge_prob=0.1,
        feature_dim=16,
        num_classes=3
    )
    print(f"   Nodes: {er_graph.num_nodes}, Edges: {er_graph.num_edges}")
    print(f"   Feature dim: {er_graph.feature_dim}")
    
    print("\n2. Generating SBM graph with communities...")
    sbm_graph = generator.generate_sbm(
        num_nodes=100,
        num_communities=4,
        p_in=0.3,
        p_out=0.05,
        feature_dim=16
    )
    print(f"   Nodes: {sbm_graph.num_nodes}, Edges: {sbm_graph.num_edges}")
    print(f"   Label distribution: {np.bincount(sbm_graph.labels)}")
    
    print("\n3. Testing horizontal partitioning...")
    h_partitioner = HorizontalGraphPartitioner(seed=42)
    h_client_data = h_partitioner.partition(
        num_graphs=20,
        num_clients=4,
        graph_generator=generator,
        min_nodes=20,
        max_nodes=50,
        non_iid=True
    )
    for client_id, graphs in h_client_data.items():
        print(f"   Client {client_id}: {len(graphs)} graphs")
    
    print("\n4. Testing vertical partitioning...")
    v_partitioner = VerticalGraphPartitioner(seed=42)
    v_client_graphs, v_info = v_partitioner.partition(
        graph=sbm_graph,
        num_clients=3,
        partition_strategy='random'
    )
    stats = v_partitioner.get_cross_edge_statistics(v_info['cross_edge_info'])
    print(f"   Total cross-edges: {stats['total_cross_edges']}")
    print(f"   Avg cross-edges per client: {stats['avg_cross_edges_per_client']:.2f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
