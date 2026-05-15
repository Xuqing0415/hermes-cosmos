"""
Feature Extractor for Meta-FL Controller

Extracts task-level meta-features for federated learning tasks.
"""

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskFeatureExtractor:
    """
    Extracts meta-features from federated learning tasks.
    
    Computes:
    - Statistical features (client count, samples, classes, dimensions)
    - Distributional features (Non-IID metrics)
    - Resource features (bandwidth, compute power)
    """
    
    def __init__(self):
        pass
    
    def extract(self, client_data: List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]],
               resources: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Extract all meta-features from task data.
        
        Args:
            client_data: List of ((X_train, y_train), (X_test, y_test)) tuples for each client
            resources: Dictionary of resource constraints
        
        Returns:
            Dictionary of meta-features
        """
        features = {}
        
        # Statistical features
        stats = self._compute_statistical_features(client_data)
        features.update(stats)
        
        # Distributional features (Non-IID metrics)
        dist = self._compute_distributional_features(client_data)
        features.update(dist)
        
        # Resource features
        if resources:
            res = self._compute_resource_features(resources)
            features.update(res)
        
        return features
    
    def _compute_statistical_features(self, client_data: List) -> Dict[str, float]:
        """
        Compute statistical features from client data.
        
        Returns:
            Dictionary of statistical meta-features
        """
        num_clients = len(client_data)
        
        # Get all labels to determine number of classes
        all_labels = []
        total_samples = 0
        feature_dim = 0
        
        for (X_train, y_train), _ in client_data:
            total_samples += len(y_train)
            all_labels.extend(y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train))
            if X_train.ndim > 1:
                feature_dim = max(feature_dim, X_train.shape[1])
        
        num_classes = len(set(all_labels))
        
        # Compute client sample distribution statistics
        sample_counts = [len(train[1]) for train, _ in client_data]
        mean_samples = np.mean(sample_counts)
        std_samples = np.std(sample_counts)
        max_samples = max(sample_counts)
        min_samples = min(sample_counts)
        
        return {
            'num_clients': float(num_clients),
            'total_samples': float(total_samples),
            'mean_samples_per_client': float(mean_samples),
            'std_samples_per_client': float(std_samples),
            'sample_imbalance_ratio': float(max_samples / (min_samples + 1e-8)),
            'num_classes': float(num_classes),
            'feature_dimension': float(feature_dim),
            'avg_samples_per_class': float(total_samples / max(num_classes, 1))
        }
    
    def _compute_distributional_features(self, client_data: List) -> Dict[str, float]:
        """
        Compute Non-IID metrics from client data.
        
        Returns:
            Dictionary of distributional meta-features
        """
        # Get label distributions per client
        num_clients = len(client_data)
        
        # Get all unique labels
        all_labels = []
        for (_, y_train), _ in client_data:
            all_labels.extend(y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train))
        
        if not all_labels:
            return {
                'non_iid_gini': 0.0,
                'non_iid_jsd': 0.0,
                'non_iid_entropy': 0.0,
                'class_imbalance': 0.0
            }
        
        unique_labels = sorted(set(all_labels))
        num_classes = len(unique_labels)
        
        # Build distribution matrix (clients x classes)
        distributions = np.zeros((num_clients, num_classes))
        
        for i, ((_, y_train), _) in enumerate(client_data):
            labels = y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train)
            for label in labels:
                if label in unique_labels:
                    distributions[i, unique_labels.index(label)] += 1
            
            # Normalize
            row_sum = distributions[i].sum()
            if row_sum > 0:
                distributions[i] /= row_sum
        
        # Compute Gini coefficient of label distribution
        gini = self._compute_gini_coefficient(distributions)
        
        # Compute Jensen-Shannon divergence between clients
        jsd = self._compute_avg_jsd(distributions)
        
        # Compute entropy of aggregated distribution
        global_dist = distributions.mean(axis=0)
        entropy_val = entropy(global_dist + 1e-10)
        
        # Compute class imbalance (max proportion / min proportion)
        class_counts = distributions.sum(axis=0)
        class_proportions = class_counts / class_counts.sum()
        class_imbalance = np.max(class_proportions) / (np.min(class_proportions) + 1e-8)
        
        return {
            'non_iid_gini': float(gini),
            'non_iid_jsd': float(jsd),
            'non_iid_entropy': float(entropy_val),
            'class_imbalance': float(class_imbalance)
        }
    
    def _compute_gini_coefficient(self, distributions: np.ndarray) -> float:
        """
        Compute Gini coefficient to measure Non-IID-ness.
        
        Higher Gini = more Non-IID.
        """
        # Compute the cumulative distribution across all clients for each class
        sorted_dist = np.sort(distributions.flatten())
        n = len(sorted_dist)
        
        # Gini coefficient formula
        numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_dist)
        denominator = n * np.sum(sorted_dist)
        
        return numerator / (denominator + 1e-8)
    
    def _compute_avg_jsd(self, distributions: np.ndarray) -> float:
        """
        Compute average Jensen-Shannon divergence between all client pairs.
        
        Higher JSD = more Non-IID.
        """
        n_clients = distributions.shape[0]
        total_jsd = 0.0
        count = 0
        
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                jsd = jensenshannon(distributions[i], distributions[j])
                total_jsd += jsd
                count += 1
        
        return total_jsd / max(count, 1)
    
    def _compute_resource_features(self, resources: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract resource-related features.
        
        Args:
            resources: Dictionary containing resource information
        
        Returns:
            Dictionary of resource features
        """
        return {
            'avg_bandwidth_mbps': float(resources.get('avg_bandwidth_mbps', 10.0)),
            'avg_compute_score': float(resources.get('avg_compute_score', 1.0)),
            'client_heterogeneity': float(resources.get('client_heterogeneity', 0.5)),
            'network_latency_ms': float(resources.get('network_latency_ms', 50.0)),
            'power_constraint': float(resources.get('power_constraint', 0.0))
        }


def compute_non_iid_metrics(client_data: List) -> Dict[str, float]:
    """
    Compute Non-IID metrics from client data.
    
    Args:
        client_data: List of ((X_train, y_train), (X_test, y_test)) tuples
    
    Returns:
        Dictionary of Non-IID metrics
    """
    extractor = TaskFeatureExtractor()
    features = extractor._compute_distributional_features(client_data)
    
    # Overall Non-IID score (0 = IID, 1 = highly Non-IID)
    metrics = list(features.values())
    overall_score = np.mean(metrics)
    
    features['non_iid_overall'] = float(overall_score)
    
    return features


def generate_synthetic_task(n_clients: int = 10, non_iid_level: float = 0.5,
                          num_classes: int = 10) -> Dict[str, float]:
    """
    Generate synthetic task features for testing.
    
    Args:
        n_clients: Number of clients
        non_iid_level: Non-IID level (0 = IID, 1 = highly Non-IID)
        num_classes: Number of classes
    
    Returns:
        Dictionary of synthetic task features
    """
    # Generate synthetic client data
    np.random.seed(42)
    
    client_data = []
    
    # Create Dirichlet distribution with specified Non-IID level
    alpha = 10.0 - (non_iid_level * 9.0)  # Lower alpha = more Non-IID
    
    for _ in range(n_clients):
        # Generate synthetic label distribution
        proportions = np.random.dirichlet([alpha] * num_classes)
        
        # Generate samples for each class
        n_samples = np.random.randint(100, 500)
        labels = []
        
        for cls, prop in enumerate(proportions):
            labels.extend([cls] * int(n_samples * prop))
        
        # Pad to desired length
        labels = labels[:n_samples]
        if len(labels) < n_samples:
            labels.extend([np.random.randint(0, num_classes) for _ in range(n_samples - len(labels))])
        
        # Create dummy tensors
        X_train = np.random.randn(len(labels), 784)
        y_train = np.array(labels)
        X_test = np.random.randn(50, 784)
        y_test = np.random.randint(0, num_classes, 50)
        
        client_data.append(((X_train, y_train), (X_test, y_test)))
    
    # Extract features
    extractor = TaskFeatureExtractor()
    
    resources = {
        'avg_bandwidth_mbps': np.random.uniform(5, 100),
        'avg_compute_score': np.random.uniform(0.3, 1.5),
        'client_heterogeneity': np.random.uniform(0.1, 0.9),
        'network_latency_ms': np.random.uniform(20, 200),
        'power_constraint': np.random.uniform(0, 1)
    }
    
    features = extractor.extract(client_data, resources)
    
    return features


if __name__ == "__main__":
    print("=== Testing Feature Extractor ===\n")
    
    # Test with synthetic data
    features = generate_synthetic_task(n_clients=20, non_iid_level=0.7)
    
    print("Extracted Meta-Features:")
    print("-" * 50)
    
    # Print features by category
    print("\nStatistical Features:")
    for key, value in features.items():
        if key in ['num_clients', 'total_samples', 'mean_samples_per_client', 
                   'std_samples_per_client', 'sample_imbalance_ratio', 
                   'num_classes', 'feature_dimension', 'avg_samples_per_class']:
            print(f"  {key}: {value:.4f}")
    
    print("\nDistributional Features (Non-IID):")
    for key, value in features.items():
        if key.startswith('non_iid') or key == 'class_imbalance':
            print(f"  {key}: {value:.4f}")
    
    print("\nResource Features:")
    for key, value in features.items():
        if key in ['avg_bandwidth_mbps', 'avg_compute_score', 'client_heterogeneity',
                   'network_latency_ms', 'power_constraint']:
            print(f"  {key}: {value:.4f}")
    
    # Interpret Non-IID level
    non_iid_score = features.get('non_iid_gini', 0)
    if non_iid_score < 0.3:
        interpretation = "Low Non-IID"
    elif non_iid_score < 0.6:
        interpretation = "Medium Non-IID"
    else:
        interpretation = "High Non-IID"
    
    print(f"\nNon-IID Interpretation: {interpretation}")
