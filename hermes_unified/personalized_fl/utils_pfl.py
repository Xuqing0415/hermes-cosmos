"""
Utility Functions for Personalized Federated Learning

Includes Non-IID data generation, evaluation metrics, and visualization tools.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Any, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_non_iid_cifar10(num_clients: int = 20, alpha: float = 0.5,
                           num_classes: int = 10, data_dir: str = './data') -> List[Tuple[Tuple, Tuple]]:
    """
    Create Non-IID CIFAR-10 data split across clients using Dirichlet distribution.
    
    Args:
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more Non-IID)
               alpha -> infinity: IID
               alpha -> 0: completely Non-IID (each client has only one class)
        num_classes: Number of classes in CIFAR-10
        data_dir: Directory to store/load CIFAR-10 data
    
    Returns:
        List of (train_data, test_data) tuples for each client
    """
    try:
        from torchvision import datasets, transforms
        has_torchvision = True
    except ImportError:
        has_torchvision = False
        logger.warning("torchvision not available, using synthetic data")
    
    if has_torchvision and os.path.exists(data_dir):
        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        except Exception as e:
            logger.warning(f"Failed to load CIFAR-10: {e}, using synthetic data")
            has_torchvision = False
    
    if not has_torchvision:
        # Generate synthetic data for testing
        logger.info("Generating synthetic CIFAR-like data")
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate 50000 training samples, 10000 test samples
        X_train = torch.randn(50000, 3, 32, 32)
        y_train = torch.randint(0, num_classes, (50000,))
        X_test = torch.randn(10000, 3, 32, 32)
        y_test = torch.randint(0, num_classes, (10000,))
        
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
    else:
        train_data = (torch.tensor(np.array(train_dataset.data)), 
                     torch.tensor(train_dataset.targets))
        test_data = (torch.tensor(np.array(test_dataset.data)), 
                    torch.tensor(test_dataset.targets))
    
    # Split data using Dirichlet distribution
    client_data = split_data_non_iid(train_data, test_data, num_clients, alpha, num_classes)
    
    return client_data


def split_data_non_iid(train_data: Tuple[torch.Tensor, torch.Tensor],
                       test_data: Tuple[torch.Tensor, torch.Tensor],
                       num_clients: int, alpha: float, num_classes: int) -> List[Tuple[Tuple, Tuple]]:
    """
    Split data across clients using Dirichlet distribution for Non-IID allocation.
    
    Args:
        train_data: (X_train, y_train)
        test_data: (X_test, y_test)
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        num_classes: Number of classes
    
    Returns:
        List of (train_data, test_data) for each client
    """
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    n_train = len(X_train)
    n_test = len(X_test)
    
    # Generate Dirichlet distribution for each class
    # For class c, client proportions are drawn from Dir(alpha)
    class_proportions = np.zeros((num_clients, num_classes))
    
    for c in range(num_classes):
        proportions = np.random.dirichlet([alpha] * num_clients)
        class_proportions[:, c] = proportions
    
    # Assign training samples to clients
    train_indices = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        class_indices = (y_train == c).nonzero(as_tuple=True)[0].numpy()
        np.random.shuffle(class_indices)
        
        # Split according to Dirichlet proportions
        start_idx = 0
        for client_id in range(num_clients):
            n_samples = int(len(class_indices) * class_proportions[client_id, c])
            end_idx = start_idx + n_samples
            train_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Assign test samples similarly
    test_indices = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        class_indices = (y_test == c).nonzero(as_tuple=True)[0].numpy()
        np.random.shuffle(class_indices)
        
        for client_id in range(num_clients):
            n_samples = int(len(class_indices) * class_proportions[client_id, c])
            start_idx = sum(int(len(class_indices) * class_proportions[j, c]) for j in range(client_id))
            end_idx = start_idx + n_samples
            test_indices[client_id].extend(class_indices[start_idx:end_idx])
    
    # Create client datasets
    client_data = []
    
    for client_id in range(num_clients):
        train_idx = np.array(train_indices[client_id])
        test_idx = np.array(test_indices[client_id])
        
        if len(train_idx) > 0:
            X_train_client = X_train[train_idx]
            y_train_client = y_train[train_idx]
        else:
            X_train_client = X_train[:1]  # Avoid empty tensor
            y_train_client = y_train[:1]
        
        if len(test_idx) > 0:
            X_test_client = X_test[test_idx]
            y_test_client = y_test[test_idx]
        else:
            X_test_client = X_test[:1]
            y_test_client = y_test[:1]
        
        client_data.append(((X_train_client, y_train_client), (X_test_client, y_test_client)))
    
    return client_data


def evaluate_personalized(clients: List, global_model: nn.Module = None) -> Dict[str, Any]:
    """
    Evaluate personalized models on each client's local test set.
    
    Args:
        clients: List of federated learning clients
        global_model: Optional global model for comparison
    
    Returns:
        Dictionary of evaluation metrics
    """
    personalized_accuracies = []
    client_dists = []  # Data distribution entropy per client
    
    for client in clients:
        acc = client.evaluate() if hasattr(client, 'evaluate') else 0
        personalized_accuracies.append(acc)
        
        # Compute data distribution entropy
        if hasattr(client, 'test_loader'):
            labels = []
            for _, y in client.test_loader:
                labels.extend(y.numpy())
            label_counts = np.bincount(labels, minlength=10)
            probs = label_counts / sum(label_counts)
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            client_dists.append(entropy)
    
    results = {
        'personalized_accuracies': personalized_accuracies,
        'mean_personalized_accuracy': np.mean(personalized_accuracies),
        'std_personalized_accuracy': np.std(personalized_accuracies),
        'min_personalized_accuracy': np.min(personalized_accuracies),
        'max_personalized_accuracy': np.max(personalized_accuracies),
        'client_entropies': client_dists if client_dists else None
    }
    
    return results


def compute_fairness_metrics(accuracies: List[float], percentiles: List[int] = [10, 50, 90]) -> Dict[str, Any]:
    """
    Compute fairness metrics for client accuracy distribution.
    
    Args:
        accuracies: List of client accuracies
        percentiles: Percentiles to compute for worst-case analysis
    
    Returns:
        Dictionary of fairness metrics
    """
    accuracies = np.array(accuracies)
    
    metrics = {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'median': np.median(accuracies),
        'min': np.min(accuracies),
        'max': np.max(accuracies),
        'range': np.max(accuracies) - np.min(accuracies)
    }
    
    # Compute percentiles
    for p in percentiles:
        metrics[f'p{p}'] = np.percentile(accuracies, p)
    
    # Jain's fairness index
    n = len(accuracies)
    if n > 0 and np.sum(accuracies) > 0:
        jains_index = np.sum(accuracies) ** 2 / (n * np.sum(accuracies ** 2))
    else:
        jains_index = 0
    metrics['jains_fairness_index'] = jains_index
    
    # Coefficient of variation
    if np.mean(accuracies) > 0:
        metrics['coefficient_of_variation'] = np.std(accuracies) / np.mean(accuracies)
    else:
        metrics['coefficient_of_variation'] = 0
    
    return metrics


def compare_methods(method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple personalized FL methods.
    
    Args:
        method_results: Dictionary mapping method names to their results
    
    Returns:
        Comparison table and analysis
    """
    comparison = {}
    
    for method_name, results in method_results.items():
        comparison[method_name] = {
            'final_accuracy': results.get('final_accuracy', 0),
            'final_global_accuracy': results.get('final_global_accuracy', 0),
            'personalized_accuracy': results.get('final_personalized_accuracy', 0)
        }
    
    # Find best method
    best_by_accuracy = max(comparison.items(), key=lambda x: x[1]['final_accuracy'])
    best_by_personalized = max(comparison.items(), key=lambda x: x[1]['personalized_accuracy'])
    
    return {
        'comparison_table': comparison,
        'best_by_global_accuracy': best_by_accuracy[0],
        'best_by_personalized': best_by_personalized[0]
    }


def print_comparison_table(comparison: Dict[str, Any]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("PERSONALIZED FEDERATED LEARNING COMPARISON")
    print("=" * 70)
    print(f"{'Method':<20} {'Global Acc':<15} {'Personalized Acc':<15} {'Fairness':<15}")
    print("-" * 70)
    
    for method, metrics in comparison.get('comparison_table', {}).items():
        print(f"{method:<20} {metrics['final_accuracy']:<15.4f} "
              f"{metrics['personalized_accuracy']:<15.4f} "
              f"{'N/A':<15}")
    
    print("=" * 70)
    print(f"\nBest by Global Accuracy: {comparison.get('best_by_global_accuracy', 'N/A')}")
    print(f"Best by Personalized: {comparison.get('best_by_personalized', 'N/A')}")


if __name__ == "__main__":
    print("=== Testing Personalized FL Utilities ===\n")
    
    # Test Non-IID data generation
    print("Creating Non-IID CIFAR-10 data split...")
    client_data = create_non_iid_cifar10(num_clients=10, alpha=0.5)
    
    print(f"Created data for {len(client_data)} clients")
    
    for i, ((X_train, y_train), (X_test, y_test)) in enumerate(client_data):
        print(f"  Client {i}: Train={len(X_train)}, Test={len(X_test)}")
    
    # Test fairness metrics
    accuracies = [0.8, 0.85, 0.75, 0.9, 0.7, 0.95, 0.6, 0.88, 0.72, 0.78]
    fairness = compute_fairness_metrics(accuracies)
    
    print("\nFairness Metrics:")
    for key, value in fairness.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
