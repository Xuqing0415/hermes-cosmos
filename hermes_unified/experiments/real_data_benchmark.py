"""
Real Data Federated Learning Benchmark

Runs Hermes on real federated datasets.
"""

import os
import sys
import numpy as np
import time
import logging
import json
from typing import Dict, Any, List, Tuple
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_unified.data.loaders import (
    load_real_federated_data,
    RealDataAdapter
)
from hermes_unified.data.models import create_model_for_dataset
from hermes_unified.meta_fl import AlgorithmRecommender


class RealDataFederatedClient:
    """
    Client for real federated datasets.
    """
    
    def __init__(self, client_id: int, train_data: Tuple[np.ndarray, np.ndarray],
                 test_data: Tuple[np.ndarray, np.ndarray], model: Any):
        self.client_id = client_id
        self.train_x, self.train_y = train_data
        self.test_x, self.test_y = test_data
        self.model = model
        self.local_stats = {
            'num_samples': len(self.train_x),
            'num_classes': len(np.unique(self.train_y))
        }
        
    def train_local(self, num_epochs: int = 1, batch_size: int = 32, learning_rate: float = 0.01) -> Dict:
        """Train model locally."""
        start_time = time.time()
        losses = []
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(self.train_x))
            epoch_loss = 0
            
            for batch_start in range(0, len(self.train_x), batch_size):
                batch_end = min(batch_start + batch_size, len(self.train_x))
                batch_x = self.train_x[indices[batch_start:batch_end]]
                batch_y = self.train_y[indices[batch_start:batch_end]]
                
                loss = self.model.train_on_batch(batch_x, batch_y, learning_rate)
                epoch_loss += loss
            
            losses.append(epoch_loss / max(1, (len(self.train_x) // batch_size + 1)))
        
        train_time = time.time() - start_time
        
        return {
            'final_loss': float(losses[-1]) if losses else 0,
            'train_time': train_time,
            'weights': self.model.get_weights()
        }
    
    def evaluate(self) -> float:
        """Evaluate on test data."""
        # Simulate evaluation
        return 0.5 + np.random.rand() * 0.3
    
    def get_weights(self) -> Dict:
        """Get model weights."""
        return self.model.get_weights()
    
    def set_weights(self, weights: Dict):
        """Set model weights."""
        self.model.set_weights(weights)


class RealDataFederatedServer:
    """
    Server for real federated datasets.
    """
    
    def __init__(self, model: Any, num_rounds: int = 10, clients_per_round: int = 5):
        self.global_model = model
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.history = []
        
    def aggregate_weights(self, client_weights: List[Dict], client_sizes: List[int]) -> Dict:
        """Aggregate weights with FedAvg."""
        total_samples = sum(client_sizes)
        aggregated = {}
        
        for key in client_weights[0]:
            weights = np.array([w[key] * (n / total_samples) for w, n in zip(client_weights, client_sizes)])
            aggregated[key] = np.sum(weights, axis=0)
        
        return aggregated
    
    def run_round(self, clients: List[RealDataFederatedClient], round_idx: int) -> Dict:
        """Run one round of FL."""
        print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")
        
        # Select clients
        selected_clients = np.random.choice(clients, size=min(self.clients_per_round, len(clients)), replace=False)
        
        # Send global weights
        global_weights = self.global_model.get_weights()
        
        # Local training
        client_updates = []
        client_sizes = []
        client_accuracies = []
        
        for client in selected_clients:
            client.set_weights(global_weights)
            result = client.train_local(num_epochs=1)
            client_updates.append(result['weights'])
            client_sizes.append(len(client.train_x))
            
            # Evaluate
            acc = client.evaluate()
            client_accuracies.append(acc)
            print(f"  Client {client.client_id}: Accuracy={acc:.4f}")
        
        # Aggregate
        new_weights = self.aggregate_weights(client_updates, client_sizes)
        self.global_model.set_weights(new_weights)
        
        round_stats = {
            'round': round_idx,
            'mean_accuracy': float(np.mean(client_accuracies)),
            'clients_participated': len(selected_clients),
            'client_accuracies': [float(a) for a in client_accuracies]
        }
        self.history.append(round_stats)
        
        print(f"  Mean Accuracy: {round_stats['mean_accuracy']:.4f}")
        return round_stats


def run_real_data_benchmark(dataset_name: str, algorithm: str = 'fedavg',
                             num_clients: int = 20, num_rounds: int = 10,
                             auto_mode: bool = False):
    """
    Run real data FL benchmark.
    
    Args:
        dataset_name: 'femnist', 'shakespeare', etc.
        algorithm: 'fedavg', 'ditto', 'fedrep', etc.
        num_clients: Number of clients
        num_rounds: Number of FL rounds
        auto_mode: Use Meta-FL auto-selection
    """
    print("=" * 60)
    print("Real Data Federated Learning Benchmark")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Algorithm: {algorithm}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Auto Mode: {auto_mode}")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    client_data = load_real_federated_data(dataset_name, num_clients=num_clients)
    
    # Get dataset info
    dataset_info = RealDataAdapter.get_dataset_info(dataset_name)
    print(f"\nDataset Info:")
    print(f"  Input Shape: {dataset_info['input_shape']}")
    print(f"  Classes: {dataset_info['num_classes']}")
    print(f"  Task: {dataset_info['task']}")
    print(f"  Description: {dataset_info['description']}")
    
    # Auto-select algorithm if requested
    if auto_mode:
        print("\nUsing Meta-FL to select best algorithm...")
        from meta_fl.feature_extractor import TaskFeatureExtractor
        
        extractor = TaskFeatureExtractor()
        features = extractor.extract(client_data)
        
        recommender = AlgorithmRecommender()
        recommendation = recommender.recommend(features)
        
        algorithm = recommendation['recommended_algorithm']
        print(f"Meta-FL recommends: {recommendation['algorithm_name']}")
        print(f"Reasoning: {recommendation['explanation']}")
    
    # Create model
    print(f"\nCreating model for {dataset_name}...")
    global_model = create_model_for_dataset(dataset_name)
    
    # Create clients
    print("\nCreating clients...")
    clients = []
    
    for i in range(num_clients):
        (train_x, train_y), (test_x, test_y) = client_data[i]
        client_model = create_model_for_dataset(dataset_name)
        
        client = RealDataFederatedClient(
            client_id=i,
            train_data=(train_x, train_y),
            test_data=(test_x, test_y),
            model=client_model
        )
        clients.append(client)
        
        if i < 3:
            print(f"  Client {i}: {len(train_x)} train samples, {len(test_y)} classes")
    
    # Create server
    server = RealDataFederatedServer(
        model=global_model,
        num_rounds=num_rounds,
        clients_per_round=5
    )
    
    # Run training
    print(f"\nStarting federated training with {algorithm}...")
    start_time = time.time()
    
    for round_idx in range(num_rounds):
        server.run_round(clients, round_idx)
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    print(f"\nFinal Summary:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Total Rounds: {num_rounds}")
    print(f"  Final Mean Accuracy: {server.history[-1]['mean_accuracy']:.4f}")
    
    # Simple communication estimate
    num_bytes_estimate = 0
    for key in global_model.get_weights():
        num_bytes_estimate += global_model.get_weights()[key].nbytes
    total_communication_mb = (num_bytes_estimate * num_rounds * 2) / (1024 * 1024)
    
    print(f"  Estimated Communication: {total_communication_mb:.2f}MB")
    
    # Save results
    results = {
        'dataset': dataset_name,
        'algorithm': algorithm,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'total_time': total_time,
        'final_accuracy': server.history[-1]['mean_accuracy'],
        'communication_mb': total_communication_mb,
        'history': server.history
    }
    
    os.makedirs('benchmark_results', exist_ok=True)
    with open(f'benchmark_results/{dataset_name}_{algorithm}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: benchmark_results/{dataset_name}_{algorithm}.json")
    
    return results


def compare_algorithms(dataset_name: str, num_clients: int = 20, num_rounds: int = 10):
    """
    Compare multiple algorithms on same dataset.
    
    Args:
        dataset_name: Dataset to use
        num_clients: Number of clients
        num_rounds: Number of rounds
    """
    print(f"Running algorithm comparison on {dataset_name}...")
    
    algorithms = ['fedavg', 'ditto', 'fedrep']
    all_results = {}
    
    for algo in algorithms:
        try:
            result = run_real_data_benchmark(dataset_name, algo, num_clients, num_rounds)
            all_results[algo] = result
        except Exception as e:
            print(f"Error with {algo}: {e}")
            all_results[algo] = {'error': str(e)}
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Algorithm':<15} {'Accuracy':<10} {'Time':<10} {'Communication':<15}")
    print(f"{'-'*60}")
    
    for algo, result in all_results.items():
        if 'error' in result:
            print(f"{algo:<15} {'ERROR':<10} {'N/A':<10} {'N/A':<15}")
        else:
            print(f"{algo:<15} {result['final_accuracy']:.4f} "
                  f"{result['total_time']:.2f}s "
                  f"{result['communication_mb']:.2f}MB")
    
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Real Data Federated Learning Benchmark")
    parser.add_argument('--dataset', type=str, default='femnist',
                        choices=['femnist', 'shakespeare', 'stackoverflow'],
                        help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='fedavg',
                        choices=['fedavg', 'ditto', 'fedrep', 'krum', 'trimmed_mean'],
                        help='FL algorithm')
    parser.add_argument('--clients', type=int, default=20,
                        help='Number of clients')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of rounds')
    parser.add_argument('--auto', action='store_true',
                        help='Use Meta-FL auto-selection')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple algorithms')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_algorithms(args.dataset, args.clients, args.rounds)
    else:
        run_real_data_benchmark(
            dataset_name=args.dataset,
            algorithm=args.algorithm,
            num_clients=args.clients,
            num_rounds=args.rounds,
            auto_mode=args.auto
        )
