"""
Cloud-Edge Federated Learning Runner

Main script to run cloud-edge federated learning experiments and compare with cloud-only training.
"""

import os
import sys
import time
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_cloud_edge_training(num_devices: int = 20, num_rounds: int = 20,
                            devices_per_round: int = 10, num_epochs: int = 1,
                            selection_strategy: str = 'random', async_mode: bool = False,
                            use_pruning: bool = False) -> dict:
    """
    Run cloud-edge federated training.
    
    Args:
        num_devices: Number of edge devices
        num_rounds: Number of federated rounds
        devices_per_round: Number of devices per round
        num_epochs: Local epochs per round
        selection_strategy: Device selection strategy
        async_mode: Use asynchronous training
        use_pruning: Enable adaptive model pruning
    
    Returns:
        Dictionary of results
    """
    from cloud_server import CloudEdgeCoordinator
    
    print("="*60)
    print("Running Cloud-Edge Federated Training")
    print("="*60)
    print(f"Devices: {num_devices}, Rounds: {num_rounds}")
    print(f"Strategy: {selection_strategy}, Async: {async_mode}, Pruning: {use_pruning}")
    print("="*60)
    
    start_time = time.time()
    
    coordinator = CloudEdgeCoordinator(num_devices=num_devices)
    coordinator.run_training(
        num_rounds=num_rounds,
        devices_per_round=devices_per_round,
        num_epochs=num_epochs,
        selection_strategy=selection_strategy,
        async_mode=async_mode
    )
    
    total_time = time.time() - start_time
    results = coordinator.get_results()
    
    results['total_time'] = total_time
    results['strategy'] = selection_strategy
    results['async_mode'] = async_mode
    results['use_pruning'] = use_pruning
    
    return results


def run_cloud_only_training(model_shape: tuple = (10, 784), num_epochs: int = 5,
                            batch_size: int = 32) -> dict:
    """
    Run cloud-only centralized training for comparison.
    
    Args:
        model_shape: Shape of model parameters
        num_epochs: Number of training epochs
        batch_size: Training batch size
    
    Returns:
        Dictionary of results
    """
    print("\n" + "="*60)
    print("Running Cloud-Only Centralized Training")
    print("="*60)
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}")
    print("="*60)
    
    start_time = time.time()
    
    # Initialize model
    model = np.random.randn(*model_shape) * 0.01
    
    # Create synthetic training data
    np.random.seed(42)
    num_samples = 10000
    X_train = np.random.randn(num_samples, model_shape[1]) * 0.5
    y_train = np.random.randint(0, model_shape[0], size=num_samples)
    
    # Create test data
    X_test = np.random.randn(1000, model_shape[1]) * 0.5
    y_test = np.random.randint(0, model_shape[0], size=1000)
    
    # Training loop
    accuracies = []
    losses = []
    
    lr = 0.01
    
    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(len(y_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        total_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(y_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Forward pass (linear model)
            logits = X_batch @ model.T
            predictions = np.argmax(logits, axis=1)
            
            # Compute loss (cross-entropy approximation)
            loss = np.mean((predictions != y_batch).astype(float))
            total_loss += loss
            num_batches += 1
            
            # Gradient descent update
            gradients = X_batch.T @ (logits - (np.eye(model_shape[0])[y_batch])) / len(y_batch)
            model -= lr * gradients
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # Evaluate
        test_logits = X_test @ model.T
        test_predictions = np.argmax(test_logits, axis=1)
        accuracy = np.mean(test_predictions == y_test)
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    total_time = time.time() - start_time
    
    results = {
        'accuracies': accuracies,
        'losses': losses,
        'final_accuracy': accuracies[-1],
        'total_time': total_time,
        'num_epochs': num_epochs,
        'model_size_mb': model.nbytes / (1024 * 1024)
    }
    
    return results


def plot_comparison(results_edge, results_cloud, output_dir: str = 'figures'):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        rounds = np.arange(1, len(results_edge['training_stats']['accuracies']) + 1)
        edge_acc = results_edge['training_stats']['accuracies']
        cloud_acc = results_cloud['accuracies']
        
        # Adjust cloud epochs to match edge rounds
        cloud_rounds = np.linspace(1, len(edge_acc), len(cloud_acc))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy comparison
        ax1.plot(rounds, edge_acc, 'b-', linewidth=2, label='Cloud-Edge')
        ax1.plot(cloud_rounds, cloud_acc, 'r--', linewidth=2, label='Cloud-Only')
        ax1.set_xlabel('Round/Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison: Cloud-Edge vs Cloud-Only')
        ax1.grid(True)
        ax1.legend()
        
        # Participation rate
        if 'participation_rates' in results_edge['training_stats']:
            ax2.plot(rounds, results_edge['training_stats']['participation_rates'], 'g-', linewidth=2, label='Participation Rate')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Participation Rate')
            ax2.set_title('Device Participation Rate')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cloud_edge_comparison.png'))
        plt.close()
        
        print(f"\nPlot saved to {os.path.join(output_dir, 'cloud_edge_comparison.png')}")
        
    except ImportError:
        print("\nMatplotlib not installed, skipping plot generation.")


def save_results(results_edge, results_cloud, output_dir: str = 'results'):
    """Save results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save edge results
    edge_path = os.path.join(output_dir, 'cloud_edge_results.json')
    with open(edge_path, 'w') as f:
        json.dump(results_edge, f, indent=2)
    
    # Save cloud results
    cloud_path = os.path.join(output_dir, 'cloud_only_results.json')
    with open(cloud_path, 'w') as f:
        json.dump(results_cloud, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")


def print_comparison_summary(results_edge, results_cloud):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("         CLOUD-EDGE COMPARISON SUMMARY         ")
    print("="*60)
    
    print("\n1. ACCURACY")
    print("-" * 30)
    edge_final = results_edge['training_stats']['accuracies'][-1]
    cloud_final = results_cloud['final_accuracy']
    print(f"Cloud-Edge Final Accuracy: {edge_final:.4f}")
    print(f"Cloud-Only Final Accuracy: {cloud_final:.4f}")
    
    diff = abs(edge_final - cloud_final)
    print(f"Difference: {diff:.4f}")
    
    if edge_final > cloud_final * 0.95:
        print("✓ Cloud-edge performance is within 5% of cloud-only!")
    else:
        print("⚠️  Cloud-edge performance is significantly worse")
    
    print("\n2. TRAINING TIME")
    print("-" * 30)
    print(f"Cloud-Edge Total Time: {results_edge['total_time']:.2f}s")
    print(f"Cloud-Only Total Time: {results_cloud['total_time']:.2f}s")
    
    time_ratio = results_edge['total_time'] / results_cloud['total_time']
    print(f"Time Ratio (Edge/Cloud): {time_ratio:.2f}x")
    
    print("\n3. COMMUNICATION")
    print("-" * 30)
    print(f"Total Communication: {results_edge['server_stats']['total_comm_cost']:.2f} MB")
    
    print("\n4. DEVICE STATISTICS")
    print("-" * 30)
    print(f"Number of Devices: {results_edge['server_stats']['num_devices']}")
    print(f"Average Participation Rate: {np.mean(results_edge['training_stats']['participation_rates']):.4f}")
    
    print("\n" + "="*60)


def main():
    """Main function to run cloud-edge federated learning experiments."""
    print("=== Cloud-Edge Federated Learning Experiment ===")
    print("Comparing Cloud-Edge vs Cloud-Only Training")
    print("="*60)
    
    # Run cloud-edge training
    edge_results = run_cloud_edge_training(
        num_devices=10,
        num_rounds=10,
        devices_per_round=5,
        num_epochs=1,
        selection_strategy='random',
        async_mode=False
    )
    
    # Run cloud-only training for comparison
    cloud_results = run_cloud_only_training(
        model_shape=(10, 784),
        num_epochs=5,
        batch_size=32
    )
    
    # Print comparison
    print_comparison_summary(edge_results, cloud_results)
    
    # Save results
    save_results(edge_results, cloud_results)
    
    # Generate plots
    plot_comparison(edge_results, cloud_results)
    
    print("\n=== Experiment Complete ===")


if __name__ == "__main__":
    main()
