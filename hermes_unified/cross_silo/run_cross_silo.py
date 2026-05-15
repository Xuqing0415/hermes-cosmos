"""
Cross-Silo Federated Learning Runner

Main script to run cross-silo federated learning experiments.
"""

import os
import sys
import time
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_cross_silo_experiment(num_silos: int = 3, clients_per_silo: int = 10,
                              num_rounds: int = 20, internal_rounds: int = 5,
                              inter_bandwidth: float = 5.0, inter_latency: int = 100) -> dict:
    """
    Run cross-silo federated learning experiment.
    
    Args:
        num_silos: Number of silos/organizations
        clients_per_silo: Number of clients per silo
        num_rounds: Number of training rounds
        internal_rounds: Internal rounds per external round
        inter_bandwidth: Bandwidth between silos in Mbps
        inter_latency: Latency between silos in ms
    
    Returns:
        Dictionary of results
    """
    from cross_silo_server import CrossSiloCoordinator
    
    print("="*60)
    print("Running Cross-Silo Federated Learning")
    print("="*60)
    print(f"Silos: {num_silos}, Clients per silo: {clients_per_silo}")
    print(f"Rounds: {num_rounds}, Internal rounds: {internal_rounds}")
    print(f"Inter-silo bandwidth: {inter_bandwidth} Mbps, Latency: {inter_latency} ms")
    print("="*60)
    
    start_time = time.time()
    
    coordinator = CrossSiloCoordinator(
        num_silos=num_silos,
        clients_per_silo=clients_per_silo
    )
    
    coordinator.run_training(num_rounds=num_rounds, internal_rounds=internal_rounds)
    
    total_time = time.time() - start_time
    results = coordinator.get_results()
    results['total_time'] = total_time
    results['config'] = {
        'num_silos': num_silos,
        'clients_per_silo': clients_per_silo,
        'num_rounds': num_rounds,
        'internal_rounds': internal_rounds,
        'inter_bandwidth': inter_bandwidth,
        'inter_latency': inter_latency
    }
    
    return results


def run_single_vs_two_level_comparison(num_rounds: int = 20) -> dict:
    """
    Compare single-level and two-level federated learning.
    
    Args:
        num_rounds: Number of training rounds
    
    Returns:
        Dictionary of comparison results
    """
    from cross_silo_server import CrossSiloCoordinator
    from cloud_server import CloudEdgeCoordinator
    
    print("\n" + "="*60)
    print("SINGLE-LEVEL VS TWO-LEVEL COMPARISON")
    print("="*60)
    
    # Two-level (cross-silo)
    print("\n--- Two-Level (Cross-Silo) ---")
    coordinator_2level = CrossSiloCoordinator(num_silos=3, clients_per_silo=10)
    coordinator_2level.run_training(num_rounds=num_rounds, internal_rounds=5)
    results_2level = coordinator_2level.get_results()
    
    # Single-level (all clients together)
    print("\n--- Single-Level (All Clients) ---")
    coordinator_1level = CloudEdgeCoordinator(num_devices=30, model_shape=(10, 784))
    coordinator_1level.run_training(num_rounds=num_rounds, devices_per_round=30, num_epochs=1)
    results_1level = coordinator_1level.get_results()
    
    # Calculate comparison metrics
    comparison = {
        'two_level': {
            'final_accuracy': results_2level['training_stats']['accuracies'][-1],
            'total_comm_cost': results_2level['server_stats']['total_inter_comm_cost'],
            'total_time': results_2level['total_time']
        },
        'single_level': {
            'final_accuracy': results_1level['training_stats']['accuracies'][-1],
            'total_comm_cost': results_1level['server_stats']['total_comm_cost'],
            'total_time': results_1level['total_time']
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\n1. FINAL ACCURACY")
    print(f"   Two-Level: {comparison['two_level']['final_accuracy']:.4f}")
    print(f"   Single-Level: {comparison['single_level']['final_accuracy']:.4f}")
    
    print("\n2. COMMUNICATION COST")
    print(f"   Two-Level: {comparison['two_level']['total_comm_cost']:.2f} MB")
    print(f"   Single-Level: {comparison['single_level']['total_comm_cost']:.2f} MB")
    
    print("\n3. TRAINING TIME")
    print(f"   Two-Level: {comparison['two_level']['total_time']:.2f}s")
    print(f"   Single-Level: {comparison['single_level']['total_time']:.2f}s")
    
    return comparison


def run_silo_scaling_experiment():
    """Run experiment to test scaling with different silo counts."""
    print("\n" + "="*60)
    print("SILO SCALING EXPERIMENT")
    print("="*60)
    
    results = {}
    
    for num_silos in [2, 3, 5, 10]:
        print(f"\n--- Testing with {num_silos} silos ---")
        
        coordinator = CrossSiloCoordinator(num_silos=num_silos, clients_per_silo=10)
        coordinator.run_training(num_rounds=10, internal_rounds=3)
        
        res = coordinator.get_results()
        results[num_silos] = {
            'final_accuracy': res['training_stats']['accuracies'][-1],
            'total_comm_cost': res['server_stats']['total_inter_comm_cost'],
            'total_time': res['total_time']
        }
        
        print(f"Final accuracy: {results[num_silos]['final_accuracy']:.4f}")
        print(f"Communication cost: {results[num_silos]['total_comm_cost']:.2f} MB")
    
    print("\n" + "="*60)
    print("SCALING SUMMARY")
    print("="*60)
    print("\nSilos | Final Accuracy | Comm Cost (MB) | Time (s)")
    print("-" * 50)
    for num_silos, res in results.items():
        print(f"  {num_silos}  |    {res['final_accuracy']:.4f}    |    {res['total_comm_cost']:.2f}     |  {res['total_time']:.2f}")
    
    return results


def plot_results(results, output_dir: str = 'figures'):
    """Generate result plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        if 'training_stats' in results and 'accuracies' in results['training_stats']:
            rounds = np.arange(1, len(results['training_stats']['accuracies']) + 1)
            accuracies = results['training_stats']['accuracies']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accuracy curve
            ax1.plot(rounds, accuracies, 'b-', linewidth=2)
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Cross-Silo Training Accuracy')
            ax1.grid(True)
            
            # Communication cost
            if 'inter_comm_costs' in results['training_stats']:
                ax2.plot(rounds, results['training_stats']['inter_comm_costs'], 'r-', linewidth=2)
                ax2.set_xlabel('Round')
                ax2.set_ylabel('Communication Cost (MB)')
                ax2.set_title('Cumulative Inter-Silo Communication')
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cross_silo_results.png'))
            plt.close()
            
            print(f"\nPlot saved to {os.path.join(output_dir, 'cross_silo_results.png')}")
    
    except ImportError:
        print("\nMatplotlib not installed, skipping plot generation.")


def save_results(results, output_dir: str = 'results'):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    results = convert_to_native(results)
    
    path = os.path.join(output_dir, 'cross_silo_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {path}")


def main():
    """Main function to run cross-silo experiments."""
    print("=== Cross-Silo Federated Learning Experiment ===")
    
    # Run basic experiment
    results = run_cross_silo_experiment(
        num_silos=3,
        clients_per_silo=10,
        num_rounds=10,
        internal_rounds=3
    )
    
    # Save and plot results
    save_results(results)
    plot_results(results)
    
    # Run comparison
    print("\n")
    comparison = run_single_vs_two_level_comparison(num_rounds=10)
    
    # Run scaling experiment
    print("\n")
    scaling_results = run_silo_scaling_experiment()
    
    print("\n=== All Experiments Complete ===")


if __name__ == "__main__":
    main()
