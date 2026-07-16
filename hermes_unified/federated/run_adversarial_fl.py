"""
Adversarial Federated Learning Experiment Script

Runs a federated learning experiment with malicious clients and defense mechanisms.
"""

import numpy as np
import time
import threading
from typing import List, Dict, Tuple

from .federated_server import FederatedServer
from .federated_client import FederatedClient
from .federated_simulator import NonIIDDataGenerator


def generate_test_data(num_samples: int = 1000, num_features: int = 784, num_classes: int = 10):
    """Generate synthetic test data."""
    X = np.random.randn(num_samples, num_features)
    true_weights = np.random.randn(num_classes, num_features)
    logits = X @ true_weights.T
    y = np.argmax(logits, axis=1)
    return X, y


def evaluate_model(model: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """Evaluate model on test data."""
    pred = X_test @ model.T
    predictions = np.argmax(pred, axis=1)
    accuracy = np.mean(predictions == y_test)
    loss = np.mean((pred - np.eye(10)[y_test]) ** 2)
    return loss, accuracy


def run_adversarial_experiment(num_clients: int = 10, num_malicious: int = 2,
                                defense_type: str = None, attack_type: str = 'label_flip',
                                num_rounds: int = 20, local_epochs: int = 1):
    """
    Run an adversarial federated learning experiment.
    
    Args:
        num_clients: Total number of clients
        num_malicious: Number of malicious clients
        defense_type: Defense mechanism ('krum', 'trimmed_mean', or None)
        attack_type: Attack type ('label_flip', 'gradient_scale', 'backdoor')
        num_rounds: Number of federated rounds
        local_epochs: Number of local epochs per client
    
    Returns:
        Experiment results
    """
    print("===  Adversarial Federated Learning Experiment ===")
    print(f"Configuration:")
    print(f"  - Total clients: {num_clients}")
    print(f"  - Malicious clients: {num_malicious}")
    print(f"  - Defense: {defense_type if defense_type else 'None'}")
    print(f"  - Attack type: {attack_type}")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Local epochs: {local_epochs}")
    print()
    
    # Generate data
    num_features = 784
    num_classes = 10
    
    # Generate training data for clients
    X_train = np.random.randn(5000, num_features)
    y_train = np.argmax(X_train @ np.random.randn(num_classes, num_features).T, axis=1)
    
    # Generate test data
    X_test, y_test = generate_test_data(num_samples=1000)
    
    # Partition data for clients (Non-IID)
    data_generator = NonIIDDataGenerator(num_classes=num_classes)
    client_data = data_generator.generate_dirichlet_partition(X_train, y_train, num_clients, alpha=0.5)
    
    # Initialize server
    server = FederatedServer(host='127.0.0.1', port=5001, 
                            defense_type=defense_type, num_clients=num_clients)
    
    # Initialize global model
    global_model = np.random.randn(num_classes, num_features) * 0.01
    server.set_model(global_model)
    
    # Start server in separate thread
    server.start()
    time.sleep(1)  # Wait for server to start
    
    # Create clients
    clients: List[FederatedClient] = []
    malicious_indices = set(np.random.choice(range(num_clients), num_malicious, replace=False))
    
    for i in range(num_clients):
        is_malicious = i in malicious_indices
        client = FederatedClient(
            client_id=i,
            server_host='127.0.0.1',
            server_port=5001,
            is_malicious=is_malicious,
            attack_type=attack_type,
            attack_params={'intensity': 5.0}
        )
        
        # Set client data
        client.set_data(*client_data[i])
        
        # Connect client
        if client.connect():
            clients.append(client)
            if is_malicious:
                print(f"  Malicious client {i} connected with {attack_type} attack")
            else:
                print(f" Honest client {i} connected")
    
    print(f"\n {len(clients)} clients connected")
    
    # Run federated learning
    results = {
        'loss_history': [],
        'accuracy_history': [],
        'round_times': [],
        'attack_events': []
    }
    
    for round_idx in range(num_rounds):
        start_time = time.time()
        
        print(f"\n Round {round_idx + 1}/{num_rounds}")
        
        # Clients train locally
        updates = []
        for client in clients:
            update = client.train(local_epochs=local_epochs)
            updates.append(update)
            client.send_update(update)
        
        # Small delay for server to process
        time.sleep(0.1)
        
        # Evaluate
        loss, accuracy = evaluate_model(server.global_model, X_test, y_test)
        results['loss_history'].append(loss)
        results['accuracy_history'].append(accuracy)
        
        # Check for attack impact
        if round_idx > 0:
            prev_acc = results['accuracy_history'][-2]
            acc_drop = (prev_acc - accuracy) * 100
            if acc_drop > 5 and any(c.is_malicious for c in clients):
                results['attack_events'].append({
                    'round': round_idx,
                    'accuracy_drop': acc_drop
                })
                print(f" Attack detected! Accuracy dropped by {acc_drop:.2f}%")
        
        round_time = time.time() - start_time
        results['round_times'].append(round_time)
        
        print(f" Round {round_idx + 1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, Time={round_time:.2f}s")
    
    # Cleanup
    for client in clients:
        client.disconnect()
    
    server.stop()
    
    # Generate report
    print("\n" + "="*60)
    print("           EXPERIMENT REPORT          ")
    print("="*60)
    
    print("\n RESULTS")
    final_acc = results['accuracy_history'][-1]
    peak_acc = max(results['accuracy_history'])
    acc_drop = (peak_acc - final_acc) * 100
    
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Peak Accuracy: {peak_acc:.4f}")
    print(f"Total Accuracy Drop: {acc_drop:.2f}%")
    print(f"Number of Attack Events: {len(results['attack_events'])}")
    
    print("\n OUTCOME")
    if acc_drop < 10:
        print(" SUCCESS: Defense effectively mitigated attacks!")
    elif acc_drop < 30:
        print("  PARTIAL SUCCESS: Some accuracy loss but model usable")
    else:
        print(" FAILURE: Attacks overwhelmed the defense")
    
    print("="*60)
    
    return results


def compare_defenses():
    """Compare different defense mechanisms."""
    print("===  Defense Comparison Experiment ===")
    
    defenses = [None, 'krum', 'trimmed_mean']
    results = {}
    
    for defense in defenses:
        print(f"\n--- Testing defense: {defense if defense else 'None'} ---")
        result = run_adversarial_experiment(
            num_clients=10,
            num_malicious=3,
            defense_type=defense,
            attack_type='gradient_scale',
            num_rounds=15,
            local_epochs=2
        )
        results[defense] = result
    
    # Print comparison table
    print("\n" + "="*70)
    print("                    DEFENSE COMPARISON TABLE                    ")
    print("="*70)
    print(f"{'Defense':<15} {'Final Acc':<10} {'Peak Acc':<10} {'Acc Drop':<10} {'Attack Events':<15}")
    print("-"*70)
    
    for defense in defenses:
        res = results[defense]
        final_acc = res['accuracy_history'][-1]
        peak_acc = max(res['accuracy_history'])
        acc_drop = (peak_acc - final_acc) * 100
        attack_events = len(res['attack_events'])
        
        print(f"{str(defense):<15} {final_acc:<10.4f} {peak_acc:<10.4f} {acc_drop:<10.2f} {attack_events:<15}")
    
    print("="*70)


if __name__ == "__main__":
    # Run comparison experiment
    compare_defenses()
