"""
FedXAI Experiment Runner

Runs federated learning with XAI explanation generation.
"""

import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xai import FederatedXAIClient, FederatedXAIServer
from xai.dashboard import run_xai_dashboard


class SimpleMLP:
    """Simple MLP model for demonstration."""
    
    def __init__(self, n_features, n_classes=2):
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.random.randn(n_features, n_classes) * 0.1
    
    def predict(self, X):
        """Simple linear prediction."""
        logits = X @ self.weights
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def run_xai_experiment(num_clients: int = 5, num_rounds: int = 10,
                      n_features: int = 20, anomaly_injection: bool = False):
    """
    Run federated learning with XAI experiment.
    
    Args:
        num_clients: Number of federated clients
        num_rounds: Number of training rounds
        n_features: Number of features
        anomaly_injection: Whether to inject anomalous client
    """
    print("=" * 60)
    print("FedXAI Experiment")
    print("=" * 60)
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Features: {n_features}")
    print(f"Anomaly injection: {anomaly_injection}")
    print("=" * 60)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create XAI server
    xai_server = FederatedXAIServer(feature_names=feature_names)
    
    # Create clients
    clients = []
    
    for i in range(num_clients):
        # Create dummy model
        model = SimpleMLP(n_features=n_features)
        
        # Create client
        client = FederatedXAIClient(
            client_id=f"client_{i}",
            model=model,
            feature_names=feature_names
        )
        
        clients.append(client)
        xai_server.register_client(client)
    
    # Set baseline for anomaly detection
    baseline_importance = np.random.rand(n_features)
    baseline_importance /= baseline_importance.sum()
    
    for client in clients:
        client.set_baseline(baseline_importance)
    
    print(f"\nRegistered {len(clients)} clients with XAI server")
    
    # Inject one anomalous client if requested
    if anomaly_injection and num_clients > 1:
        # Client 0 will have very different importance pattern
        clients[0].set_baseline(baseline_importance * 0.1)
        clients[0]._baseline_importance = baseline_importance * 0.1
        print("Injected anomalous client (client_0)")
    
    # Run federated learning rounds
    print("\n--- Running Federated Learning with XAI ---")
    
    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
        
        # Simulate local training and explanation
        for i, client in enumerate(clients):
            # Generate synthetic test data
            X_local = np.random.randn(100, n_features)
            
            # Generate local explanation
            exp = client.generate_local_explanation(X_local, method='shap', n_samples=50)
        
        # Aggregate XAI
        result = xai_server.run_xai_round(detect_anomalies=True)
        
        print(f"XAI Round {result['round']}:")
        print(f"  Clients processed: {result['n_clients']}")
        
        if result['anomalies']:
            print(f"  Anomalies detected: {len(result['anomalies'])}")
            for alert in result['anomalies']:
                print(f"    - {alert['client_id']}: deviation={alert['deviation']:.4f}")
        else:
            print(f"  Anomalies detected: 0")
        
        # Update dashboard
        from xai.dashboard import dashboard
        dashboard_data = xai_server.generate_dashboard_data()
        dashboard.update(dashboard_data)
        
        time.sleep(0.5)
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    global_exp = xai_server.get_global_explanation()
    print(f"\nGlobal Feature Importance (Top 5):")
    for name, imp in global_exp['top_features'][:5]:
        print(f"  {name}: {imp:.4f}")
    
    print(f"\nTotal anomaly alerts: {global_exp['n_anomalies']}")
    print(f"Total clients: {global_exp['total_clients']}")
    
    # Client-specific analysis
    print("\n--- Client Importance Vectors ---")
    for client in clients[:3]:  # Show first 3
        importance = client.get_feature_importance_vector()
        deviation = client.compute_importance_deviation()
        print(f"{client.client_id}:")
        print(f"  Top feature: {feature_names[np.argmax(importance)]}")
        print(f"  Deviation: {deviation:.4f}")
    
    return xai_server


def start_dashboard_server(port: int = 8502):
    """Start the XAI dashboard in a background thread."""
    import threading
    thread = threading.Thread(target=run_xai_dashboard, kwargs={'port': port}, daemon=True)
    thread.start()
    print(f"XAI Dashboard started on http://localhost:{port}")
    return thread


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FedXAI Experiment')
    parser.add_argument('--clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=10, help='Number of rounds')
    parser.add_argument('--features', type=int, default=20, help='Number of features')
    parser.add_argument('--anomaly', action='store_true', help='Inject anomalous client')
    parser.add_argument('--dashboard-port', type=int, default=8502, help='Dashboard port')
    parser.add_argument('--no-dashboard', action='store_true', help='Skip dashboard')
    
    args = parser.parse_args()
    
    # Start dashboard if not disabled
    if not args.no_dashboard:
        start_dashboard_server(args.dashboard_port)
        time.sleep(2)  # Wait for dashboard to start
    
    # Run experiment
    server = run_xai_experiment(
        num_clients=args.clients,
        num_rounds=args.rounds,
        n_features=args.features,
        anomaly_injection=args.anomaly
    )
    
    print("\nExperiment complete!")
    print(f"Access dashboard at http://localhost:{args.dashboard_port}")
    
    if not args.no_dashboard:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
