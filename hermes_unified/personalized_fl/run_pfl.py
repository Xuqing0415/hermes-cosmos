"""
Personalized Federated Learning Experiment Runner

Runs and compares FedAvg, Ditto, and FedRep methods.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personalized_fl import (
    DittoClient, DittoServer,
    FedRepClient, FedRepServer,
    create_non_iid_cifar10,
    evaluate_personalized,
    compute_fairness_metrics,
    print_comparison_table
)


class FedAvgServer:
    """Standard FedAvg server for comparison."""
    
    def __init__(self, num_classes: int = 10, device: str = 'cpu'):
        from .ditto import SimpleCNN
        self.num_classes = num_classes
        self.device = device
        self.global_model = SimpleCNN(num_classes=num_classes).to(device)
        self.clients = []
        self.round_history = []
    
    def add_client(self, client):
        self.clients.append(client)
    
    def aggregate(self, client_weights, client_sizes):
        total_size = sum(client_sizes)
        weights = [s / total_size for s in client_sizes]
        
        aggregated = {}
        for key in client_weights[0].keys():
            aggregated[key] = np.sum([weights[i] * client_weights[i][key] 
                                     for i in range(len(client_weights))], axis=0)
        return aggregated
    
    def run_federated_training(self, num_rounds: int = 10, clients_per_round: int = 10,
                               local_epochs: int = 5):
        print(f"Starting FedAvg training with {len(self.clients)} clients")
        
        for round_idx in range(num_rounds):
            global_weights = {k: v.clone() for k, v in self.global_model.state_dict().items()}
            
            selected = np.random.choice(len(self.clients), size=min(clients_per_round, len(self.clients)), replace=False)
            
            updates = []
            sizes = []
            
            for idx in selected:
                client = self.clients[idx]
                client.model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})
                
                from .ditto import SimpleCNN
                client.model = SimpleCNN(num_classes=self.num_classes).to(self.device)
                client.model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})
                
                optimizer = torch.optim.SGD(client.model.parameters(), lr=0.01, momentum=0.9)
                criterion = torch.nn.CrossEntropyLoss()
                
                client.model.train()
                for epoch in range(local_epochs):
                    for data, target in client.train_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        output = client.model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                
                updates.append({k: v.detach().cpu().numpy() for k, v in client.model.state_dict().items()})
                sizes.append(len(client.train_loader.dataset))
            
            aggregated = self.aggregate(updates, sizes)
            self.global_model.load_state_dict({k: torch.tensor(v) for k, v in aggregated.items()})
            
            accuracies = []
            for client in self.clients:
                client.model.load_state_dict({k: v.clone() for k, v in self.global_model.state_dict().items()})
                acc = client.evaluate()
                accuracies.append(acc)
            
            self.round_history.append({
                'round': round_idx + 1,
                'accuracy': np.mean(accuracies)
            })
            
            print(f"Round {round_idx + 1}: Accuracy = {np.mean(accuracies):.4f}")
        
        return self.round_history


def run_experiment(method: str = 'all', num_clients: int = 20, num_rounds: int = 10,
                   alpha: float = 0.5, data_dir: str = './data'):
    """
    Run personalized FL experiment.
    
    Args:
        method: 'all', 'fedavg', 'ditto', or 'fedrep'
        num_clients: Number of clients
        num_rounds: Number of federated rounds
        alpha: Non-IID parameter (lower = more heterogeneous)
        data_dir: Directory for CIFAR-10 data
    """
    print("=" * 70)
    print("PERSONALIZED FEDERATED LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Non-IID Alpha: {alpha}")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    print("Generating Non-IID data split...")
    client_data = create_non_iid_cifar10(num_clients=num_clients, alpha=alpha, data_dir=data_dir)
    
    results = {}
    
    if method in ['all', 'fedavg']:
        print("\n" + "=" * 70)
        print("FEDAVG (Standard)")
        print("=" * 70)
        
        from .ditto import SimpleCNN
        
        server = FedAvgServer(num_classes=10, device=device)
        
        for i in range(num_clients):
            train_data, test_data = client_data[i]
            train_X, train_y = train_data
            test_X, test_y = test_data
            
            model = SimpleCNN().to(device)
            
            from torch.utils.data import DataLoader, TensorDataset
            train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
            test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=64, shuffle=False)
            
            client = type('obj', (object,), {
                'model': model,
                'train_loader': train_loader,
                'test_loader': test_loader,
                'evaluate': lambda: evaluate_model(model, test_loader, device)
            })()
            server.add_client(client)
        
        history = server.run_federated_training(num_rounds=num_rounds, clients_per_round=10, local_epochs=5)
        results['FedAvg'] = {'round_history': history, 'final_accuracy': history[-1]['accuracy'] if history else 0}
    
    if method in ['all', 'ditto']:
        print("\n" + "=" * 70)
        print("DITTO (Personalized with Proximal Term)")
        print("=" * 70)
        
        server = DittoServer(num_classes=10, device=device)
        
        for i in range(num_clients):
            train_data, test_data = client_data[i]
            client = DittoClient(i, train_data, test_data, lambda_=1.0, device=device)
            server.add_client(client)
        
        server.run_federated_training(num_rounds=num_rounds, clients_per_round=10, local_epochs=5, lambda_=1.0)
        results['Ditto'] = server.get_final_results()
    
    if method in ['all', 'fedrep']:
        print("\n" + "=" * 70)
        print("FEDREP (Representation Learning)")
        print("=" * 70)
        
        server = FedRepServer(num_classes=10, device=device)
        
        for i in range(num_clients):
            train_data, test_data = client_data[i]
            client = FedRepClient(i, train_data, test_data, device=device)
            server.add_client(client)
        
        server.run_federated_training(num_rounds=num_rounds, clients_per_round=10, head_epochs=1, rep_epochs=1)
        results['FedRep'] = server.get_final_results()
    
    return results


def evaluate_model(model, test_loader, device):
    """Evaluate a model on test data."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0


def main():
    """Main function to run experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Personalized Federated Learning')
    parser.add_argument('--method', type=str, default='all', choices=['all', 'fedavg', 'ditto', 'fedrep'])
    parser.add_argument('--clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=10, help='Number of rounds')
    parser.add_argument('--alpha', type=float, default=0.5, help='Non-IID parameter (lower=more heterogeneous)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    results = run_experiment(
        method=args.method,
        num_clients=args.clients,
        num_rounds=args.rounds,
        alpha=args.alpha,
        data_dir=args.data_dir
    )
    
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        for method, res in results.items():
            print(f"\n{method}:")
            print(f"  Final Accuracy: {res.get('final_accuracy', res.get('round_history', [{}])[-1].get('accuracy', 0)):.4f}")
        
        best_method = max(results.items(), key=lambda x: x[1].get('final_accuracy', 0))
        print(f"\nBest Method: {best_method[0]} with accuracy {best_method[1].get('final_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
