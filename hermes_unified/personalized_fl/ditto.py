"""
Ditto: Personalized Federated Learning with Proximal Term

Ditto optimizes a local loss plus a proximal term (distance to global model).
The hyperparameter λ controls the degree of personalization:
- λ=0: pure local training
- λ=∞: global model (FedAvg behavior)

Reference: Li et al., "Ditto: Fair and Robust Federated Learning with Personalization"
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DittoClient:
    """
    Ditto client that trains a personalized model with proximal term.
    
    The client optimizes:
        min_w: L_local(w) + (λ/2) * ||w - w_global||^2
    
    where L_local is the standard classification loss on local data.
    """
    
    def __init__(self, client_id: int, train_data: Tuple[torch.Tensor, torch.Tensor],
                 test_data: Tuple[torch.Tensor, torch.Tensor], num_classes: int = 10,
                 lambda_: float = 1.0, device: str = 'cpu'):
        self.client_id = client_id
        self.lambda_ = lambda_
        self.device = device
        
        # Local model
        self.model = SimpleCNN(num_classes=num_classes).to(device)
        
        # Data loaders
        train_dataset = TensorDataset(train_data[0], train_data[1])
        test_dataset = TensorDataset(test_data[0], test_data[1])
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        # Statistics
        self.train_losses = []
        self.test_accuracies = []
        self.prox_losses = []
    
    def set_global_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """Set the global model (for computing proximal term)."""
        self.global_model = copy.deepcopy(global_state_dict)
    
    def train(self, local_epochs: int = 5) -> Dict[str, Any]:
        """
        Train locally with proximal term regularization.
        
        Args:
            local_epochs: Number of local training epochs
        
        Returns:
            Training statistics
        """
        self.model.train()
        
        epoch_losses = []
        epoch_prox = []
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_prox = 0.0
            num_batches = 0
            
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Standard classification loss
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Proximal term: ||w - w_global||^2
                prox_loss = 0.0
                for p, p_global in zip(self.model.parameters(), self.global_model.values()):
                    prox_loss += torch.sum((p - p_global) ** 2)
                
                prox_loss = (self.lambda_ / 2) * prox_loss
                
                # Total loss
                total_loss = loss + prox_loss
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_prox += prox_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_prox = epoch_prox / num_batches
            epoch_losses.append(avg_loss)
            epoch_prox.append(avg_prox)
        
        self.train_losses.extend(epoch_losses)
        self.prox_losses.extend(epoch_prox)
        
        # Evaluate on local test set
        accuracy = self.evaluate()
        
        return {
            'client_id': self.client_id,
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'final_prox': epoch_prox[-1] if epoch_prox else 0,
            'test_accuracy': accuracy
        }
    
    def evaluate(self) -> float:
        """Evaluate on local test data."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0
        self.test_accuracies.append(accuracy)
        return accuracy
    
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights as numpy arrays."""
        return {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights from numpy arrays."""
        state_dict = {k: torch.tensor(v) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)
    
    def get_personalized_model(self) -> nn.Module:
        """Get the personalized model."""
        return self.model
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'lambda': self.lambda_,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies,
            'final_accuracy': self.test_accuracies[-1] if self.test_accuracies else 0
        }


class DittoServer:
    """
    Ditto server that coordinates training across clients.
    
    Uses FedAvg to aggregate global model, while each client
    maintains its own personalized model.
    """
    
    def __init__(self, num_classes: int = 10, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device
        
        # Global model
        self.global_model = SimpleCNN(num_classes=num_classes).to(device)
        
        # Clients
        self.clients: List[DittoClient] = []
        
        # Statistics
        self.round_history = []
        self.global_round_history = []
    
    def add_client(self, client: DittoClient):
        """Add a client to the server."""
        self.clients.append(client)
    
    def broadcast_global_model(self):
        """Send global model to all clients."""
        global_weights = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        for client in self.clients:
            client.set_global_model(global_weights)
    
    def aggregate(self, client_updates: List[Dict[str, np.ndarray]],
                  client_weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using FedAvg.
        
        Args:
            client_updates: List of model weight dictionaries
            client_weights: Weight for each client (default: uniform)
        
        Returns:
            Aggregated model weights
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        aggregated = {}
        
        for key in client_updates[0].keys():
            tensors = [client_weights[i] * client_updates[i][key]
                      for i in range(len(client_updates))]
            aggregated[key] = np.sum(tensors, axis=0)
        
        return aggregated
    
    def run_federated_training(self, num_rounds: int = 10, clients_per_round: int = 10,
                              local_epochs: int = 5, lambda_: float = 1.0):
        """
        Run Ditto federated training.
        
        Args:
            num_rounds: Number of federated rounds
            clients_per_round: Number of clients to train per round
            local_epochs: Local training epochs per round
            lambda_: Proximal term weight
        """
        print(f"Starting Ditto training with {len(self.clients)} clients")
        print(f"Rounds: {num_rounds}, Clients per round: {clients_per_round}")
        print(f"Local epochs: {local_epochs}, Lambda: {lambda_}")
        print("=" * 60)
        
        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
            
            # Broadcast global model
            self.broadcast_global_model()
            
            # Select random clients
            selected_clients = np.random.choice(
                len(self.clients), size=min(clients_per_round, len(self.clients)),
                replace=False
            )
            
            # Local training
            client_results = []
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                client.lambda_ = lambda_
                result = client.train(local_epochs)
                client_results.append(result)
            
            # Aggregate global model
            updates = []
            weights = []
            for client_idx in selected_clients:
                updates.append(self.clients[client_idx].get_model_weights())
                weights.append(1.0)
            
            aggregated_weights = self.aggregate(updates, weights)
            self.global_model.load_state_dict({k: torch.tensor(v) for k, v in aggregated_weights.items()})
            
            # Evaluate global model on all clients
            global_accuracies = []
            for client in self.clients:
                client.set_global_model({k: v.clone() for k, v in self.global_model.state_dict().items()})
                # Train one step to get accurate eval (with global model)
                acc = client.evaluate()
                global_accuracies.append(acc)
            
            avg_global_acc = np.mean(global_accuracies)
            
            # Evaluate personalized models
            personalized_accuracies = [r['test_accuracy'] for r in client_results]
            avg_personalized_acc = np.mean(personalized_accuracies)
            
            self.global_round_history.append({
                'round': round_idx + 1,
                'global_accuracy': avg_global_acc,
                'personalized_accuracy': avg_personalized_acc,
                'selected_clients': len(selected_clients)
            })
            
            print(f"Global model accuracy: {avg_global_acc:.4f}")
            print(f"Personalized accuracy: {avg_personalized_acc:.4f}")
        
        print("\n=== Training Complete ===")
    
    def get_final_results(self) -> Dict[str, Any]:
        """Get final results for all clients."""
        results = {
            'global_model_history': self.global_round_history,
            'client_stats': [client.get_stats() for client in self.clients],
            'final_global_accuracy': self.global_round_history[-1]['global_accuracy'] if self.global_round_history else 0,
            'final_personalized_accuracy': self.global_round_history[-1]['personalized_accuracy'] if self.global_round_history else 0
        }
        return results


def run_ditto_experiment(num_clients: int = 20, num_rounds: int = 10):
    """Run a complete Ditto experiment."""
    from .utils_pfl import create_non_iid_cifar10
    
    print("=== Ditto Personalized Federated Learning Experiment ===\n")
    
    # Create Non-IID data
    data_dir = './data'
    client_data = create_non_iid_cifar10(num_clients=num_clients, alpha=0.5, data_dir=data_dir)
    
    # Create server
    server = DittoServer()
    
    # Create clients
    for i in range(num_clients):
        train_data, test_data = client_data[i]
        client = DittoClient(
            client_id=i,
            train_data=train_data,
            test_data=test_data,
            lambda_=1.0
        )
        server.add_client(client)
    
    # Run training
    server.run_federated_training(
        num_rounds=num_rounds,
        clients_per_round=10,
        local_epochs=5,
        lambda_=1.0
    )
    
    return server.get_final_results()
