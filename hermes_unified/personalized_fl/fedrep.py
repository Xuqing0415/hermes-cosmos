"""
FedRep: Federated Representation Learning

FedRep learns a globally shared representation (feature extractor) while
each client trains a local prediction head (last few layers).

Key idea:
- Shared representation: trained via aggregation across all clients
- Local head: trained locally by each client
- Alternating optimization between representation and head

Reference: Collins et al., "FedRep: Towards Universal Federated Learning via Representation Learning"
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedRepRepresentation(nn.Module):
    """
    Shared representation (feature extractor) network.
    
    This is trained globally via aggregation across all clients.
    """
    
    def __init__(self, feature_dim: int = 128):
        super(FedRepRepresentation, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, feature_dim)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        return x


class FedRepHead(nn.Module):
    """
    Local prediction head (classifier) for each client.
    
    This is trained locally by each client.
    """
    
    def __init__(self, feature_dim: int = 128, num_classes: int = 10):
        super(FedRepHead, self).__init__()
        self.fc2 = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.fc2(x)


class FedRepClient:
    """
    FedRep client with representation and local head.
    
    Each client has:
    - A shared representation (initialized from global, updated locally)
    - A local head (trained locally, aggregated server-side)
    """
    
    def __init__(self, client_id: int, train_data: Tuple[torch.Tensor, torch.Tensor],
                 test_data: Tuple[torch.Tensor, torch.Tensor], num_classes: int = 10,
                 feature_dim: int = 128, device: str = 'cpu'):
        self.client_id = client_id
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        
        # Representation and head networks
        self.representation = FedRepRepresentation(feature_dim).to(device)
        self.head = FedRepHead(feature_dim, num_classes).to(device)
        
        # Data loaders
        train_dataset = TensorDataset(train_data[0], train_data[1])
        test_dataset = TensorDataset(test_data[0], test_data[1])
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Optimizers
        self.rep_optimizer = optim.SGD(self.representation.parameters(), lr=0.01, momentum=0.9)
        self.head_optimizer = optim.SGD(self.head.parameters(), lr=0.1, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        # Statistics
        self.head_train_losses = []
        self.rep_train_losses = []
        self.test_accuracies = []
    
    def set_global_representation(self, rep_state_dict: Dict[str, torch.Tensor]):
        """Set the global representation from server."""
        self.representation.load_state_dict(rep_state_dict)
    
    def get_representation_weights(self) -> Dict[str, np.ndarray]:
        """Get representation weights as numpy arrays."""
        return {k: v.detach().cpu().numpy() for k, v in self.representation.state_dict().items()}
    
    def get_head_weights(self) -> Dict[str, np.ndarray]:
        """Get head weights as numpy arrays."""
        return {k: v.detach().cpu().numpy() for k, v in self.head.state_dict().items()}
    
    def set_head_weights(self, weights: Dict[str, np.ndarray]):
        """Set head weights from server."""
        self.head.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
    
    def train_head(self, num_epochs: int = 1) -> float:
        """
        Train local head while keeping representation fixed.
        
        Args:
            num_epochs: Number of epochs to train head
        
        Returns:
            Average training loss
        """
        self.representation.eval()  # Freeze representation
        self.head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.head_optimizer.zero_grad()
                
                # Get features using fixed representation
                with torch.no_grad():
                    features = self.representation(data)
                
                # Train head
                output = self.head(features)
                loss = self.criterion(output, target)
                loss.backward()
                self.head_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.head_train_losses.append(avg_loss)
        return avg_loss
    
    def train_representation(self, num_epochs: int = 1) -> float:
        """
        Train representation while keeping head fixed.
        
        Args:
            num_epochs: Number of epochs to train representation
        
        Returns:
            Average training loss
        """
        self.representation.train()
        self.head.eval()  # Freeze head
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.rep_optimizer.zero_grad()
                
                # Get features and predictions
                features = self.representation(data)
                output = self.head(features)
                loss = self.criterion(output, target)
                
                loss.backward()
                self.rep_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.rep_train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self) -> float:
        """Evaluate on local test data using full model (representation + head)."""
        self.representation.eval()
        self.head.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                features = self.representation(data)
                output = self.head(features)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0
        self.test_accuracies.append(accuracy)
        return accuracy
    
    def get_full_model(self) -> nn.Module:
        """Get full model (representation + head combined)."""
        class FullModel(nn.Module):
            def __init__(self, rep, head):
                super().__init__()
                self.representation = rep
                self.head = head
            
            def forward(self, x):
                features = self.representation(x)
                return self.head(features)
        
        return FullModel(self.representation, self.head)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'head_losses': self.head_train_losses,
            'rep_losses': self.rep_train_losses,
            'test_accuracies': self.test_accuracies,
            'final_accuracy': self.test_accuracies[-1] if self.test_accuracies else 0
        }


class FedRepServer:
    """
    FedRep server that coordinates representation learning across clients.
    
    Alternates between:
    1. Client-side head training (representation fixed)
    2. Client-side representation training (head fixed)
    3. Server-side aggregation of heads and representations
    """
    
    def __init__(self, num_classes: int = 10, feature_dim: int = 128, device: str = 'cpu'):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        
        # Global representation and head
        self.global_representation = FedRepRepresentation(feature_dim).to(device)
        self.global_head = FedRepHead(feature_dim, num_classes).to(device)
        
        # Clients
        self.clients: List[FedRepClient] = []
        
        # Statistics
        self.round_history = []
    
    def add_client(self, client: FedRepClient):
        """Add a client to the server."""
        self.clients.append(client)
    
    def broadcast_representation(self):
        """Send global representation to all clients."""
        rep_weights = {k: v.clone() for k, v in self.global_representation.state_dict().items()}
        for client in self.clients:
            client.set_global_representation(rep_weights)
    
    def aggregate_representations(self, client_reps: List[Dict[str, np.ndarray]],
                                  weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate client representations using FedAvg.
        """
        if weights is None:
            weights = [1.0 / len(client_reps)] * len(client_reps)
        
        aggregated = {}
        
        for key in client_reps[0].keys():
            tensors = [weights[i] * client_reps[i][key] for i in range(len(client_reps))]
            aggregated[key] = np.sum(tensors, axis=0)
        
        return aggregated
    
    def aggregate_heads(self, client_heads: List[Dict[str, np.ndarray]],
                       weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate client heads using FedAvg.
        """
        if weights is None:
            weights = [1.0 / len(client_heads)] * len(client_heads)
        
        aggregated = {}
        
        for key in client_heads[0].keys():
            tensors = [weights[i] * client_heads[i][key] for i in range(len(client_heads))]
            aggregated[key] = np.sum(tensors, axis=0)
        
        return aggregated
    
    def run_federated_training(self, num_rounds: int = 10, clients_per_round: int = 10,
                              head_epochs: int = 1, rep_epochs: int = 1):
        """
        Run FedRep federated training.
        
        Alternating optimization:
        1. Broadcast global representation to all clients
        2. Each client trains local head (representation fixed)
        3. Aggregate client heads to update global head
        4. Each client trains representation (head fixed)
        5. Aggregate client representations to update global representation
        
        Args:
            num_rounds: Number of federated rounds
            clients_per_round: Number of clients to train per round
            head_epochs: Epochs for local head training
            rep_epochs: Epochs for local representation training
        """
        print(f"Starting FedRep training with {len(self.clients)} clients")
        print(f"Rounds: {num_rounds}, Clients per round: {clients_per_round}")
        print(f"Head epochs: {head_epochs}, Representation epochs: {rep_epochs}")
        print("=" * 60)
        
        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
            
            # Select random clients
            selected_clients = np.random.choice(
                len(self.clients), size=min(clients_per_round, len(self.clients)),
                replace=False
            )
            
            # Phase 1: Train local heads (representation fixed)
            print("Phase 1: Training local heads...")
            self.broadcast_representation()
            
            head_updates = []
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                client.train_head(num_epochs=head_epochs)
                head_updates.append(client.get_head_weights())
            
            # Aggregate heads
            aggregated_head = self.aggregate_heads(head_updates)
            self.global_head.load_state_dict({k: torch.tensor(v) for k, v in aggregated_head.items()})
            
            # Phase 2: Train representations (head fixed)
            print("Phase 2: Training representations...")
            
            # Broadcast global head to all clients
            head_weights = {k: v.clone() for k, v in self.global_head.state_dict().items()}
            for client in self.clients:
                client.set_head_weights(head_weights)
            
            rep_updates = []
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                client.train_representation(num_epochs=rep_epochs)
                rep_updates.append(client.get_representation_weights())
            
            # Aggregate representations
            aggregated_rep = self.aggregate_representations(rep_updates)
            self.global_representation.load_state_dict({k: torch.tensor(v) for k, v in aggregated_rep.items()})
            
            # Evaluate
            accuracies = []
            for client in self.clients:
                client.set_global_representation({k: v.clone() for k, v in self.global_representation.state_dict().items()})
                client.set_head_weights({k: v.clone() for k, v in self.global_head.state_dict().items()})
                acc = client.evaluate()
                accuracies.append(acc)
            
            avg_accuracy = np.mean(accuracies)
            
            self.round_history.append({
                'round': round_idx + 1,
                'accuracy': avg_accuracy,
                'selected_clients': len(selected_clients)
            })
            
            print(f"Average accuracy: {avg_accuracy:.4f}")
        
        print("\n=== Training Complete ===")
    
    def get_final_results(self) -> Dict[str, Any]:
        """Get final results for all clients."""
        return {
            'round_history': self.round_history,
            'client_stats': [client.get_stats() for client in self.clients],
            'final_accuracy': self.round_history[-1]['accuracy'] if self.round_history else 0
        }


def run_fedrep_experiment(num_clients: int = 20, num_rounds: int = 10):
    """Run a complete FedRep experiment."""
    from .utils_pfl import create_non_iid_cifar10
    
    print("=== FedRep Personalized Federated Learning Experiment ===\n")
    
    # Create Non-IID data
    data_dir = './data'
    client_data = create_non_iid_cifar10(num_clients=num_clients, alpha=0.5, data_dir=data_dir)
    
    # Create server
    server = FedRepServer()
    
    # Create clients
    for i in range(num_clients):
        train_data, test_data = client_data[i]
        client = FedRepClient(
            client_id=i,
            train_data=train_data,
            test_data=test_data
        )
        server.add_client(client)
    
    # Run training
    server.run_federated_training(
        num_rounds=num_rounds,
        clients_per_round=10,
        head_epochs=1,
        rep_epochs=1
    )
    
    return server.get_final_results()
