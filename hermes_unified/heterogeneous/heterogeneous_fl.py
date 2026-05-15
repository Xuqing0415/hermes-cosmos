"""
Heterogeneous Model Federated Learning Module

Implements federated learning where clients can have different model architectures.
Uses knowledge distillation and feature alignment to enable knowledge sharing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeacherModel(nn.Module):
    """
    Large teacher model that serves as the global model.
    Uses ResNet-18 architecture for better performance.
    """
    
    def __init__(self, num_classes: int = 10):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # For feature distillation
        self.feature_maps = {}
    
    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        self.feature_maps['conv2'] = x
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        self.feature_maps['conv4'] = x
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        self.feature_maps['conv6'] = x
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        
        if return_features:
            return logits, self.feature_maps
        return logits
    
    def get_features(self, layer_name: str):
        return self.feature_maps.get(layer_name, None)


class StudentCNN(nn.Module):
    """
    Small CNN student model for resource-constrained devices.
    """
    
    def __init__(self, num_classes: int = 10):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.feature_maps = {}
    
    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        self.feature_maps['conv1'] = x
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        self.feature_maps['conv3'] = x
        
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        
        if return_features:
            return logits, self.feature_maps
        return logits


class StudentMobileNet(nn.Module):
    """
    MobileNet-like lightweight model.
    """
    
    def __init__(self, num_classes: int = 10):
        super(StudentMobileNet, self).__init__()
        
        def depthwise_conv(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.ReLU()
            )
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.dw1 = depthwise_conv(32, 64)
        self.dw2 = depthwise_conv(64, 128, stride=2)
        self.dw3 = depthwise_conv(128, 128)
        self.dw4 = depthwise_conv(128, 256, stride=2)
        self.dw5 = depthwise_conv(256, 256)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)
        
        self.feature_maps = {}
    
    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x = self.dw4(x)
        x = self.dw5(x)
        
        self.feature_maps['final'] = x
        
        x = x.view(-1, 256 * 4 * 4)
        logits = self.fc(x)
        
        if return_features:
            return logits, self.feature_maps
        return logits


class StudentMLP(nn.Module):
    """
    Simple MLP model for extremely resource-constrained devices.
    """
    
    def __init__(self, num_classes: int = 10, input_dim: int = 32 * 32 * 3):
        super(StudentMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class HeterogeneousClient:
    """
    Federated client with heterogeneous model architecture.
    """
    
    def __init__(self, client_id: str, model_type: str = "cnn",
                 num_classes: int = 10, local_data: Tuple = None):
        self.client_id = client_id
        self.model_type = model_type
        
        # Create appropriate model
        if model_type == "cnn":
            self.model = StudentCNN(num_classes=num_classes)
        elif model_type == "mobilenet":
            self.model = StudentMobileNet(num_classes=num_classes)
        elif model_type == "mlp":
            self.model = StudentMLP(num_classes=num_classes)
        else:
            self.model = StudentCNN(num_classes=num_classes)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        
        # Local data
        if local_data is None:
            # Generate dummy data
            X = torch.randn(100, 3, 32, 32)
            y = torch.randint(0, num_classes, (100,))
            self.train_loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
        else:
            X, y = local_data
            self.train_loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
        
        # Statistics
        self.accuracies = []
        self.losses = []
    
    def train_local(self, teacher_logits: torch.Tensor = None,
                   teacher_features: Dict = None, distillation_weight: float = 0.5):
        """
        Train local model with optional knowledge distillation.
        
        Args:
            teacher_logits: Soft labels from teacher model
            teacher_features: Intermediate features from teacher for feature distillation
            distillation_weight: Weight for distillation loss vs CE loss
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            # Standard classification loss
            ce_loss = self.criterion_ce(output, target)
            
            # Knowledge distillation loss if teacher logits provided
            if teacher_logits is not None:
                soft_target = F.softmax(teacher_logits[batch_idx * 16:(batch_idx + 1) * 16] / 2.0, dim=1)
                soft_pred = F.log_softmax(output / 2.0, dim=1)
                dist_loss = F.kl_div(soft_pred, soft_target, reduction='batchmean') * (2.0 ** 2)
                loss = distillation_weight * dist_loss + (1 - distillation_weight) * ce_loss
            else:
                loss = ce_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        self.accuracies.append(accuracy)
        self.losses.append(avg_loss)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def get_logits(self, data: torch.Tensor) -> torch.Tensor:
        """Get logits from local model."""
        self.model.eval()
        with torch.no_grad():
            return self.model(data)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights as numpy arrays."""
        return {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'model_type': self.model_type,
            'accuracies': self.accuracies,
            'losses': self.losses,
            'avg_accuracy': np.mean(self.accuracies[-5:]) if self.accuracies else 0
        }


class HeterogeneousServer:
    """
    Server for heterogeneous federated learning with knowledge distillation.
    """
    
    def __init__(self, num_classes: int = 10):
        self.teacher_model = TeacherModel(num_classes=num_classes)
        self.optimizer = optim.Adam(self.teacher_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.clients: List[HeterogeneousClient] = []
        
        # Statistics
        self.round_history = []
        self.total_communication = 0.0
    
    def add_client(self, client: HeterogeneousClient):
        """Add a heterogeneous client."""
        self.clients.append(client)
    
    def distill_knowledge(self, data_loader: DataLoader):
        """
        Distill knowledge from teacher to clients.
        
        The teacher provides soft labels and intermediate features
        for clients to learn from.
        """
        self.teacher_model.eval()
        
        all_teacher_logits = []
        all_features = {}
        
        for data, _ in data_loader:
            with torch.no_grad():
                logits, features = self.teacher_model(data, return_features=True)
                all_teacher_logits.append(logits)
            
            for layer_name, feature in features.items():
                if layer_name not in all_features:
                    all_features[layer_name] = []
                all_features[layer_name].append(feature)
        
        return torch.cat(all_teacher_logits), all_features
    
    def update_teacher(self, client_logits: List[torch.Tensor], labels: torch.Tensor):
        """
        Update teacher model using client predictions.
        
        The teacher learns from client predictions, effectively
        aggregating knowledge from heterogeneous models.
        """
        self.teacher_model.train()
        self.optimizer.zero_grad()
        
        # Get teacher predictions
        teacher_logits = self.teacher_model(labels.view(-1, 3, 32, 32))
        
        # Knowledge transfer loss - teacher should match client consensus
        client_avg_logits = torch.stack(client_logits).mean(dim=0)
        dist_loss = F.kl_div(
            F.log_softmax(teacher_logits / 2.0, dim=1),
            F.softmax(client_avg_logits / 2.0, dim=1),
            reduction='batchmean'
        ) * (2.0 ** 2)
        
        # Also apply standard CE loss
        ce_loss = self.criterion(teacher_logits, labels)
        
        loss = 0.5 * dist_loss + 0.5 * ce_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_teacher(self, test_loader: DataLoader) -> float:
        """Evaluate teacher model on test data."""
        self.teacher_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.teacher_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def run_federated_training(self, num_rounds: int = 10, local_epochs: int = 1,
                              distillation_weight: float = 0.5):
        """
        Run heterogeneous federated learning with knowledge distillation.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local training epochs per round
            distillation_weight: Weight for distillation loss
        """
        print(f"Starting heterogeneous federated training with {len(self.clients)} clients")
        print(f"Client types: {[c.model_type for c in self.clients]}")
        
        # Create dummy test data
        test_data = torch.randn(100, 3, 32, 32)
        test_labels = torch.randint(0, 10, (100,))
        test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=16)
        
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            # Distill knowledge from teacher to clients
            print("Distilling knowledge from teacher...")
            client_data = torch.randn(200, 3, 32, 32)  # Shared data for distillation
            teacher_logits, _ = self.distill_knowledge(
                DataLoader(TensorDataset(client_data, torch.zeros(200)), batch_size=16)
            )
            
            # Local training with distillation
            print("Local training with knowledge distillation...")
            client_logits = []
            
            for client in self.clients:
                for _ in range(local_epochs):
                    stats = client.train_local(
                        teacher_logits=teacher_logits,
                        distillation_weight=distillation_weight
                    )
                
                # Get client predictions for teacher update
                with torch.no_grad():
                    client.model.eval()
                    client_logits.append(client.model(client_data))
            
            # Update teacher using client knowledge
            print("Updating teacher model...")
            self.update_teacher(client_logits, test_labels[:200])
            
            # Evaluate
            accuracy = self.evaluate_teacher(test_loader)
            
            self.round_history.append({
                'round': round_idx + 1,
                'teacher_accuracy': accuracy,
                'client_stats': [client.get_stats() for client in self.clients]
            })
            
            print(f"Teacher accuracy: {accuracy:.4f}")
            for client in self.clients:
                stats = client.get_stats()
                print(f"  {client.client_id} ({client.model_type}): {stats['avg_accuracy']:.4f}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return {
            'round_history': self.round_history,
            'total_communication_mb': self.total_communication / (1024 * 1024),
            'client_stats': [client.get_stats() for client in self.clients]
        }


def create_heterogeneous_cluster(num_clients: int = 5) -> HeterogeneousServer:
    """Create a server with heterogeneous clients."""
    server = HeterogeneousServer()
    
    model_types = ["cnn", "mobilenet", "mlp"]
    
    for i in range(num_clients):
        model_type = model_types[i % len(model_types)]
        client = HeterogeneousClient(f"client_{i}", model_type=model_type)
        server.add_client(client)
    
    return server


if __name__ == "__main__":
    print("=== Testing Heterogeneous Federated Learning ===")
    
    server = create_heterogeneous_cluster(5)
    server.run_federated_training(num_rounds=5, local_epochs=1)
    
    results = server.get_results()
    print("\n=== Final Results ===")
    print(f"Teacher accuracy: {results['round_history'][-1]['teacher_accuracy']:.4f}")
