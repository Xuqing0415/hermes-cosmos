"""
FedSimCLR Coordinator

Implements the complete federated SimCLR training pipeline.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Dict, List, Any, Optional
import numpy as np

from .contrastive_client import SimCLRClient, MoCoClient
from .global_buffer import PrivacyPreservingBuffer, FederatedBufferServer
from .momentum_encoder import FederatedMomentumEncoder


class ResNetEncoder(nn.Module):
    """Simple ResNet-style encoder for contrastive learning."""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 128):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class SimCLRAugmentation:
    """SimCLR data augmentation pipeline."""
    
    def __init__(self, image_size: int = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class FedSimCLRCoordinator:
    """Coordinator for Federated SimCLR training."""
    
    def __init__(self, num_clients: int = 5, num_rounds: int = 50, 
                 local_epochs: int = 5, device: str = 'cpu'):
        """
        Initialize FedSimCLR coordinator.
        
        Args:
            num_clients: Number of clients
            num_rounds: Number of federated rounds
            local_epochs: Number of local epochs per round
            device: Training device
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.device = device
        
        self.clients: List[SimCLRClient] = []
        self.buffer_server = FederatedBufferServer(
            max_size=65536,
            feature_dim=128,
            device=device,
            use_dp=True
        )
        
        self.global_encoder = None
        self.global_projection = None
        
        self.results = {
            'loss_history': [],
            'accuracy_history': [],
            'buffer_size_history': []
        }
    
    def setup(self):
        """Set up clients and server."""
        base_encoder = ResNetEncoder(input_channels=3, output_dim=128)
        self.global_encoder = base_encoder.to(self.device)
        
        self.global_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ).to(self.device)
        
        for i in range(self.num_clients):
            client_encoder = ResNetEncoder(input_channels=3, output_dim=128)
            client = SimCLRClient(
                client_id=i,
                encoder=client_encoder,
                projection_dim=128,
                device=self.device
            )
            self.clients.append(client)
    
    def sample_clients(self, fraction: float = 0.5) -> List[SimCLRClient]:
        """
        Sample a subset of clients for training.
        
        Args:
            fraction: Fraction of clients to sample
        
        Returns:
            List of sampled clients
        """
        num_samples = max(1, int(self.num_clients * fraction))
        indices = np.random.choice(self.num_clients, num_samples, replace=False)
        return [self.clients[i] for i in indices]
    
    def aggregate_parameters(self, updates: List[Dict[str, Dict[str, torch.Tensor]]]):
        """
        Aggregate client updates.
        
        Args:
            updates: List of parameter updates from clients
        """
        if not updates:
            return
        
        for key in ['encoder', 'projection']:
            if key not in updates[0]:
                continue
            
            for param_name in updates[0][key]:
                params = []
                for update in updates:
                    if key in update and param_name in update[key]:
                        params.append(update[key][param_name])
                
                if params:
                    avg_update = torch.mean(torch.stack(params), dim=0)
                    
                    if key == 'encoder':
                        self.global_encoder.state_dict()[param_name].add_(avg_update.to(self.device))
                    elif key == 'projection':
                        self.global_projection.state_dict()[param_name].add_(avg_update.to(self.device))
    
    def get_global_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get current global parameters."""
        return {
            'encoder': {k: v.detach().cpu().clone() for k, v in self.global_encoder.state_dict().items()},
            'projection': {k: v.detach().cpu().clone() for k, v in self.global_projection.state_dict().items()}
        }
    
    def run(self, dataloaders: List):
        """
        Run federated SimCLR training.
        
        Args:
            dataloaders: List of dataloaders for each client
        """
        print(f"Starting FedSimCLR with {self.num_clients} clients")
        
        for round_idx in range(self.num_rounds):
            print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")
            
            global_params = self.get_global_parameters()
            updates = []
            
            sampled_clients = self.sample_clients(fraction=0.8)
            
            for client in sampled_clients:
                dataloader = dataloaders[client.client_id]
                
                update = client.local_train(
                    dataloader,
                    global_params,
                    self.buffer_server.buffer,
                    num_epochs=self.local_epochs
                )
                updates.append(update)
                
                features = self._extract_features(client, dataloader)
                self.buffer_server.receive_features(client.client_id, features)
            
            self.aggregate_parameters(updates)
            
            buffer_info = self.buffer_server.get_buffer_info()
            self.results['buffer_size_history'].append(buffer_info['size'])
            
            avg_loss = np.mean([c.loss_history[-1] for c in sampled_clients])
            self.results['loss_history'].append(avg_loss)
            
            print(f"Round {round_idx + 1} - Avg Loss: {avg_loss:.4f}")
            print(f"Buffer Size: {buffer_info['size']}")
        
        return self.results
    
    def _extract_features(self, client: SimCLRClient, dataloader) -> torch.Tensor:
        """Extract features from client for global buffer."""
        client.local_encoder.eval()
        features = []
        
        with torch.no_grad():
            for views in dataloader:
                x = views[0].to(self.device)
                feat = client.local_encoder(x)
                features.append(feat)
        
        return torch.cat(features, dim=0)
    
    def linear_evaluation(self, train_dataloader, test_dataloader, num_epochs: int = 10):
        """
        Perform linear evaluation on learned representations.
        
        Args:
            train_dataloader: Training dataloader with labels
            test_dataloader: Test dataloader with labels
            num_epochs: Number of evaluation epochs
        
        Returns:
            Final test accuracy
        """
        self.global_encoder.eval()
        
        classifier = nn.Linear(128, 10).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            classifier.train()
            total_loss = 0.0
            
            for x, y in train_dataloader:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    features = self.global_encoder(x.to(self.device))
                
                logits = classifier(features)
                loss = F.cross_entropy(logits, y.to(self.device))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            classifier.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for x, y in test_dataloader:
                    features = self.global_encoder(x.to(self.device))
                    logits = classifier(features)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y.to(self.device)).sum().item()
                    total += y.size(0)
            
            accuracy = correct / total
            self.results['accuracy_history'].append(accuracy)
            
            print(f"Linear Eval Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={accuracy:.4f}")
        
        return accuracy


def generate_cifar10_dataloaders(num_clients: int = 5, batch_size: int = 32):
    """Generate partitioned CIFAR-10 dataloaders for federated learning."""
    transform = SimCLRAugmentation(image_size=32)
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    num_samples = len(train_dataset)
    samples_per_client = num_samples // num_clients
    
    dataloaders = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples
        
        indices = list(range(start, end))
        subset = torch.utils.data.Subset(train_dataset, indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)
    
    return dataloaders


def run_fed_simclr_demo():
    """Run FedSimCLR demonstration."""
    print("\n" + "=" * 70)
    print("  FEDERATED SELF-SUPERVISED LEARNING (FedSimCLR)")
    print("=" * 70)
    
    try:
        import torch
        print(f"\n1. Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        print("\n2. Setting up FedSimCLR coordinator...")
        coordinator = FedSimCLRCoordinator(
            num_clients=3,
            num_rounds=3,
            local_epochs=1,
            device='cpu'
        )
        coordinator.setup()
        print(f"    Coordinator setup with {len(coordinator.clients)} clients")
        
        print("\n3. Creating CIFAR-10 dataloaders...")
        dataloaders = generate_cifar10_dataloaders(num_clients=3)
        print(f"    {len(dataloaders)} dataloaders created")
        
        print("\n4. Testing Global Buffer...")
        buffer = coordinator.buffer_server.buffer
        print(f"    Buffer size: {buffer.get_size()}")
        
        print("\n5. Testing Client Initialization...")
        client = coordinator.clients[0]
        print(f"    Client {client.client_id} created")
        
        print("\n6. Testing Feature Extraction...")
        dummy = torch.randn(2, 3, 32, 32)
        features = coordinator.global_encoder(dummy)
        print(f"    Feature shape: {features.shape}")
        
        print("\n7. Testing Contrastive Loss...")
        from .contrastive_loss import NTXentLoss
        loss_fn = NTXentLoss()
        z1 = torch.randn(8, 128)
        z2 = torch.randn(8, 128)
        loss = loss_fn(z1, z2)
        print(f"    NT-Xent Loss: {loss.item():.4f}")
        
        print("\n FedSimCLR demo completed successfully!")
        
    except ImportError as e:
        print(f" Import error: {e}")
    except Exception as e:
        print(f" Error in FedSimCLR demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_fed_simclr_demo()