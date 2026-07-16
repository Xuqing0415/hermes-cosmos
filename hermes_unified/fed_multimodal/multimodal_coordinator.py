"""
Federated Multimodal Learning Coordinator

Integrates all components for end-to-end federated multimodal learning.
"""

import torch
from typing import Dict, List, Any, Optional

from .multimodal_client import MultimodalFedClient
from .multimodal_server import MultimodalFedServer
from .modality_aligner import FederatedAlignerServer
from .missing_modality_handler import ModalityMasker, KnowledgeDistillationFiller
from .data_utils import (
    generate_synthetic_multimodal_data,
    split_modalities_across_clients,
    create_dataloader
)


class MultimodalFedCoordinator:
    """Coordinator for federated multimodal learning."""
    
    def __init__(self, num_clients: int = 3, num_classes: int = 10, 
                 missing_modalities_per_client: int = 1, device: str = 'cpu'):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.device = device
        
        self.clients: List[MultimodalFedClient] = []
        self.server: Optional[MultimodalFedServer] = None
        self.aligner: Optional[FederatedAlignerServer] = None
        
        self.missing_modality_handler = ModalityMasker(mask_prob=0.1)
        self.distillation_filler = KnowledgeDistillationFiller()
        
        self.results = {
            'round_accuracy': [],
            'communication_history': [],
            'alignment_loss': []
        }
    
    def setup(self):
        """Set up clients and server."""
        data = generate_synthetic_multimodal_data(num_samples=1000)
        client_datasets = split_modalities_across_clients(
            data, 
            num_clients=self.num_clients,
            missing_modalities_per_client=1
        )
        
        self.clients = []
        all_modalities = set()
        
        for i, dataset in enumerate(client_datasets):
            modalities = set()
            for sample in dataset[:10]:
                modalities.update(sample['modalities'].keys())
            
            all_modalities.update(modalities)
            
            client = MultimodalFedClient(
                client_id=i,
                available_modalities=list(modalities),
                num_classes=self.num_classes,
                device=self.device
            )
            self.clients.append(client)
        
        self.server = MultimodalFedServer(list(all_modalities), self.num_classes)
        self.aligner = FederatedAlignerServer(feature_dim=64)
        
        if self.clients:
            self.server.initialize_shared_params(self.clients[0].get_shared_parameters_dict())
        
        return client_datasets
    
    def run(self, num_rounds: int = 5, local_epochs: int = 1, 
            apply_dp: bool = False) -> Dict[str, Any]:
        """
        Run federated multimodal learning.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Number of local epochs per client
            apply_dp: Whether to apply differential privacy
        
        Returns:
            Results dictionary
        """
        client_datasets = self.setup()
        
        print(f"Starting Federated Multimodal Learning with {self.num_clients} clients")
        print(f"Modalities across clients: {self.server.modalities}")
        
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            global_params = self.server.get_global_parameters()
            updates = []
            
            for i, client in enumerate(self.clients):
                print(f"Training client {client.client_id}...")
                dataloader = create_dataloader(client_datasets[i], batch_size=32)
                
                update = client.local_train(
                    dataloader,
                    global_params,
                    num_epochs=local_epochs
                )
                updates.append(update)
            
            self.server.aggregate_updates(updates, apply_dp=apply_dp)
            
            accuracy = self.evaluate()
            self.results['round_accuracy'].append(accuracy)
            self.results['communication_history'].append(self.server.total_communication)
            
            print(f"Round {round_idx + 1} - Accuracy: {accuracy:.4f}")
            print(f"Total Communication: {self.server.total_communication:.2f} MB")
        
        return self.results
    
    def evaluate(self) -> float:
        """Evaluate all clients."""
        total_correct = 0
        total_samples = 0
        
        for client in self.clients:
            accuracy = client.evaluate(create_dataloader([]))
            total_correct += accuracy
            total_samples += 1
        
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def run_with_alignment(self, num_rounds: int = 5, local_epochs: int = 1):
        """Run with cross-modal alignment."""
        client_datasets = self.setup()
        
        print("\n=== Federated Multimodal Learning with Cross-Modal Alignment ===")
        
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            global_params = self.server.get_global_parameters()
            updates = []
            all_features = {'image': [], 'text': [], 'tabular': []}
            
            for i, client in enumerate(self.clients):
                print(f"Training client {client.client_id}...")
                dataloader = create_dataloader(client_datasets[i], batch_size=32)
                
                update = client.local_train(
                    dataloader,
                    global_params,
                    num_epochs=local_epochs
                )
                updates.append(update)
                
                for modality in client.available_modalities:
                    if modality in all_features:
                        for batch in dataloader:
                            if modality in batch['modalities']:
                                data = batch['modalities'][modality].to(self.device)
                                encoder = client.modality_encoders[modality]
                                encoder.eval()
                                with torch.no_grad():
                                    features = encoder(data)
                                all_features[modality].append(features)
            
            self.server.aggregate_updates(updates)
            
            aligned_features = {}
            for modality, feats in all_features.items():
                if feats:
                    aligned_features[modality] = torch.cat(feats, dim=0)
            
            if len(aligned_features) >= 2:
                alignment_loss = self.aligner.train_alignment(aligned_features)
                self.results['alignment_loss'].append(alignment_loss)
                print(f"Alignment Loss: {alignment_loss:.4f}")
            
            accuracy = self.evaluate()
            self.results['round_accuracy'].append(accuracy)
            
            print(f"Round {round_idx + 1} - Accuracy: {accuracy:.4f}")
        
        return self.results


def run_multimodal_demo():
    """Run demonstration of federated multimodal learning."""
    print("\n" + "=" * 70)
    print("  FEDERATED MULTIMODAL LEARNING DEMO")
    print("=" * 70)
    
    try:
        coordinator = MultimodalFedCoordinator(
            num_clients=3,
            num_classes=10,
            missing_modalities_per_client=1,
            device='cpu'
        )
        
        print("\n1. Setting up federation...")
        coordinator.setup()
        print(f"   Clients: {len(coordinator.clients)}")
        print(f"   Server initialized: {coordinator.server is not None}")
        
        print("\n2. Running federated training...")
        results = coordinator.run(num_rounds=3, local_epochs=1)
        
        print("\n3. Results:")
        for i, acc in enumerate(results['round_accuracy']):
            print(f"   Round {i+1}: Accuracy = {acc:.4f}")
        
        print("\n4. Running with alignment...")
        coordinator2 = MultimodalFedCoordinator(
            num_clients=3,
            num_classes=10,
            missing_modalities_per_client=1,
            device='cpu'
        )
        results2 = coordinator2.run_with_alignment(num_rounds=3, local_epochs=1)
        
        print("\n5. Results with alignment:")
        for i, acc in enumerate(results2['round_accuracy']):
            print(f"   Round {i+1}: Accuracy = {acc:.4f}")
        
        print("\n Federated Multimodal Learning demo completed!")
        
    except Exception as e:
        print(f" Error in Multimodal Learning demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_multimodal_demo()