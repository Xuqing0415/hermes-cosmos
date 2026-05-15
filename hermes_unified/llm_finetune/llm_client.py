"""
LLM Federated Client Module

Implements federated learning clients for LLM fine-tuning with LoRA.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, Any
import copy


class LLMFederatedClient:
    """
    Federated client for LLM fine-tuning with LoRA.
    
    Each client maintains its own LoRA-adapted model and local dataset.
    """
    
    def __init__(self, client_id: int, model, tokenizer, dataset, 
                 batch_size: int = 4, lr: float = 2e-4):
        """
        Initialize LLM federated client.
        
        Args:
            client_id: Unique client identifier
            model: Base model with LoRA adapter
            tokenizer: Tokenizer for the model
            dataset: Local dataset for this client
            batch_size: Training batch size
            lr: Learning rate
        """
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Device
        self.device = next(model.parameters()).device
        
        # Training statistics
        self.train_loss_history = []
        self.eval_loss_history = []
    
    def _create_dataloader(self, dataset, shuffle: bool = True):
        """Create dataloader for the dataset."""
        sampler = RandomSampler(dataset) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True
        )
    
    def local_train(self, global_lora_weights: Dict[str, torch.Tensor], 
                    local_epochs: int = 1) -> Dict[str, torch.Tensor]:
        """
        Perform local training and return LoRA weight updates.
        
        Args:
            global_lora_weights: Global LoRA weights to initialize with
            local_epochs: Number of local training epochs
            
        Returns:
            weight_updates: Difference between local and global weights
        """
        # Load global LoRA weights
        self._load_lora_weights(global_lora_weights)
        
        # Set model to training mode
        self.model.train()
        
        dataloader = self._create_dataloader(self.dataset)
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                
                loss = outputs.loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.train_loss_history.append(avg_loss)
            print(f"Client {self.client_id}, Epoch {epoch+1}/{local_epochs}, Loss: {avg_loss:.4f}")
        
        # Compute weight updates (local - global)
        local_weights = self._extract_lora_weights()
        updates = {
            k: (local_weights[k] - global_lora_weights[k]).cpu()
            for k in global_lora_weights.keys()
        }
        
        return updates
    
    def _load_lora_weights(self, weights: Dict[str, torch.Tensor]):
        """Load LoRA weights into the model."""
        current_state = self.model.state_dict()
        for k, v in weights.items():
            if k in current_state:
                current_state[k].data.copy_(v.to(self.device))
        self.model.load_state_dict(current_state, strict=False)
    
    def _extract_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights from the model."""
        return {
            k: v.detach().cpu().clone() 
            for k, v in self.model.state_dict().items() 
            if 'lora' in k.lower()
        }
    
    def evaluate(self, eval_dataset) -> float:
        """Evaluate the model on the evaluation dataset."""
        self.model.eval()
        
        dataloader = self._create_dataloader(eval_dataset, shuffle=False)
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.eval_loss_history.append(avg_loss)
        
        return avg_loss
    
    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate text from a prompt."""
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'dataset_size': len(self.dataset),
            'train_loss_history': self.train_loss_history,
            'eval_loss_history': self.eval_loss_history
        }


class LLMClientFactory:
    """Factory class for creating LLM federated clients."""
    
    @staticmethod
    def create_clients(model, tokenizer, client_datasets, **kwargs) -> list:
        """
        Create multiple LLM federated clients.
        
        Args:
            model: Base model with LoRA adapter
            tokenizer: Tokenizer
            client_datasets: List of datasets for each client
            kwargs: Additional arguments for client creation
            
        Returns:
            List of LLMFederatedClient instances
        """
        clients = []
        
        for client_id, dataset in enumerate(client_datasets):
            # Create a copy of the model for each client
            client_model = copy.deepcopy(model)
            
            client = LLMFederatedClient(
                client_id=client_id,
                model=client_model,
                tokenizer=tokenizer,
                dataset=dataset,
                **kwargs
            )
            
            clients.append(client)
            print(f"Created client {client_id} with {len(dataset)} samples")
        
        return clients


if __name__ == "__main__":
    # Test client creation
    from llm_model import load_llm, wrap_with_lora
    from non_iid_data import create_synthetic_non_iid_data
    
    print("=== Testing LLM Federated Client ===")
    
    # Load model
    model, tokenizer = load_llm(use_quantization=False)
    model, _ = wrap_with_lora(model)
    
    # Create test data
    client_datasets = create_synthetic_non_iid_data(num_clients=2, samples_per_client=20, alpha=0.5)
    
    # Create clients
    clients = LLMClientFactory.create_clients(model, tokenizer, client_datasets)
    
    print(f"\nCreated {len(clients)} clients")
    
    # Test local training
    global_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora' in k.lower()}
    
    for client in clients:
        print(f"\nTraining client {client.client_id}")
        updates = client.local_train(global_weights, local_epochs=1)
        print(f"Number of weight updates: {len(updates)}")
