"""
LLM Federated Server Module

Implements federated learning server for LLM fine-tuning with LoRA.
"""

import torch
import numpy as np
from typing import Dict, List, Any
import copy


class LLMFederatedServer:
    """
    Federated server for LLM fine-tuning.
    
    Manages global LoRA weights and aggregates updates from clients.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize LLM federated server.
        
        Args:
            model: Base model with LoRA adapter
            tokenizer: Tokenizer for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Global LoRA weights
        self.global_lora_weights = self._extract_lora_weights()
        
        # Statistics
        self.round = 0
        self.aggregation_history = []
        self.evaluation_history = []
        self.total_communication = 0.0  # in MB
    
    def _extract_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights from the model."""
        return {
            k: v.detach().cpu().clone() 
            for k, v in self.model.state_dict().items() 
            if 'lora' in k.lower()
        }
    
    def _set_lora_weights(self, weights: Dict[str, torch.Tensor]):
        """Set LoRA weights on the model."""
        current_state = self.model.state_dict()
        for k, v in weights.items():
            if k in current_state:
                current_state[k].data.copy_(v.to(self.device))
        self.model.load_state_dict(current_state, strict=False)
    
    def aggregate(self, updates: List[Dict[str, torch.Tensor]], 
                  method: str = 'fedavg') -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.
        
        Args:
            updates: List of weight updates from clients
            method: Aggregation method ('fedavg', 'krum', 'median')
            
        Returns:
            aggregated_weights: New global weights
        """
        if not updates:
            return self.global_lora_weights
        
        # Calculate communication cost
        for update in updates:
            self.total_communication += self._calculate_update_size(update)
        
        if method == 'fedavg':
            new_weights = self._fedavg_aggregate(updates)
        elif method == 'krum':
            new_weights = self._krum_aggregate(updates)
        elif method == 'median':
            new_weights = self._median_aggregate(updates)
        else:
            new_weights = self._fedavg_aggregate(updates)
        
        # Update global weights (add updates to current weights)
        for k in self.global_lora_weights.keys():
            self.global_lora_weights[k] = self.global_lora_weights[k] + new_weights[k]
        
        # Load new weights into model
        self._set_lora_weights(self.global_lora_weights)
        
        self.round += 1
        self.aggregation_history.append({
            'round': self.round,
            'method': method,
            'num_updates': len(updates)
        })
        
        return self.global_lora_weights
    
    def _fedavg_aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation."""
        aggregated = {}
        
        for key in updates[0].keys():
            tensors = [update[key] for update in updates]
            aggregated[key] = torch.stack(tensors).mean(dim=0)
        
        return aggregated
    
    def _krum_aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Krum aggregation for robustness."""
        n = len(updates)
        f = n // 3  # Number of Byzantine clients we can tolerate
        
        # Compute distances between updates
        distances = []
        for i, u1 in enumerate(updates):
            dist_sum = 0.0
            for j, u2 in enumerate(updates):
                if i != j:
                    for k in u1.keys():
                        dist_sum += torch.norm(u1[k] - u2[k]).item()
            distances.append((i, dist_sum))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Select top (n - f - 1) updates
        selected_indices = [i for i, _ in distances[:n - f - 1]]
        selected_updates = [updates[i] for i in selected_indices]
        
        # Average the selected updates
        return self._fedavg_aggregate(selected_updates)
    
    def _median_aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Median aggregation for robustness."""
        aggregated = {}
        
        for key in updates[0].keys():
            tensors = torch.stack([update[key] for update in updates])
            aggregated[key] = torch.median(tensors, dim=0).values
        
        return aggregated
    
    def _calculate_update_size(self, update: Dict[str, torch.Tensor]) -> float:
        """Calculate size of update in MB."""
        total_bytes = sum(t.numel() * t.element_size() for t in update.values())
        return total_bytes / (1024 * 1024)
    
    def evaluate(self, eval_dataset, batch_size: int = 4) -> float:
        """
        Evaluate the global model on the evaluation dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            batch_size: Batch size for evaluation
            
        Returns:
            perplexity: Perplexity score
        """
        self.model.eval()
        
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                
                loss = outputs.loss
                total_loss += loss.item() * batch['input_ids'].shape[0]
                total_tokens += batch['input_ids'].numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        self.evaluation_history.append({
            'round': self.round,
            'loss': avg_loss,
            'perplexity': perplexity
        })
        
        return perplexity
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global LoRA weights."""
        return copy.deepcopy(self.global_lora_weights)
    
    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate text using the global model."""
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
        """Get server statistics."""
        return {
            'round': self.round,
            'total_communication': self.total_communication,
            'evaluation_history': self.evaluation_history,
            'aggregation_history': self.aggregation_history,
            'num_lora_params': sum(w.numel() for w in self.global_lora_weights.values())
        }


class LLMFederatedCoordinator:
    """
    Coordinator for LLM federated fine-tuning.
    
    Manages the overall federated learning process.
    """
    
    def __init__(self, model, tokenizer, client_datasets, test_dataset):
        """
        Initialize coordinator.
        
        Args:
            model: Base model with LoRA adapter
            tokenizer: Tokenizer
            client_datasets: List of client datasets
            test_dataset: Test dataset for evaluation
        """
        self.server = LLMFederatedServer(model, tokenizer)
        self.tokenizer = tokenizer
        self.client_datasets = client_datasets
        self.test_dataset = test_dataset
        
        # Create clients
        from llm_client import LLMClientFactory
        self.clients = LLMClientFactory.create_clients(
            model, tokenizer, client_datasets, batch_size=4
        )
        
        self.results = {
            'perplexity_history': [],
            'communication_history': [],
            'round_times': []
        }
    
    def run(self, num_rounds: int = 10, clients_per_round: int = 5, 
            local_epochs: int = 1, aggregation_method: str = 'fedavg'):
        """
        Run federated training.
        
        Args:
            num_rounds: Number of federated rounds
            clients_per_round: Number of clients to select per round
            local_epochs: Number of local epochs per client
            aggregation_method: Aggregation method
        """
        import time
        
        print(f"Starting federated training with {len(self.clients)} clients")
        print(f"Rounds: {num_rounds}, Clients per round: {clients_per_round}")
        
        for round_idx in range(num_rounds):
            start_time = time.time()
            
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            # Select clients
            selected_indices = np.random.choice(
                len(self.clients), 
                size=min(clients_per_round, len(self.clients)),
                replace=False
            )
            selected_clients = [self.clients[i] for i in selected_indices]
            
            print(f"Selected clients: {[c.client_id for c in selected_clients]}")
            
            # Get global weights
            global_weights = self.server.get_global_weights()
            
            # Local training
            updates = []
            for client in selected_clients:
                print(f"Training client {client.client_id}...")
                update = client.local_train(global_weights, local_epochs=local_epochs)
                updates.append(update)
            
            # Aggregate updates
            self.server.aggregate(updates, method=aggregation_method)
            
            # Evaluate
            perplexity = self.server.evaluate(self.test_dataset)
            print(f"Round {round_idx + 1} Perplexity: {perplexity:.2f}")
            
            # Record results
            self.results['perplexity_history'].append(perplexity)
            self.results['communication_history'].append(self.server.total_communication)
            self.results['round_times'].append(time.time() - start_time)
        
        print("\n=== Training Complete ===")
        print(f"Final Perplexity: {self.results['perplexity_history'][-1]:.2f}")
        print(f"Total Communication: {self.server.total_communication:.2f} MB")
        
        return self.results


if __name__ == "__main__":
    # Test server
    from llm_model import load_llm, wrap_with_lora
    from non_iid_data import create_synthetic_non_iid_data, tokenize_dataset
    
    print("=== Testing LLM Federated Server ===")
    
    # Load model
    model, tokenizer = load_llm(use_quantization=False)
    model, _ = wrap_with_lora(model)
    
    # Create test data
    client_datasets = create_synthetic_non_iid_data(num_clients=3, samples_per_client=30, alpha=0.5)
    
    # Tokenize datasets
    tokenized_datasets = [tokenize_dataset(ds, tokenizer) for ds in client_datasets]
    
    # Create server
    server = LLMFederatedServer(model, tokenizer)
    print(f"Server created with {len(server.global_lora_weights)} LoRA weight tensors")
    
    # Test aggregation
    num_clients = 3
    updates = []
    
    for _ in range(num_clients):
        update = {k: torch.randn_like(v) * 0.01 for k, v in server.global_lora_weights.items()}
        updates.append(update)
    
    server.aggregate(updates, method='fedavg')
    print(f"Aggregation complete. Round: {server.round}")
    
    # Test evaluation
    eval_perplexity = server.evaluate(tokenized_datasets[0])
    print(f"Evaluation Perplexity: {eval_perplexity:.2f}")
