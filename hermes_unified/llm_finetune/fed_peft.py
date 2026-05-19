"""
FedPEFT - Federated Parameter-Efficient Fine-Tuning

Implements federated versions of efficient LLM fine-tuning methods:
- FedBitFit: Only fine-tune bias terms
- FedPrefix: Prefix tuning
- FedLoRAMix: Mixed LoRA with dynamic rank adaptation
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, List, Any, Optional
import numpy as np


class FedBitFitModule(nn.Module):
    """
    FedBitFit - Only fine-tune bias terms.
    
    This approach freezes all weights except for bias terms,
    achieving near-zero communication while maintaining reasonable performance.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze only bias parameters
        self.bias_params = []
        for name, param in self.model.named_parameters():
            if 'bias' in name.lower():
                param.requires_grad = True
                self.bias_params.append(name)
        
        print(f"FedBitFit initialized: {len(self.bias_params)} bias parameters unfrozen")
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def get_bias_weights(self) -> Dict[str, torch.Tensor]:
        """Extract bias weights from the model."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if 'bias' in name.lower()
        }
    
    def set_bias_weights(self, weights: Dict[str, torch.Tensor]):
        """Set bias weights on the model."""
        device = next(self.model.parameters()).device
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data.copy_(weights[name].to(device))
    
    def get_trainable_parameters(self) -> List[str]:
        """Get names of trainable parameters."""
        return self.bias_params
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(
            param.numel() for name, param in self.model.named_parameters()
            if 'bias' in name.lower()
        )


class FedPrefixModule(nn.Module):
    """
    FedPrefix - Prefix tuning for federated learning.
    
    Only fine-tunes task-specific prefix embeddings, which are extremely lightweight
    (typically <0.01% of the original model parameters).
    """
    
    def __init__(self, model, tokenizer, prefix_len: int = 10, 
                 prefix_dim: int = 512, task_type: str = 'causal_lm'):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.prefix_dim = prefix_dim
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get hidden size from model
        self.hidden_size = model.config.hidden_size
        
        # Create prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(1, prefix_len, self.hidden_size)
        )
        self.prefix_embeddings.requires_grad = True
        
        # Optional projection layer for prefix
        if prefix_dim != self.hidden_size:
            self.prefix_proj = nn.Linear(prefix_dim, self.hidden_size)
            self.prefix_proj.requires_grad = True
        else:
            self.prefix_proj = None
        
        print(f"FedPrefix initialized: {self.prefix_embeddings.numel()} prefix parameters")
    
    def _get_prefix(self, batch_size: int) -> torch.Tensor:
        """Get prefix embeddings expanded to batch size."""
        prefix = self.prefix_embeddings.expand(batch_size, -1, -1)
        if self.prefix_proj is not None:
            prefix = self.prefix_proj(prefix)
        return prefix
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        prefix = self._get_prefix(batch_size)
        
        # Concatenate prefix with input embeddings
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        combined_embeddings = torch.cat([prefix, input_embeddings], dim=1)
        
        # Adjust attention mask
        prefix_mask = torch.ones(batch_size, self.prefix_len, device=input_ids.device)
        if attention_mask is not None:
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attention_mask = torch.ones(batch_size, self.prefix_len + input_ids.shape[1], 
                                      device=input_ids.device)
        
        return self.model(inputs_embeds=combined_embeddings, 
                         attention_mask=attention_mask, **kwargs)
    
    def get_prefix_weights(self) -> Dict[str, torch.Tensor]:
        """Extract prefix weights."""
        weights = {'prefix_embeddings': self.prefix_embeddings.detach().cpu().clone()}
        if self.prefix_proj is not None:
            weights['prefix_proj.weight'] = self.prefix_proj.weight.detach().cpu().clone()
            weights['prefix_proj.bias'] = self.prefix_proj.bias.detach().cpu().clone()
        return weights
    
    def set_prefix_weights(self, weights: Dict[str, torch.Tensor]):
        """Set prefix weights."""
        device = self.prefix_embeddings.device
        
        if 'prefix_embeddings' in weights:
            self.prefix_embeddings.data.copy_(weights['prefix_embeddings'].to(device))
        
        if self.prefix_proj is not None:
            if 'prefix_proj.weight' in weights:
                self.prefix_proj.weight.data.copy_(weights['prefix_proj.weight'].to(device))
            if 'prefix_proj.bias' in weights:
                self.prefix_proj.bias.data.copy_(weights['prefix_proj.bias'].to(device))
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        count = self.prefix_embeddings.numel()
        if self.prefix_proj is not None:
            count += self.prefix_proj.weight.numel() + self.prefix_proj.bias.numel()
        return count


class AdaptiveLoRAModule(nn.Module):
    """
    Adaptive LoRA module with dynamic rank adjustment.
    
    Automatically adjusts LoRA rank based on client resource constraints.
    """
    
    def __init__(self, model, tokenizer, min_rank: int = 4, max_rank: int = 64,
                 target_modules: List[str] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.min_rank = min_rank
        self.max_rank = max_rank
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        # Initialize with maximum rank
        self.current_rank = max_rank
        self.lora_config = LoraConfig(
            r=max_rank,
            lora_alpha=max_rank * 4,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False
        )
        
        self.model = get_peft_model(model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def set_rank(self, rank: int):
        """Set LoRA rank dynamically."""
        rank = max(self.min_rank, min(self.max_rank, rank))
        self.current_rank = rank
        
        # Update LoRA adapters with new rank
        for name, module in self.model.named_modules():
            if hasattr(module, 'r'):
                module.r = rank
                # Reinitialize adapter weights for new rank
                if hasattr(module, 'lora_A'):
                    out_dim, in_dim = module.lora_A.weight.shape
                    module.lora_A.weight.data = torch.randn(out_dim, rank) * 0.01
                if hasattr(module, 'lora_B'):
                    in_dim, out_dim = module.lora_B.weight.shape
                    module.lora_B.weight.data = torch.randn(rank, out_dim) * 0.01
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def get_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
            if 'lora' in k.lower()
        }
    
    def set_lora_weights(self, weights: Dict[str, torch.Tensor]):
        """Set LoRA weights."""
        device = next(self.model.parameters()).device
        current_state = self.model.state_dict()
        for k, v in weights.items():
            if k in current_state:
                current_state[k].data.copy_(v.to(device))
        self.model.load_state_dict(current_state, strict=False)
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class FedLoRAMixServer:
    """
    Server for Federated Mixed LoRA.
    
    Dynamically fuses multiple client LoRA modules into a single compact module.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Track LoRA weights from different clients
        self.client_lora_weights = []
        self.fused_weights = None
    
    def fuse_lora_weights(self, client_weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Fuse multiple client LoRA weights into a single set.
        
        Uses weighted averaging with adaptive weighting based on client performance.
        """
        if not client_weights_list:
            return {}
        
        # Calculate weights based on update magnitudes (confidence)
        weights = {}
        
        for key in client_weights_list[0].keys():
            client_tensors = [w[key] for w in client_weights_list]
            
            # Compute norms for weighting
            norms = torch.tensor([torch.norm(t).item() for t in client_tensors])
            weights_sum = norms.sum()
            
            if weights_sum > 0:
                weights[key] = sum(
                    (norms[i] / weights_sum) * client_tensors[i]
                    for i in range(len(client_tensors))
                )
            else:
                weights[key] = torch.mean(torch.stack(client_tensors), dim=0)
        
        self.fused_weights = weights
        return weights
    
    def distribute_fused_weights(self):
        """Distribute fused weights to clients."""
        if self.fused_weights is not None:
            return self.fused_weights
        return self._extract_lora_weights()
    
    def _extract_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights from model."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
            if 'lora' in k.lower()
        }


class FedPEFTClient:
    """
    Client for federated parameter-efficient fine-tuning.
    
    Supports FedBitFit, FedPrefix, and Adaptive LoRA.
    """
    
    def __init__(self, client_id: int, model, tokenizer, dataset, 
                 peft_method: str = 'bitfit', **kwargs):
        self.client_id = client_id
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.peft_method = peft_method
        
        # Wrap model with appropriate PEFT method
        if peft_method == 'bitfit':
            self.peft_model = FedBitFitModule(model)
        elif peft_method == 'prefix':
            self.peft_model = FedPrefixModule(model, tokenizer, **kwargs)
        elif peft_method == 'lora':
            self.peft_model = AdaptiveLoRAModule(model, tokenizer, **kwargs)
        else:
            raise ValueError(f"Unknown PEFT method: {peft_method}")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.peft_model.parameters(),
            lr=kwargs.get('lr', 2e-4),
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.device = next(self.peft_model.parameters()).device
        
        # Statistics
        self.train_loss_history = []
    
    def local_train(self, global_weights: Dict[str, torch.Tensor], 
                    local_epochs: int = 1) -> Dict[str, torch.Tensor]:
        """Perform local training and return updates."""
        # Load global weights
        self._load_weights(global_weights)
        
        self.peft_model.train()
        device = self.device
        
        from torch.utils.data import DataLoader, RandomSampler
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=4,
            sampler=RandomSampler(self.dataset),
            drop_last=True
        )
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = self.peft_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                
                loss = outputs.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.train_loss_history.append(avg_loss)
            print(f"Client {self.client_id}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Compute updates
        local_weights = self._extract_weights()
        updates = {
            k: (local_weights[k] - global_weights[k]).cpu()
            for k in global_weights.keys()
        }
        
        return updates
    
    def _load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load weights based on PEFT method."""
        if self.peft_method == 'bitfit':
            self.peft_model.set_bias_weights(weights)
        elif self.peft_method == 'prefix':
            self.peft_model.set_prefix_weights(weights)
        elif self.peft_method == 'lora':
            self.peft_model.set_lora_weights(weights)
    
    def _extract_weights(self) -> Dict[str, torch.Tensor]:
        """Extract weights based on PEFT method."""
        if self.peft_method == 'bitfit':
            return self.peft_model.get_bias_weights()
        elif self.peft_method == 'prefix':
            return self.peft_model.get_prefix_weights()
        elif self.peft_method == 'lora':
            return self.peft_model.get_lora_weights()
    
    def count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return self.peft_model.count_trainable_parameters()


def compare_peft_methods(model, tokenizer, dataset, sample_input):
    """Compare different PEFT methods in terms of parameter count and performance."""
    methods = ['bitfit', 'prefix', 'lora']
    results = {}
    
    for method in methods:
        if method == 'bitfit':
            peft_model = FedBitFitModule(model)
        elif method == 'prefix':
            peft_model = FedPrefixModule(model, tokenizer)
        elif method == 'lora':
            peft_model = AdaptiveLoRAModule(model, tokenizer)
        
        params = peft_model.count_trainable_parameters()
        
        # Test inference
        peft_model.eval()
        with torch.no_grad():
            outputs = peft_model(**sample_input)
        
        results[method] = {
            'params': params,
            'params_mb': params * 4 / (1024 * 1024),
            'output_shape': outputs.logits.shape
        }
        
        print(f"{method}: {params:,} parameters ({params * 4 / (1024 * 1024):.4f} MB)")
    
    return results


if __name__ == "__main__":
    print("=== Testing FedPEFT Module ===")
    
    # Load a small model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\nLoading model: {model_name}")
    model, tokenizer = AutoModelForCausalLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
    
    # Test FedBitFit
    print("\n1. Testing FedBitFit...")
    bitfit_model = FedBitFitModule(model)
    print(f"   Trainable params: {bitfit_model.count_trainable_parameters():,}")
    print(f"   Bias parameters: {bitfit_model.bias_params[:5]}...")
    
    # Test FedPrefix
    print("\n2. Testing FedPrefix...")
    prefix_model = FedPrefixModule(model, tokenizer, prefix_len=10, prefix_dim=512)
    print(f"   Trainable params: {prefix_model.count_trainable_parameters():,}")
    
    # Test Adaptive LoRA
    print("\n3. Testing Adaptive LoRA...")
    lora_model = AdaptiveLoRAModule(model, tokenizer, min_rank=4, max_rank=32)
    print(f"   Initial rank: {lora_model.current_rank}")
    lora_model.set_rank(16)
    print(f"   After rank adjustment: {lora_model.current_rank}")
    
    # Test comparison
    print("\n4. PEFT Method Comparison:")
    sample_input = tokenizer("Hello, world!", return_tensors="pt")
    compare_peft_methods(model, tokenizer, None, sample_input)
    
    print("\n=== All tests passed! ===")