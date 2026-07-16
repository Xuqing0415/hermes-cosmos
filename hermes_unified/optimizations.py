#!/usr/bin/env python3
"""
Hermes Unified Optimizations Module


1. Top-k + 
2. 
3. Checkpoint
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os


class GradientCompressor:
    """ -  Top-k + """
    
    def __init__(self, compression_ratio: float = 0.1, error_compensation: bool = True):
        """
        Args:
            compression_ratio:  (0.01-0.5)
            error_compensation: 
        """
        self.compression_ratio = compression_ratio
        self.error_compensation = error_compensation
        self.residual_error = {}  # 
    
    def compress(self, grad: torch.Tensor, param_name: str = "") -> Tuple[torch.Tensor, Dict]:
        """
        
        
        Args:
            grad: 
            param_name: 
        
        Returns:
            compressed_grad: 
            metadata: 
        """
        if self.compression_ratio >= 1.0:
            return grad, {"type": "full", "shape": grad.shape}
        
        # Flatten 
        grad_flat = grad.flatten()
        n_elements = grad_flat.numel()
        n_top = max(1, int(n_elements * self.compression_ratio))
        
        # Top-k 
        top_values, top_indices = torch.topk(grad_flat.abs(), n_top, largest=True)
        
        # 
        signs = torch.sign(grad_flat[top_indices])
        compressed_values = top_values * signs
        
        # 
        if self.error_compensation:
            if param_name not in self.residual_error:
                self.residual_error[param_name] = torch.zeros_like(grad)
            
            # 
            grad_with_error = grad + self.residual_error[param_name]
            
            # 
            self.residual_error[param_name] = grad_with_error - \
                torch.zeros_like(grad_flat).scatter_(0, top_indices, compressed_values).view(grad.shape)
            
            # 
            compressed_values = top_values * torch.sign(grad_with_error.flatten()[top_indices])
        
        return (top_indices, compressed_values), {
            "type": "topk",
            "shape": grad.shape,
            "n_top": n_top,
            "compression_ratio": self.compression_ratio,
            "param_name": param_name
        }
    
    def decompress(self, compressed_data: Tuple, metadata: Dict) -> torch.Tensor:
        """
        
        
        Args:
            compressed_data:  (indices, values)
            metadata: 
        
        Returns:
            decompressed_grad: 
        """
        if metadata["type"] == "full":
            return compressed_data
        
        indices, values = compressed_data
        shape = metadata["shape"]
        
        # 
        decompressed = torch.zeros(int(np.prod(shape)), device=values.device)
        decompressed.scatter_(0, indices, values)
        
        return decompressed.view(shape)


class ShardedParameterServer:
    """"""
    
    def __init__(self, num_shards: int = 4, device: str = "cpu"):
        """
        Args:
            num_shards: 
            device: 
        """
        self.num_shards = num_shards
        self.device = torch.device(device)
        self.shards = [{} for _ in range(num_shards)]  # 
        self.param_to_shard = {}  #  -> 
        self.shard_sizes = [0] * num_shards  # 
    
    def _get_shard_index(self, param_name: str) -> int:
        """"""
        if param_name in self.param_to_shard:
            return self.param_to_shard[param_name]
        
        # 
        shard_idx = hash(param_name) % self.num_shards
        self.param_to_shard[param_name] = shard_idx
        return shard_idx
    
    def register_parameter(self, param_name: str, param: torch.nn.Parameter):
        """"""
        shard_idx = self._get_shard_index(param_name)
        self.shards[shard_idx][param_name] = param.data.to(self.device).clone()
        self.shard_sizes[shard_idx] += 1
    
    def update_parameters(self, updates: Dict[str, torch.Tensor]):
        """"""
        for param_name, update in updates.items():
            shard_idx = self._get_shard_index(param_name)
            if param_name in self.shards[shard_idx]:
                self.shards[shard_idx][param_name] += update.to(self.device)
    
    def get_parameters(self, param_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """"""
        result = {}
        
        if param_names is None:
            # 
            for shard in self.shards:
                result.update(shard)
        else:
            # 
            for param_name in param_names:
                shard_idx = self._get_shard_index(param_name)
                if param_name in self.shards[shard_idx]:
                    result[param_name] = self.shards[shard_idx][param_name]
        
        return result
    
    def get_shard_info(self) -> Dict:
        """"""
        return {
            "num_shards": self.num_shards,
            "shard_sizes": self.shard_sizes,
            "total_params": sum(self.shard_sizes)
        }


class LightweightCheckpointer:
    """Checkpoint"""
    
    def __init__(self, save_dir: str = "checkpoints", async_save: bool = True):
        """
        Args:
            save_dir: checkpoint
            async_save: 
        """
        self.save_dir = save_dir
        self.async_save = async_save
        self.save_counter = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model: nn.Module, epoch: int = 0, save_optimizer: bool = False,
             optimizer: Optional[torch.optim.Optimizer] = None) -> str:
        """
        checkpoint
        
        Args:
            model: 
            epoch: epoch
            save_optimizer: 
            optimizer: 
        
        Returns:
            checkpoint_path: 
        """
        # 
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'save_time': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else 'unknown'
        }
        
        if save_optimizer and optimizer is not None:
            state_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        # 
        checkpoint_name = f"checkpoint_epoch_{epoch}_{self.save_counter}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        
        if self.async_save:
            # 
            import threading
            thread = threading.Thread(target=self._async_save, args=(state_dict, checkpoint_path))
            thread.start()
        else:
            torch.save(state_dict, checkpoint_path)
        
        self.save_counter += 1
        return checkpoint_path
    
    def _async_save(self, state_dict: Dict, path: str):
        """checkpoint"""
        torch.save(state_dict, path)
    
    def load(self, model: nn.Module, checkpoint_path: str, 
             optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """
        checkpoint
        
        Args:
            model: 
            checkpoint_path: checkpoint
            optimizer: 
        
        Returns:
            epoch: epoch
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # 
        model.load_state_dict(state_dict['model_state_dict'])
        
        # 
        if optimizer is not None and 'optimizer_state_dict' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        return state_dict.get('epoch', 0)
    
    def list_checkpoints(self) -> List[str]:
        """checkpoint"""
        checkpoints = []
        for f in os.listdir(self.save_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
                checkpoints.append(f)
        return sorted(checkpoints)
    
    def clean_old_checkpoints(self, keep_last: int = 5):
        """checkpoint"""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last:
            old_checkpoints = checkpoints[:-keep_last]
            for checkpoint in old_checkpoints:
                os.remove(os.path.join(self.save_dir, checkpoint))


class GradientFusion:
    """ - """
    
    def __init__(self, fusion_threshold: int = 1024):
        """
        Args:
            fusion_threshold: 
        """
        self.fusion_threshold = fusion_threshold
        self.pending_grads = {}  # 
    
    def add_gradient(self, param_name: str, grad: torch.Tensor):
        """"""
        grad_bytes = grad.element_size() * grad.numel()
        
        if grad_bytes < self.fusion_threshold:
            if param_name not in self.pending_grads:
                self.pending_grads[param_name] = []
            self.pending_grads[param_name].append(grad)
            return False  # 
        
        return True  # 
    
    def fuse_pending(self) -> Dict[str, torch.Tensor]:
        """"""
        fused = {}
        
        for param_name, grads in self.pending_grads.items():
            if len(grads) > 1:
                # 
                fused[param_name] = torch.stack(grads).mean(dim=0)
            elif len(grads) == 1:
                fused[param_name] = grads[0]
        
        self.pending_grads = {}
        return fused
    
    def get_pending_count(self) -> int:
        """"""
        return sum(len(grads) for grads in self.pending_grads.values())


# 
if __name__ == "__main__":
    print("Testing Optimization Modules...")
    
    # 
    compressor = GradientCompressor(compression_ratio=0.1)
    grad = torch.randn(1000)
    compressed, meta = compressor.compress(grad, "test_param")
    decompressed = compressor.decompress(compressed, meta)
    print(f"Gradient Compression: Original={grad.numel()}, Compressed={compressed[0].numel()}")
    
    # 
    ps = ShardedParameterServer(num_shards=2)
    ps.register_parameter("param1", torch.nn.Parameter(torch.randn(10)))
    ps.register_parameter("param2", torch.nn.Parameter(torch.randn(20)))
    ps.register_parameter("param3", torch.nn.Parameter(torch.randn(30)))
    print(f"Sharded PS Info: {ps.get_shard_info()}")
    
    # Checkpoint
    model = nn.Linear(10, 5)
    checkpointer = LightweightCheckpointer(async_save=False)
    path = checkpointer.save(model, epoch=5)
    print(f"Checkpoint saved to: {path}")
    
    # 
    fusion = GradientFusion(fusion_threshold=100)
    grad1 = torch.randn(10)  # 
    grad2 = torch.randn(10)  # 
    fusion.add_gradient("small1", grad1)
    fusion.add_gradient("small2", grad2)
    print(f"Pending grads before fusion: {fusion.get_pending_count()}")
    fused = fusion.fuse_pending()
    print(f"Fused grads count: {len(fused)}")
    
    print("All tests passed!")
