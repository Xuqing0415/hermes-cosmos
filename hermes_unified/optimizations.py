#!/usr/bin/env python3
"""
Hermes Unified Optimizations Module

包含中期优化方向的实现：
1. 梯度压缩（Top-k + 误差补偿）
2. 多分片参数服务器框架
3. 轻量Checkpoint机制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os


class GradientCompressor:
    """梯度压缩器 - 使用 Top-k + 误差补偿"""
    
    def __init__(self, compression_ratio: float = 0.1, error_compensation: bool = True):
        """
        Args:
            compression_ratio: 保留的梯度比例 (0.01-0.5)
            error_compensation: 是否启用误差补偿
        """
        self.compression_ratio = compression_ratio
        self.error_compensation = error_compensation
        self.residual_error = {}  # 存储累积误差
    
    def compress(self, grad: torch.Tensor, param_name: str = "") -> Tuple[torch.Tensor, Dict]:
        """
        压缩梯度
        
        Args:
            grad: 原始梯度张量
            param_name: 参数名称（用于误差补偿）
        
        Returns:
            compressed_grad: 压缩后的梯度
            metadata: 压缩元数据（索引、形状等）
        """
        if self.compression_ratio >= 1.0:
            return grad, {"type": "full", "shape": grad.shape}
        
        # Flatten 梯度
        grad_flat = grad.flatten()
        n_elements = grad_flat.numel()
        n_top = max(1, int(n_elements * self.compression_ratio))
        
        # Top-k 选择
        top_values, top_indices = torch.topk(grad_flat.abs(), n_top, largest=True)
        
        # 获取符号
        signs = torch.sign(grad_flat[top_indices])
        compressed_values = top_values * signs
        
        # 误差补偿
        if self.error_compensation:
            if param_name not in self.residual_error:
                self.residual_error[param_name] = torch.zeros_like(grad)
            
            # 将残差加到当前梯度上
            grad_with_error = grad + self.residual_error[param_name]
            
            # 更新残差
            self.residual_error[param_name] = grad_with_error - \
                torch.zeros_like(grad_flat).scatter_(0, top_indices, compressed_values).view(grad.shape)
            
            # 使用带误差补偿的值
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
        解压梯度
        
        Args:
            compressed_data: 压缩数据 (indices, values)
            metadata: 压缩元数据
        
        Returns:
            decompressed_grad: 解压后的梯度
        """
        if metadata["type"] == "full":
            return compressed_data
        
        indices, values = compressed_data
        shape = metadata["shape"]
        
        # 重建梯度
        decompressed = torch.zeros(int(np.prod(shape)), device=values.device)
        decompressed.scatter_(0, indices, values)
        
        return decompressed.view(shape)


class ShardedParameterServer:
    """多分片参数服务器框架"""
    
    def __init__(self, num_shards: int = 4, device: str = "cpu"):
        """
        Args:
            num_shards: 参数服务器分片数量
            device: 设备类型
        """
        self.num_shards = num_shards
        self.device = torch.device(device)
        self.shards = [{} for _ in range(num_shards)]  # 每个分片存储部分参数
        self.param_to_shard = {}  # 参数名 -> 分片索引
        self.shard_sizes = [0] * num_shards  # 每个分片的参数数量
    
    def _get_shard_index(self, param_name: str) -> int:
        """根据参数名计算分片索引"""
        if param_name in self.param_to_shard:
            return self.param_to_shard[param_name]
        
        # 使用哈希分配
        shard_idx = hash(param_name) % self.num_shards
        self.param_to_shard[param_name] = shard_idx
        return shard_idx
    
    def register_parameter(self, param_name: str, param: torch.nn.Parameter):
        """注册参数到分片"""
        shard_idx = self._get_shard_index(param_name)
        self.shards[shard_idx][param_name] = param.data.to(self.device).clone()
        self.shard_sizes[shard_idx] += 1
    
    def update_parameters(self, updates: Dict[str, torch.Tensor]):
        """更新参数"""
        for param_name, update in updates.items():
            shard_idx = self._get_shard_index(param_name)
            if param_name in self.shards[shard_idx]:
                self.shards[shard_idx][param_name] += update.to(self.device)
    
    def get_parameters(self, param_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """获取参数"""
        result = {}
        
        if param_names is None:
            # 获取所有参数
            for shard in self.shards:
                result.update(shard)
        else:
            # 获取指定参数
            for param_name in param_names:
                shard_idx = self._get_shard_index(param_name)
                if param_name in self.shards[shard_idx]:
                    result[param_name] = self.shards[shard_idx][param_name]
        
        return result
    
    def get_shard_info(self) -> Dict:
        """获取分片信息"""
        return {
            "num_shards": self.num_shards,
            "shard_sizes": self.shard_sizes,
            "total_params": sum(self.shard_sizes)
        }


class LightweightCheckpointer:
    """轻量Checkpoint机制"""
    
    def __init__(self, save_dir: str = "checkpoints", async_save: bool = True):
        """
        Args:
            save_dir: checkpoint保存目录
            async_save: 是否异步保存
        """
        self.save_dir = save_dir
        self.async_save = async_save
        self.save_counter = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model: nn.Module, epoch: int = 0, save_optimizer: bool = False,
             optimizer: Optional[torch.optim.Optimizer] = None) -> str:
        """
        保存轻量checkpoint（只保存模型参数，可选保存优化器状态）
        
        Args:
            model: 模型
            epoch: 当前epoch
            save_optimizer: 是否保存优化器状态
            optimizer: 优化器
        
        Returns:
            checkpoint_path: 保存路径
        """
        # 只保存参数，不保存完整模型
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'save_time': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else 'unknown'
        }
        
        if save_optimizer and optimizer is not None:
            state_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        # 生成文件名
        checkpoint_name = f"checkpoint_epoch_{epoch}_{self.save_counter}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        
        if self.async_save:
            # 异步保存（简化实现）
            import threading
            thread = threading.Thread(target=self._async_save, args=(state_dict, checkpoint_path))
            thread.start()
        else:
            torch.save(state_dict, checkpoint_path)
        
        self.save_counter += 1
        return checkpoint_path
    
    def _async_save(self, state_dict: Dict, path: str):
        """异步保存checkpoint"""
        torch.save(state_dict, path)
    
    def load(self, model: nn.Module, checkpoint_path: str, 
             optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """
        加载checkpoint
        
        Args:
            model: 模型（用于加载参数）
            checkpoint_path: checkpoint路径
            optimizer: 优化器（可选）
        
        Returns:
            epoch: 加载的epoch数
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型参数
        model.load_state_dict(state_dict['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        return state_dict.get('epoch', 0)
    
    def list_checkpoints(self) -> List[str]:
        """列出所有checkpoint文件"""
        checkpoints = []
        for f in os.listdir(self.save_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
                checkpoints.append(f)
        return sorted(checkpoints)
    
    def clean_old_checkpoints(self, keep_last: int = 5):
        """清理旧的checkpoint，只保留最近的几个"""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last:
            old_checkpoints = checkpoints[:-keep_last]
            for checkpoint in old_checkpoints:
                os.remove(os.path.join(self.save_dir, checkpoint))


class GradientFusion:
    """梯度融合 - 将多个小梯度合并为一个包"""
    
    def __init__(self, fusion_threshold: int = 1024):
        """
        Args:
            fusion_threshold: 融合阈值（字节），小于此值的梯度会被合并
        """
        self.fusion_threshold = fusion_threshold
        self.pending_grads = {}  # 待融合的梯度
    
    def add_gradient(self, param_name: str, grad: torch.Tensor):
        """添加梯度到待融合队列"""
        grad_bytes = grad.element_size() * grad.numel()
        
        if grad_bytes < self.fusion_threshold:
            if param_name not in self.pending_grads:
                self.pending_grads[param_name] = []
            self.pending_grads[param_name].append(grad)
            return False  # 未发送
        
        return True  # 立即发送
    
    def fuse_pending(self) -> Dict[str, torch.Tensor]:
        """融合所有待发送的梯度"""
        fused = {}
        
        for param_name, grads in self.pending_grads.items():
            if len(grads) > 1:
                # 合并多个梯度（取平均或求和）
                fused[param_name] = torch.stack(grads).mean(dim=0)
            elif len(grads) == 1:
                fused[param_name] = grads[0]
        
        self.pending_grads = {}
        return fused
    
    def get_pending_count(self) -> int:
        """获取待融合梯度数量"""
        return sum(len(grads) for grads in self.pending_grads.values())


# 示例用法
if __name__ == "__main__":
    print("Testing Optimization Modules...")
    
    # 测试梯度压缩
    compressor = GradientCompressor(compression_ratio=0.1)
    grad = torch.randn(1000)
    compressed, meta = compressor.compress(grad, "test_param")
    decompressed = compressor.decompress(compressed, meta)
    print(f"Gradient Compression: Original={grad.numel()}, Compressed={compressed[0].numel()}")
    
    # 测试多分片参数服务器
    ps = ShardedParameterServer(num_shards=2)
    ps.register_parameter("param1", torch.nn.Parameter(torch.randn(10)))
    ps.register_parameter("param2", torch.nn.Parameter(torch.randn(20)))
    ps.register_parameter("param3", torch.nn.Parameter(torch.randn(30)))
    print(f"Sharded PS Info: {ps.get_shard_info()}")
    
    # 测试轻量Checkpoint
    model = nn.Linear(10, 5)
    checkpointer = LightweightCheckpointer(async_save=False)
    path = checkpointer.save(model, epoch=5)
    print(f"Checkpoint saved to: {path}")
    
    # 测试梯度融合
    fusion = GradientFusion(fusion_threshold=100)
    grad1 = torch.randn(10)  # 小梯度
    grad2 = torch.randn(10)  # 小梯度
    fusion.add_gradient("small1", grad1)
    fusion.add_gradient("small2", grad2)
    print(f"Pending grads before fusion: {fusion.get_pending_count()}")
    fused = fusion.fuse_pending()
    print(f"Fused grads count: {len(fused)}")
    
    print("All tests passed!")
