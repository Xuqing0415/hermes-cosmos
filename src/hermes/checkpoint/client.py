"""
Hermes Checkpoint客户端 - 为PyTorch训练脚本提供自动Checkpoint功能
"""

import asyncio
import torch
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import requests

@dataclass
class CheckpointConfig:
    """Checkpoint配置"""
    interval: int = 100  # 每N步保存一次
    backend: str = "redis"  # redis, local
    redis_url: str = "redis://localhost:6379/0"
    server_url: str = "http://localhost:50052"
    enabled: bool = True

class HermesCheckpointer:
    """Hermes Checkpointer客户端"""
    
    def __init__(self, job_id: str, config: Optional[CheckpointConfig] = None):
        self.job_id = job_id
        self.config = config or CheckpointConfig()
        self.step_count = 0
        self.last_checkpoint_id = None
        self._model_hooks = []
    
    def save(self, data: Dict[str, Any]) -> str:
        """同步保存Checkpoint"""
        return asyncio.run(self.async_save(data))
    
    async def async_save(self, data: Dict[str, Any]) -> str:
        """异步保存Checkpoint"""
        if not self.config.enabled:
            return ""
        
        self.step_count += 1
        
        # 检查是否达到保存间隔
        if self.step_count % self.config.interval != 0:
            return ""
        
        try:
            # 构建请求数据
            payload = {
                "job_id": self.job_id,
                "step": self.step_count,
                "data": data,
                "delta_from": self.last_checkpoint_id
            }
            
            # 发送请求到Checkpoint服务
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                requests.post,
                f"{self.config.server_url}/checkpoints",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                self.last_checkpoint_id = result.get("checkpoint_id")
                return self.last_checkpoint_id
            else:
                print(f"Checkpoint保存失败: {response.text}")
                return ""
        except Exception as e:
            print(f"Checkpoint保存异常: {e}")
            return ""
    
    def load(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """同步加载Checkpoint"""
        return asyncio.run(self.async_load(checkpoint_id))
    
    async def async_load(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """异步加载Checkpoint"""
        try:
            if checkpoint_id is None:
                # 获取最新的checkpoint
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    requests.get,
                    f"{self.config.server_url}/checkpoints/latest/{self.job_id}"
                )
                
                if response.status_code == 200:
                    checkpoint_id = response.json().get("checkpoint_id")
                else:
                    return {}
            else:
                # 获取指定的checkpoint
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    requests.get,
                    f"{self.config.server_url}/checkpoints/{checkpoint_id}"
                )
                
                if response.status_code == 200:
                    return response.json().get("data", {})
                
            return {}
        except Exception as e:
            print(f"Checkpoint加载异常: {e}")
            return {}
    
    def wrap_model(self, model: torch.nn.Module):
        """包装模型，自动在forward后保存Checkpoint"""
        def hook(module, input, output):
            self.save({"model": model.state_dict()})
        
        model.register_forward_hook(hook)
        self._model_hooks.append(hook)
        return model
    
    def close(self):
        """清理资源"""
        for hook in self._model_hooks:
            hook.remove()

# 便捷函数：启用自动Checkpoint
def enable_checkpointing(job_id: str, interval: int = 100, 
                        backend: str = "redis", server_url: str = "http://localhost:50052"):
    """
    启用自动Checkpoint功能
    
    在用户训练脚本中只需调用：
    from hermes_checkpoint import enable_checkpointing
    
    enable_checkpointing(job_id="my-training-job", interval=100)
    
    之后训练代码无需改动，每100步自动保存
    """
    config = CheckpointConfig(
        interval=interval,
        backend=backend,
        server_url=server_url
    )
    checkpointer = HermesCheckpointer(job_id, config)
    
    # 注入到全局命名空间（方便用户使用）
    import builtins
    builtins.__dict__['hermes_checkpointer'] = checkpointer
    
    # 返回checkpointer供高级用户使用
    return checkpointer

# 示例用法
if __name__ == "__main__":
    # 创建checkpointer
    checkpointer = HermesCheckpointer("test-job")
    
    # 模拟训练循环
    for step in range(500):
        # 训练逻辑...
        model_state = {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)}
        optimizer_state = {"lr": 0.001, "step": step}
        
        # 手动保存
        cp_id = checkpointer.save({
            "model": model_state,
            "optimizer": optimizer_state,
            "step": step
        })
        
        if cp_id:
            print(f"Step {step}: Checkpoint saved as {cp_id}")
    
    # 加载最新checkpoint
    loaded = checkpointer.load()
    print(f"加载的Checkpoint: {loaded.keys()}")
