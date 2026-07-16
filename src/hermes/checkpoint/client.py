"""
Hermes Checkpoint - PyTorchCheckpoint
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
    """Checkpoint"""
    interval: int = 100  # N
    backend: str = "redis"  # redis, local
    redis_url: str = "redis://localhost:6379/0"
    server_url: str = "http://localhost:50052"
    enabled: bool = True

class HermesCheckpointer:
    """Hermes Checkpointer"""
    
    def __init__(self, job_id: str, config: Optional[CheckpointConfig] = None):
        self.job_id = job_id
        self.config = config or CheckpointConfig()
        self.step_count = 0
        self.last_checkpoint_id = None
        self._model_hooks = []
    
    def save(self, data: Dict[str, Any]) -> str:
        """Checkpoint"""
        return asyncio.run(self.async_save(data))
    
    async def async_save(self, data: Dict[str, Any]) -> str:
        """Checkpoint"""
        if not self.config.enabled:
            return ""
        
        self.step_count += 1
        
        # 
        if self.step_count % self.config.interval != 0:
            return ""
        
        try:
            # 
            payload = {
                "job_id": self.job_id,
                "step": self.step_count,
                "data": data,
                "delta_from": self.last_checkpoint_id
            }
            
            # Checkpoint
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
                print(f"Checkpoint: {response.text}")
                return ""
        except Exception as e:
            print(f"Checkpoint: {e}")
            return ""
    
    def load(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Checkpoint"""
        return asyncio.run(self.async_load(checkpoint_id))
    
    async def async_load(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Checkpoint"""
        try:
            if checkpoint_id is None:
                # checkpoint
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
                # checkpoint
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    requests.get,
                    f"{self.config.server_url}/checkpoints/{checkpoint_id}"
                )
                
                if response.status_code == 200:
                    return response.json().get("data", {})
                
            return {}
        except Exception as e:
            print(f"Checkpoint: {e}")
            return {}
    
    def wrap_model(self, model: torch.nn.Module):
        """forwardCheckpoint"""
        def hook(module, input, output):
            self.save({"model": model.state_dict()})
        
        model.register_forward_hook(hook)
        self._model_hooks.append(hook)
        return model
    
    def close(self):
        """"""
        for hook in self._model_hooks:
            hook.remove()

# Checkpoint
def enable_checkpointing(job_id: str, interval: int = 100, 
                        backend: str = "redis", server_url: str = "http://localhost:50052"):
    """
    Checkpoint
    
    
    from hermes_checkpoint import enable_checkpointing
    
    enable_checkpointing(job_id="my-training-job", interval=100)
    
    100
    """
    config = CheckpointConfig(
        interval=interval,
        backend=backend,
        server_url=server_url
    )
    checkpointer = HermesCheckpointer(job_id, config)
    
    # 
    import builtins
    builtins.__dict__['hermes_checkpointer'] = checkpointer
    
    # checkpointer
    return checkpointer

# 
if __name__ == "__main__":
    # checkpointer
    checkpointer = HermesCheckpointer("test-job")
    
    # 
    for step in range(500):
        # ...
        model_state = {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)}
        optimizer_state = {"lr": 0.001, "step": step}
        
        # 
        cp_id = checkpointer.save({
            "model": model_state,
            "optimizer": optimizer_state,
            "step": step
        })
        
        if cp_id:
            print(f"Step {step}: Checkpoint saved as {cp_id}")
    
    # checkpoint
    loaded = checkpointer.load()
    print(f"Checkpoint: {loaded.keys()}")
