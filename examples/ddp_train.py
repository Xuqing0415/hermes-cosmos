"""
PyTorch DDP - Checkpoint
"""

import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import redis
import json

# 
CHECKPOINT_INTERVAL = 10  # N
SHARED_STORAGE = "/shared/checkpoints"
REDIS_HOST = os.environ.get("REDIS_HOST", "redis-service")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

class SimpleModel(nn.Module):
    """"""
    def __init__(self, input_size=10, output_size=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def init_distributed():
    """"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    
    dist.init_process_group(
        backend="gloo",  # glooTCP
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    
    return rank, world_size

def get_redis_client():
    """Redis"""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def global_checkpoint(model, optimizer, rank, world_size, step):
    """Checkpoint - rank"""
    r = get_redis_client()
    
    if rank == 0:
        # Rank 0
        print(f"[Rank {rank}] CheckpointStep {step}")
        
        # 
        r.set("checkpoint_barrier", step)
        r.delete(f"checkpoint_done_{step}")
        
        # Checkpoint
        local_path = f"{SHARED_STORAGE}/ckpt_rank{rank}_step{step}.pt"
        torch.save({
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, local_path)
        r.lpush(f"checkpoint_done_{step}", rank)
        
        # rank
        done_count = 0
        timeout = 60
        start_time = time.time()
        while done_count < world_size and time.time() - start_time < timeout:
            done_count = r.llen(f"checkpoint_done_{step}")
            time.sleep(0.5)
        
        if done_count == world_size:
            # Checkpoint
            meta = {
                'step': step,
                'world_size': world_size,
                'timestamp': time.time(),
                'checkpoint_paths': [f"{SHARED_STORAGE}/ckpt_rank{r}_step{step}.pt" for r in range(world_size)]
            }
            r.set(f"global_checkpoint_{step}", json.dumps(meta))
            r.set("latest_global_checkpoint", step)
            print(f"[Rank {rank}] CheckpointStep {step}")
        else:
            print(f"[Rank {rank}] Checkpoint{done_count}/{world_size}")
    
    else:
        # rank
        r = get_redis_client()
        current_barrier = int(r.get("checkpoint_barrier") or 0)
        while current_barrier < step:
            time.sleep(0.5)
            current_barrier = int(r.get("checkpoint_barrier") or 0)
        
        # Checkpoint
        local_path = f"{SHARED_STORAGE}/ckpt_rank{rank}_step{step}.pt"
        torch.save({
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, local_path)
        r.lpush(f"checkpoint_done_{step}", rank)
        print(f"[Rank {rank}] CheckpointStep {step}")

def load_global_checkpoint(model, optimizer, rank):
    """Checkpoint"""
    r = get_redis_client()
    latest_step = int(r.get("latest_global_checkpoint") or 0)
    
    if latest_step == 0:
        print(f"[Rank {rank}] Checkpoint")
        return 0
    
    print(f"[Rank {rank}] Step {latest_step}")
    
    # Checkpoint
    local_path = f"{SHARED_STORAGE}/ckpt_rank{rank}_step{latest_step}.pt"
    if os.path.exists(local_path):
        checkpoint = torch.load(local_path, map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[Rank {rank}] Checkpoint")
        return checkpoint['step']
    else:
        print(f"[Rank {rank}] Checkpoint")
        return 0

def check_stop_signal():
    """"""
    r = get_redis_client()
    stop_signal = r.get("training_paused")
    if stop_signal == "true":
        print("[Rank] ...")
        while r.get("training_paused") == "true":
            time.sleep(1)
        print("[Rank] ")

def main():
    rank, world_size = init_distributed()
    print(f"[Rank {rank}/{world_size}] ")
    
    # 
    model = SimpleModel()
    model = DDP(model, device_ids=None) if world_size > 1 else model
    
    # 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 
    X = torch.randn(32, 10)
    y = torch.randn(32, 10)
    
    # 
    start_step = load_global_checkpoint(model, optimizer, rank)
    
    # 
    step = start_step
    while True:
        # 
        check_stop_signal()
        
        # 
        output = model(X)
        loss = nn.MSELoss()(output, y)
        
        # 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        if step % CHECKPOINT_INTERVAL == 0:
            global_checkpoint(model, optimizer, rank, world_size, step)
        
        if rank == 0:
            print(f"[Rank 0] Step {step}, Loss: {loss.item():.4f}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
