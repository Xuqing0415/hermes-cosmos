"""
Region
Checkpoint
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
import asyncio

# 
CHECKPOINT_INTERVAL = 20  # Region
ASYNC_CHECKPOINT = True  # Checkpoint
REGION = os.environ.get("REGION", "unknown")
REDIS_HOST = os.environ.get("REDIS_HOST", "global-redis-service")

class CrossRegionModel(nn.Module):
    """Region"""
    def __init__(self):
        super(CrossRegionModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def init_cross_region_distributed():
    """Region"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # rank
    rank_addrs_str = os.environ.get("RANK_ADDRS", "")
    
    print(f"[Region {REGION}] Rank {rank}/{world_size} ")
    print(f"[Region {REGION}] Rank: {rank_addrs_str}")
    
    # 
    rank_addrs = {}
    for item in rank_addrs_str.split(","):
        if ":" in item:
            r, addr = item.split(":", 1)
            rank_addrs[int(r)] = addr
    
    # GlooTCPRegion
    master_addr = rank_addrs.get(0, "localhost")
    
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:29500",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=300)  # Region
    )
    
    return rank, world_size

def get_global_redis():
    """Redis"""
    return redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def async_global_checkpoint(model, optimizer, rank, world_size, step):
    """Checkpoint"""
    r = get_global_redis()
    
    # Checkpoint
    checkpoint_data = {
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'region': REGION,
        'rank': rank,
        'timestamp': time.time()
    }
    
    # Redis
    import pickle
    checkpoint_key = f"checkpoint:{os.environ['JOB_ID']}:{REGION}:{rank}:{step}"
    r.setex(checkpoint_key, 3600, pickle.dumps(checkpoint_data))
    
    # 
    r.hset(f"checkpoint_meta:{os.environ['JOB_ID']}:{step}", REGION, rank)
    
    # RegionstepCheckpoint
    meta = r.hgetall(f"checkpoint_meta:{os.environ['JOB_ID']}:{step}")
    
    if len(meta) == world_size:
        # RegionCheckpoint
        r.set(f"global_checkpoint_{os.environ['JOB_ID']}_latest", step)
        if rank == 0:
            print(f"[Region {REGION}] CheckpointStep {step}")
    
    print(f"[Region {REGION}] Rank {rank} CheckpointStep {step}")

def load_global_checkpoint(model, optimizer, rank):
    """Checkpoint"""
    r = get_global_redis()
    job_id = os.environ.get("JOB_ID")
    
    latest_step = int(r.get(f"global_checkpoint_{job_id}_latest") or 0)
    
    if latest_step == 0:
        print(f"[Region {REGION}] Checkpoint")
        return 0
    
    print(f"[Region {REGION}] Step {latest_step}")
    
    # Checkpoint
    import pickle
    checkpoint_key = f"checkpoint:{job_id}:{REGION}:{rank}:{latest_step}"
    checkpoint_data = r.get(checkpoint_key)
    
    if checkpoint_data:
        checkpoint = pickle.loads(checkpoint_data)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[Region {REGION}] Checkpoint")
        return checkpoint['step']
    
    return 0

def check_region_health():
    """Region"""
    r = get_global_redis()
    
    # 
    r.setex(f"heartbeat:{REGION}:{os.environ.get('RANK')}", 60, time.time())
    
    # Region
    healthy_regions = []
    for key in r.scan_iter("heartbeat:*"):
        region = key.split(":")[1]
        if region not in healthy_regions:
            healthy_regions.append(region)
    
    return healthy_regions

def main():
    import datetime
    
    rank, world_size = init_cross_region_distributed()
    
    print(f"[Region {REGION}] ")
    
    # 
    model = CrossRegionModel()
    model = DDP(model, device_ids=None)
    
    # 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 
    start_step = load_global_checkpoint(model, optimizer, rank)
    
    # 
    X = torch.randn(32, 10)
    y = torch.randn(32, 10)
    
    # 
    step = start_step
    last_checkpoint_time = time.time()
    
    while True:
        # Region
        healthy_regions = check_region_health()
        
        if len(healthy_regions) < 2:
            print(f"[Region {REGION}]  {len(healthy_regions)} Region")
        
        # 
        output = model(X)
        loss = nn.MSELoss()(output, y)
        
        # 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        # CheckpointRegion
        current_time = time.time()
        if current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL:
            async_global_checkpoint(model, optimizer, rank, world_size, step)
            last_checkpoint_time = current_time
        
        if rank == 0:
            print(f"[Region {REGION}] Step {step}, Loss: {loss.item():.4f}")
        
        # Region
        time.sleep(0.1)

if __name__ == "__main__":
    main()
