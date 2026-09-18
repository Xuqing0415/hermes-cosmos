"""
跨Region分布式训练脚本
支持高延迟网络、异步Checkpoint、区域感知
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

# 全局配置
CHECKPOINT_INTERVAL = 20  # 跨Region增加间隔
ASYNC_CHECKPOINT = True  # 异步Checkpoint
REGION = os.environ.get("REGION", "unknown")
REDIS_HOST = os.environ.get("REDIS_HOST", "global-redis-service")

class CrossRegionModel(nn.Module):
    """跨Region训练模型"""
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
    """初始化跨Region分布式环境"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 从环境变量获取所有rank的地址
    rank_addrs_str = os.environ.get("RANK_ADDRS", "")
    
    print(f"[Region {REGION}] Rank {rank}/{world_size} 初始化")
    print(f"[Region {REGION}] 所有Rank地址: {rank_addrs_str}")
    
    # 解析地址列表
    rank_addrs = {}
    for item in rank_addrs_str.split(","):
        if ":" in item:
            r, addr = item.split(":", 1)
            rank_addrs[int(r)] = addr
    
    # 使用Gloo后端（支持TCP跨Region）
    master_addr = rank_addrs.get(0, "localhost")
    
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:29500",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=300)  # 跨Region增加超时
    )
    
    return rank, world_size

def get_global_redis():
    """获取全局Redis连接"""
    return redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def async_global_checkpoint(model, optimizer, rank, world_size, step):
    """异步全局Checkpoint"""
    r = get_global_redis()
    
    # 保存本地Checkpoint到全局存储
    checkpoint_data = {
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'region': REGION,
        'rank': rank,
        'timestamp': time.time()
    }
    
    # 序列化并存储到Redis（生产环境用对象存储）
    import pickle
    checkpoint_key = f"checkpoint:{os.environ['JOB_ID']}:{REGION}:{rank}:{step}"
    r.setex(checkpoint_key, 3600, pickle.dumps(checkpoint_data))
    
    # 上报元数据到全局协调器
    r.hset(f"checkpoint_meta:{os.environ['JOB_ID']}:{step}", REGION, rank)
    
    # 检查是否所有Region都完成了该step的Checkpoint
    meta = r.hgetall(f"checkpoint_meta:{os.environ['JOB_ID']}:{step}")
    
    if len(meta) == world_size:
        # 所有Region完成，标记为全局Checkpoint
        r.set(f"global_checkpoint_{os.environ['JOB_ID']}_latest", step)
        if rank == 0:
            print(f"[Region {REGION}] 全局Checkpoint完成，Step {step}")
    
    print(f"[Region {REGION}] Rank {rank} Checkpoint保存完成，Step {step}")

def load_global_checkpoint(model, optimizer, rank):
    """从全局Checkpoint恢复"""
    r = get_global_redis()
    job_id = os.environ.get("JOB_ID")
    
    latest_step = int(r.get(f"global_checkpoint_{job_id}_latest") or 0)
    
    if latest_step == 0:
        print(f"[Region {REGION}] 无全局Checkpoint，从头开始")
        return 0
    
    print(f"[Region {REGION}] 从Step {latest_step}恢复")
    
    # 加载本地Checkpoint
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
        print(f"[Region {REGION}] Checkpoint加载成功")
        return checkpoint['step']
    
    return 0

def check_region_health():
    """检查Region健康状态"""
    r = get_global_redis()
    
    # 发送心跳
    r.setex(f"heartbeat:{REGION}:{os.environ.get('RANK')}", 60, time.time())
    
    # 检查其他Region
    healthy_regions = []
    for key in r.scan_iter("heartbeat:*"):
        region = key.split(":")[1]
        if region not in healthy_regions:
            healthy_regions.append(region)
    
    return healthy_regions

def main():
    import datetime
    
    rank, world_size = init_cross_region_distributed()
    
    print(f"[Region {REGION}] 分布式环境初始化完成")
    
    # 创建模型
    model = CrossRegionModel()
    model = DDP(model, device_ids=None)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 检查是否需要恢复
    start_step = load_global_checkpoint(model, optimizer, rank)
    
    # 创建数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 10)
    
    # 训练循环
    step = start_step
    last_checkpoint_time = time.time()
    
    while True:
        # 检查Region健康
        healthy_regions = check_region_health()
        
        if len(healthy_regions) < 2:
            print(f"[Region {REGION}] 警告：仅有 {len(healthy_regions)} 个Region健康")
        
        # 前向传播
        output = model(X)
        loss = nn.MSELoss()(output, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        # 异步Checkpoint（每个Region独立保存）
        current_time = time.time()
        if current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL:
            async_global_checkpoint(model, optimizer, rank, world_size, step)
            last_checkpoint_time = current_time
        
        if rank == 0:
            print(f"[Region {REGION}] Step {step}, Loss: {loss.item():.4f}")
        
        # 模拟跨Region延迟
        time.sleep(0.1)

if __name__ == "__main__":
    main()
