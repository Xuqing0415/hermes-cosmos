"""
PyTorch DDP分布式训练脚本 - 支持全局Checkpoint和容错恢复
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

# 全局配置
CHECKPOINT_INTERVAL = 10  # 每N步保存一次
SHARED_STORAGE = "/shared/checkpoints"
REDIS_HOST = os.environ.get("REDIS_HOST", "redis-service")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

class SimpleModel(nn.Module):
    """简单的线性模型用于测试"""
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
    """初始化分布式环境"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    
    dist.init_process_group(
        backend="gloo",  # 使用gloo支持TCP
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    
    return rank, world_size

def get_redis_client():
    """获取Redis客户端"""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def global_checkpoint(model, optimizer, rank, world_size, step):
    """全局Checkpoint - 所有rank同步保存"""
    r = get_redis_client()
    
    if rank == 0:
        # Rank 0负责协调
        print(f"[Rank {rank}] 发起全局Checkpoint，Step {step}")
        
        # 设置屏障信号
        r.set("checkpoint_barrier", step)
        r.delete(f"checkpoint_done_{step}")
        
        # 保存自己的Checkpoint
        local_path = f"{SHARED_STORAGE}/ckpt_rank{rank}_step{step}.pt"
        torch.save({
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, local_path)
        r.lpush(f"checkpoint_done_{step}", rank)
        
        # 等待所有rank完成
        done_count = 0
        timeout = 60
        start_time = time.time()
        while done_count < world_size and time.time() - start_time < timeout:
            done_count = r.llen(f"checkpoint_done_{step}")
            time.sleep(0.5)
        
        if done_count == world_size:
            # 记录全局Checkpoint元数据
            meta = {
                'step': step,
                'world_size': world_size,
                'timestamp': time.time(),
                'checkpoint_paths': [f"{SHARED_STORAGE}/ckpt_rank{r}_step{step}.pt" for r in range(world_size)]
            }
            r.set(f"global_checkpoint_{step}", json.dumps(meta))
            r.set("latest_global_checkpoint", step)
            print(f"[Rank {rank}] 全局Checkpoint完成，Step {step}")
        else:
            print(f"[Rank {rank}] 全局Checkpoint超时，仅{done_count}/{world_size}完成")
    
    else:
        # 其他rank等待信号
        r = get_redis_client()
        current_barrier = int(r.get("checkpoint_barrier") or 0)
        while current_barrier < step:
            time.sleep(0.5)
            current_barrier = int(r.get("checkpoint_barrier") or 0)
        
        # 保存本地Checkpoint
        local_path = f"{SHARED_STORAGE}/ckpt_rank{rank}_step{step}.pt"
        torch.save({
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, local_path)
        r.lpush(f"checkpoint_done_{step}", rank)
        print(f"[Rank {rank}] 本地Checkpoint完成，Step {step}")

def load_global_checkpoint(model, optimizer, rank):
    """从全局Checkpoint恢复"""
    r = get_redis_client()
    latest_step = int(r.get("latest_global_checkpoint") or 0)
    
    if latest_step == 0:
        print(f"[Rank {rank}] 无全局Checkpoint，从头开始")
        return 0
    
    print(f"[Rank {rank}] 从Step {latest_step}恢复")
    
    # 加载本地Checkpoint
    local_path = f"{SHARED_STORAGE}/ckpt_rank{rank}_step{latest_step}.pt"
    if os.path.exists(local_path):
        checkpoint = torch.load(local_path, map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[Rank {rank}] Checkpoint加载成功")
        return checkpoint['step']
    else:
        print(f"[Rank {rank}] 本地Checkpoint不存在")
        return 0

def check_stop_signal():
    """检查是否需要暂停（等待恢复）"""
    r = get_redis_client()
    stop_signal = r.get("training_paused")
    if stop_signal == "true":
        print("[Rank] 检测到暂停信号，进入等待...")
        while r.get("training_paused") == "true":
            time.sleep(1)
        print("[Rank] 恢复信号收到，继续训练")

def main():
    rank, world_size = init_distributed()
    print(f"[Rank {rank}/{world_size}] 初始化完成")
    
    # 创建模型
    model = SimpleModel()
    model = DDP(model, device_ids=None) if world_size > 1 else model
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 10)
    
    # 检查是否需要恢复
    start_step = load_global_checkpoint(model, optimizer, rank)
    
    # 训练循环
    step = start_step
    while True:
        # 检查暂停信号
        check_stop_signal()
        
        # 前向传播
        output = model(X)
        loss = nn.MSELoss()(output, y)
        
        # 反向传播
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
