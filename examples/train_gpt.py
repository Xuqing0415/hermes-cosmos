"""
真实的 AI 训练作业示例 - NanoGPT 风格的小型语言模型训练
适合在 16-64 GPU 上运行，验证 Hermes 的调度和容错能力

这是一个简化版的 GPT-like 模型训练脚本，用于测试：
1. 多卡分布式训练
2. Hermes Checkpoint 集成
3. 故障恢复能力
"""

import os
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Hermes SDK 集成
try:
    from hermes.checkpoint import HermesCheckpointer
    HERMES_AVAILABLE = True
except ImportError:
    HERMES_AVAILABLE = False

# 模型配置
class GPTConfig:
    vocab_size = 50257
    n_layer = 12
    n_head = 12
    n_embd = 768
    block_size = 1024
    dropout = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

def get_batch(split, device):
    """生成随机训练数据"""
    data = torch.randint(0, 50257, (64, 1024))
    x = data[:, :-1].to(device)
    y = data[:, 1:].to(device)
    return x, y

def train():
    # 初始化分布式
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device = f'cuda:{rank}'
    
    # 创建模型
    config = GPTConfig()
    model = GPT(config).to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Hermes Checkpointer
    checkpoint_interval = int(os.environ.get('HERMES_CHECKPOINT_INTERVAL', '300'))
    job_id = os.environ.get('HERMES_JOB_ID', 'test-job')
    
    if HERMES_AVAILABLE:
        checkpointer = HermesCheckpointer(job_id=job_id)
        print(f"[Rank {rank}] Hermes Checkpointer initialized")
    
    # 尝试从Checkpoint恢复
    start_epoch = 0
    if HERMES_AVAILABLE:
        try:
            state = checkpointer.load()
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch']
            print(f"[Rank {rank}] Recovered from checkpoint, starting at epoch {start_epoch}")
        except Exception as e:
            print(f"[Rank {rank}] No checkpoint found, starting fresh: {e}")
    
    # 训练循环
    total_epochs = 10
    batch_size = 8
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        num_batches = 100  # 每epoch训练步数
        
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            x, y = get_batch('train', device)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if rank == 0 and batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{total_epochs}] Batch {batch_idx}/{num_batches} Loss: {loss.item():.4f}")
        
        # Checkpoint保存（仅rank 0）
        if rank == 0 and HERMES_AVAILABLE:
            if (epoch + 1) % (checkpoint_interval // 60) == 0:
                checkpointer.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': total_loss / num_batches,
                })
                print(f"[Rank {rank}] Checkpoint saved at epoch {epoch+1}")
        
        # 同步
        dist.barrier()
    
    dist.destroy_process_group()
    print(f"[Rank {rank}] Training completed!")

if __name__ == '__main__':
    train()
