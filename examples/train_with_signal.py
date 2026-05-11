"""
Hermes训练脚本 - 支持主动迁移信号处理
收到SIGUSR1信号时优雅保存Checkpoint并退出
"""

import os
import sys
import signal
import time
import torch
import torch.nn as nn
import torch.optim as optim

# 全局标志
should_exit = False
checkpoint_saved = False

class SimpleModel(nn.Module):
    """简单训练模型"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def save_checkpoint(model, optimizer, step):
    """保存Checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'timestamp': time.time()
    }
    
    checkpoint_path = f"/shared/checkpoints/hermes_checkpoint_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"[CHECKPOINT] Checkpoint saved at step {step}")
    
    return checkpoint_path

def handle_migration_signal(sig, frame):
    """处理迁移信号SIGUSR1"""
    global should_exit, checkpoint_saved
    
    if sig == signal.SIGUSR1:
        print("\n[MIGRATION] 收到主动迁移信号 (SIGUSR1)")
        should_exit = True
        
        if not checkpoint_saved:
            print("[MIGRATION] 正在保存Checkpoint...")
            checkpoint_saved = True
        else:
            print("[MIGRATION] Checkpoint已保存")

def handle_sigterm(sig, frame):
    """处理SIGTERM信号"""
    print("\n[TERMINATION] 收到终止信号，优雅退出")
    sys.exit(0)

def main():
    global should_exit
    
    # 注册信号处理
    signal.signal(signal.SIGUSR1, handle_migration_signal)
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    print("[TRAINING] 启动训练，支持主动迁移信号")
    print("[TRAINING] 等待 SIGUSR1 信号进行优雅迁移")
    
    # 创建模型和优化器
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 检查是否需要恢复
    resume_step = 0
    resume_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[TRAINING] 从Checkpoint恢复: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_step = checkpoint['step']
        print(f"[TRAINING] 恢复到step {resume_step}")
    
    # 创建模拟数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 10)
    
    # 训练循环
    step = resume_step
    while True:
        # 检查退出标志
        if should_exit:
            print("[TRAINING] 主动迁移触发，保存最终Checkpoint")
            save_checkpoint(model, optimizer, step)
            print("[TRAINING] 优雅退出，等待新Pod启动")
            sys.exit(0)
        
        # 训练步骤
        optimizer.zero_grad()
        output = model(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        step += 1
        
        # 定期保存Checkpoint
        if step % 10 == 0:
            save_checkpoint(model, optimizer, step)
        
        if step % 100 == 0:
            print(f"[TRAINING] Step {step}, Loss: {loss.item():.4f}")
        
        time.sleep(0.1)  # 模拟训练时间

if __name__ == "__main__":
    main()
