"""
Hermes - 
SIGUSR1Checkpoint
"""

import os
import sys
import signal
import time
import torch
import torch.nn as nn
import torch.optim as optim

# 
should_exit = False
checkpoint_saved = False

class SimpleModel(nn.Module):
    """"""
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
    """Checkpoint"""
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
    """SIGUSR1"""
    global should_exit, checkpoint_saved
    
    if sig == signal.SIGUSR1:
        print("\n[MIGRATION]  (SIGUSR1)")
        should_exit = True
        
        if not checkpoint_saved:
            print("[MIGRATION] Checkpoint...")
            checkpoint_saved = True
        else:
            print("[MIGRATION] Checkpoint")

def handle_sigterm(sig, frame):
    """SIGTERM"""
    print("\n[TERMINATION] ")
    sys.exit(0)

def main():
    global should_exit
    
    # 
    signal.signal(signal.SIGUSR1, handle_migration_signal)
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    print("[TRAINING] ")
    print("[TRAINING]  SIGUSR1 ")
    
    # 
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 
    resume_step = 0
    resume_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[TRAINING] Checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_step = checkpoint['step']
        print(f"[TRAINING] step {resume_step}")
    
    # 
    X = torch.randn(32, 10)
    y = torch.randn(32, 10)
    
    # 
    step = resume_step
    while True:
        # 
        if should_exit:
            print("[TRAINING] Checkpoint")
            save_checkpoint(model, optimizer, step)
            print("[TRAINING] Pod")
            sys.exit(0)
        
        # 
        optimizer.zero_grad()
        output = model(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        step += 1
        
        # Checkpoint
        if step % 10 == 0:
            save_checkpoint(model, optimizer, step)
        
        if step % 100 == 0:
            print(f"[TRAINING] Step {step}, Loss: {loss.item():.4f}")
        
        time.sleep(0.1)  # 

if __name__ == "__main__":
    main()
