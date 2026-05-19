"""
Knowledge Distillation for Federated Transfer Learning

Implements federated knowledge distillation methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional


class DistillationLoss(nn.Module):
    """Distillation loss combining hard and soft labels."""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                hard_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        soft_labels = F.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        distill_loss = F.kl_div(student_probs, soft_labels, reduction='batchmean') * (self.temperature ** 2)
        
        if hard_labels is not None:
            hard_loss = F.cross_entropy(student_logits, hard_labels)
            return self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return distill_loss


class TeacherStudentTrainer:
    """Trainer for teacher-student distillation."""
    
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, loss_fn: DistillationLoss = None):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.loss_fn = loss_fn or DistillationLoss()
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-3)
    
    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        self.student_model.train()
        self.teacher_model.eval()
        
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
        
        student_logits = self.student_model(x)
        loss = self.loss_fn(student_logits, teacher_logits, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class FedDistillationServer:
    """Server for federated knowledge distillation."""
    
    def __init__(self, teacher_model):
        self.teacher_model = teacher_model
        self.teacher_model.eval()
    
    def get_teacher_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.teacher_model.state_dict().items()}
    
    def aggregate_student_weights(self, student_weights_list: List[Dict[str, torch.Tensor]],
                                  weights: Optional[List[float]] = None):
        if not student_weights_list:
            return {}
        
        if weights is None:
            weights = [1.0 / len(student_weights_list)] * len(student_weights_list)
        
        aggregated = {}
        for key in student_weights_list[0].keys():
            aggregated[key] = sum(w * student_weights_list[i][key] for i, w in enumerate(weights))
        
        return aggregated


class FedDistillationClient:
    """Client for federated knowledge distillation."""
    
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, dataset, device='cpu'):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.dataset = dataset
        self.device = device
        self.loss_fn = DistillationLoss()
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-3)
    
    def local_train(self, teacher_weights: Dict[str, torch.Tensor], num_epochs: int = 1):
        self.teacher_model.load_state_dict({k: v.to(self.device) for k, v in teacher_weights.items()})
        self.teacher_model.eval()
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device) if len(batch) > 1 else None
                
                with torch.no_grad():
                    teacher_logits = self.teacher_model(x)
                
                student_logits = self.student_model(x)
                loss = self.loss_fn(student_logits, teacher_logits, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return {k: v.detach().cpu().clone() for k, v in self.student_model.state_dict().items()}