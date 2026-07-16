#!/usr/bin/env python3
"""
Hermes Unified Adaptive Optimizer


1.  vs 
2. 
3.  GPU 
4. /
"""

import sqlite3
import time
from typing import Dict, List, Optional
import numpy as np

# 
GradientCompressor = None
GradientFusion = None

def _lazy_import():
    global GradientCompressor, GradientFusion
    from hermes_unified.optimizations import GradientCompressor as GC, GradientFusion as GF
    GradientCompressor = GC
    GradientFusion = GF


class OptimizationAdvisor:
    """ - """
    
    def __init__(self, db_path: str = "benchmark_metrics.db"):
        self.db_path = db_path
        self.optimizations = {
            'gradient_compression': False,
            'gradient_fusion': False,
            'gradient_clipping': False,
            'auto_batch_size': False,
        }
        self.compression_ratio = 0.1
        self.fusion_threshold = 1024
        self.clip_value = 1.0
    
    def analyze_metrics(self, mode: str = "ddp", recent_batches: int = 100) -> Dict:
        """
        
        
        Returns:
            analysis: 
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 
        cursor.execute('''
            SELECT loss, grad_norm, comm_delay_ms, gpu_memory_used, gpu_utilization
            FROM training_metrics 
            WHERE mode = ? 
            ORDER BY id DESC LIMIT ?
        ''', (mode, recent_batches))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"status": "no_data"}
        
        # 
        losses = [row[0] for row in rows]
        grad_norms = [row[1] for row in rows if row[1] is not None]
        comm_delays = [row[2] for row in rows if row[2] is not None]
        gpu_utilizations = [row[4] for row in rows if row[4] is not None]
        
        # 
        analysis = {
            "mode": mode,
            "sample_count": len(rows),
            "avg_loss": np.mean(losses),
            "loss_std": np.std(losses),
            "avg_grad_norm": np.mean(grad_norms) if grad_norms else None,
            "grad_norm_std": np.std(grad_norms) if grad_norms else None,
            "avg_comm_delay_ms": np.mean(comm_delays) if comm_delays else None,
            "comm_delay_std": np.std(comm_delays) if comm_delays else None,
            "avg_gpu_utilization": np.mean(gpu_utilizations) if gpu_utilizations else None,
            "gpu_util_std": np.std(gpu_utilizations) if gpu_utilizations else None,
        }
        
        return analysis
    
    def suggest_optimizations(self, analysis: Dict) -> List[Dict]:
        """
        
        
        Args:
            analysis: 
            
        Returns:
            suggestions: 
        """
        suggestions = []
        
        # 1:  > 10% → 
        if analysis["avg_comm_delay_ms"] is not None:
            # 5-10
            step_time_estimate = analysis["avg_comm_delay_ms"] * 5  # 
            if analysis["avg_comm_delay_ms"] > 0.1 * step_time_estimate:
                suggestions.append({
                    "type": "gradient_compression",
                    "action": "enable",
                    "reason": f"({analysis['avg_comm_delay_ms']:.2f}ms) > 10%",
                    "params": {"compression_ratio": self.compression_ratio}
                })
                self.optimizations['gradient_compression'] = True
        
        # 2:  → 
        if analysis["grad_norm_std"] is not None and analysis["avg_grad_norm"] is not None:
            grad_norm_cv = analysis["grad_norm_std"] / analysis["avg_grad_norm"]
            if grad_norm_cv > 0.3:  #  > 30%
                suggestions.append({
                    "type": "gradient_clipping",
                    "action": "enable",
                    "reason": f"(CV={grad_norm_cv:.2f} > 0.3)",
                    "params": {"clip_value": self.clip_value}
                })
                self.optimizations['gradient_clipping'] = True
        
        # 3: GPU80% → batch size
        if analysis["avg_gpu_utilization"] is not None:
            if analysis["avg_gpu_utilization"] < 80:
                suggestions.append({
                    "type": "auto_batch_size",
                    "action": "increase",
                    "reason": f"GPU({analysis['avg_gpu_utilization']:.1f}%) < 80%",
                    "params": {"suggested_increase": "2x"}
                })
                self.optimizations['auto_batch_size'] = True
        
        # 4:  → 
        if analysis["comm_delay_std"] is not None:
            comm_cv = analysis["comm_delay_std"] / analysis["avg_comm_delay_ms"]
            if comm_cv > 0.5:  #  > 50%
                suggestions.append({
                    "type": "gradient_fusion",
                    "action": "enable",
                    "reason": f"(CV={comm_cv:.2f} > 0.5)",
                    "params": {"fusion_threshold": self.fusion_threshold}
                })
                self.optimizations['gradient_fusion'] = True
        
        return suggestions
    
    def get_optimization_status(self) -> Dict:
        """"""
        return self.optimizations


class AdaptiveWorker:
    """Worker - """
    
    def __init__(self, model, optimizer, device, amp_enabled: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.amp_enabled = amp_enabled
        
        # 
        if GradientCompressor is None:
            _lazy_import()
        
        # 
        self.gradient_compressor = GradientCompressor(compression_ratio=0.1)
        self.gradient_fusion = GradientFusion(fusion_threshold=1024)
        self.amp_scaler = None
        if amp_enabled and torch.cuda.is_available():
            self.amp_scaler = torch.cuda.amp.GradScaler()
        
        # 
        self.use_compression = False
        self.use_fusion = False
        self.use_clipping = False
        self.clip_value = 1.0
        
        # 
        self.step_count = 0
        self.comm_delay_history = []
        self.grad_norm_history = []
        
    def train_step(self, inputs, targets, criterion):
        """
        
        
        Args:
            inputs: 
            targets: 
            criterion: 
            
        Returns:
            loss: 
        """
        self.step_count += 1
        
        self.optimizer.zero_grad()
        
        # 
        comm_start = time.time()
        
        # AMP 
        if self.amp_enabled and self.amp_scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
        
        # 
        comm_delay_ms = (time.time() - comm_start) * 1000
        self.comm_delay_history.append(comm_delay_ms)
        
        # AMP 
        if self.amp_enabled and self.amp_scaler is not None:
            self.amp_scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 
        if self.use_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_value
            )
            self.grad_norm_history.append(grad_norm.item())
        
        # 
        if self.use_compression:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    compressed, meta = self.gradient_compressor.compress(
                        param.grad, name
                    )
                    decompressed = self.gradient_compressor.decompress(compressed, meta)
                    param.grad = decompressed
        
        # 
        if self.use_fusion:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.gradient_fusion.add_gradient(name, param.grad)
        
        # 
        if self.amp_enabled and self.amp_scaler is not None:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()
        
        return loss.item()
    
    def update_strategy(self, suggestions: List[Dict]):
        """
        
        
        Args:
            suggestions: 
        """
        for suggestion in suggestions:
            opt_type = suggestion["type"]
            action = suggestion["action"]
            params = suggestion.get("params", {})
            
            if opt_type == "gradient_compression":
                if action == "enable":
                    self.use_compression = True
                    self.gradient_compressor.compression_ratio = params.get("compression_ratio", 0.1)
                else:
                    self.use_compression = False
            
            elif opt_type == "gradient_fusion":
                if action == "enable":
                    self.use_fusion = True
                    self.gradient_fusion.fusion_threshold = params.get("fusion_threshold", 1024)
                else:
                    self.use_fusion = False
            
            elif opt_type == "gradient_clipping":
                if action == "enable":
                    self.use_clipping = True
                    self.clip_value = params.get("clip_value", 1.0)
                else:
                    self.use_clipping = False
            
            elif opt_type == "auto_batch_size":
                if action == "increase":
                    print(f"[ADAPTIVE]  batch size: {params.get('suggested_increase')}")
    
    def get_stats(self) -> Dict:
        """"""
        return {
            "step_count": self.step_count,
            "avg_comm_delay_ms": np.mean(self.comm_delay_history) if self.comm_delay_history else 0,
            "grad_norm_std": np.std(self.grad_norm_history) if self.grad_norm_history else 0,
            "use_compression": self.use_compression,
            "use_fusion": self.use_fusion,
            "use_clipping": self.use_clipping,
        }


def run_adaptive_training_example():
    """"""
    print("=== Adaptive Training Example ===")
    
    # 
    advisor = OptimizationAdvisor()
    
    # 
    mock_analysis = {
        "mode": "gossip",
        "sample_count": 100,
        "avg_loss": 0.5,
        "loss_std": 0.1,
        "avg_grad_norm": 2.5,
        "grad_norm_std": 1.0,  # 
        "avg_comm_delay_ms": 50.0,
        "comm_delay_std": 30.0,  # 
        "avg_gpu_utilization": 75.0,  # 80%
        "gpu_util_std": 5.0,
    }
    
    print("\n:")
    print(f"  : {mock_analysis['avg_comm_delay_ms']:.2f}ms")
    print(f"  : {mock_analysis['grad_norm_std']:.2f}")
    print(f"  GPU: {mock_analysis['avg_gpu_utilization']:.1f}%")
    
    # 
    suggestions = advisor.suggest_optimizations(mock_analysis)
    
    print("\n:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. [{suggestion['type']}] {suggestion['action']} - {suggestion['reason']}")
        if 'params' in suggestion:
            print(f"     : {suggestion['params']}")
    
    print("\n:")
    status = advisor.get_optimization_status()
    for opt, enabled in status.items():
        print(f"  {opt}: {' ' if enabled else ' '}")


if __name__ == "__main__":
    #  torch
    global torch
    import torch
    
    run_adaptive_training_example()
