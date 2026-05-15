#!/usr/bin/env python3
"""
Hermes Unified Adaptive Optimizer

根据实时指标自动选择优化策略：
1. 分析通信延迟 vs 计算时间
2. 分析梯度范数波动
3. 分析 GPU 利用率
4. 自动开启/关闭优化策略
"""

import sqlite3
import time
from typing import Dict, List, Optional
import numpy as np

# 延迟导入以避免循环依赖
GradientCompressor = None
GradientFusion = None

def _lazy_import():
    global GradientCompressor, GradientFusion
    from hermes_unified.optimizations import GradientCompressor as GC, GradientFusion as GF
    GradientCompressor = GC
    GradientFusion = GF


class OptimizationAdvisor:
    """优化顾问 - 根据指标数据提供优化建议"""
    
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
        分析指标数据库，识别性能瓶颈
        
        Returns:
            analysis: 分析结果字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取最近批次的指标
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
        
        # 提取数据
        losses = [row[0] for row in rows]
        grad_norms = [row[1] for row in rows if row[1] is not None]
        comm_delays = [row[2] for row in rows if row[2] is not None]
        gpu_utilizations = [row[4] for row in rows if row[4] is not None]
        
        # 计算统计量
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
        根据分析结果建议优化策略
        
        Args:
            analysis: 分析结果
            
        Returns:
            suggestions: 优化建议列表
        """
        suggestions = []
        
        # 规则1: 通信延迟 > 计算时间的10% → 开启梯度压缩
        if analysis["avg_comm_delay_ms"] is not None:
            # 假设每步计算时间约为通信延迟的5-10倍（粗略估计）
            step_time_estimate = analysis["avg_comm_delay_ms"] * 5  # 粗略估计
            if analysis["avg_comm_delay_ms"] > 0.1 * step_time_estimate:
                suggestions.append({
                    "type": "gradient_compression",
                    "action": "enable",
                    "reason": f"通信延迟({analysis['avg_comm_delay_ms']:.2f}ms) > 计算时间的10%",
                    "params": {"compression_ratio": self.compression_ratio}
                })
                self.optimizations['gradient_compression'] = True
        
        # 规则2: 梯度范数波动大 → 开启梯度裁剪
        if analysis["grad_norm_std"] is not None and analysis["avg_grad_norm"] is not None:
            grad_norm_cv = analysis["grad_norm_std"] / analysis["avg_grad_norm"]
            if grad_norm_cv > 0.3:  # 变异系数 > 30%
                suggestions.append({
                    "type": "gradient_clipping",
                    "action": "enable",
                    "reason": f"梯度范数波动大(CV={grad_norm_cv:.2f} > 0.3)",
                    "params": {"clip_value": self.clip_value}
                })
                self.optimizations['gradient_clipping'] = True
        
        # 规则3: GPU利用率低于80% → 建议增大batch size
        if analysis["avg_gpu_utilization"] is not None:
            if analysis["avg_gpu_utilization"] < 80:
                suggestions.append({
                    "type": "auto_batch_size",
                    "action": "increase",
                    "reason": f"GPU利用率({analysis['avg_gpu_utilization']:.1f}%) < 80%",
                    "params": {"suggested_increase": "2x"}
                })
                self.optimizations['auto_batch_size'] = True
        
        # 规则4: 通信延迟波动大 → 开启梯度融合
        if analysis["comm_delay_std"] is not None:
            comm_cv = analysis["comm_delay_std"] / analysis["avg_comm_delay_ms"]
            if comm_cv > 0.5:  # 变异系数 > 50%
                suggestions.append({
                    "type": "gradient_fusion",
                    "action": "enable",
                    "reason": f"通信延迟波动大(CV={comm_cv:.2f} > 0.5)",
                    "params": {"fusion_threshold": self.fusion_threshold}
                })
                self.optimizations['gradient_fusion'] = True
        
        return suggestions
    
    def get_optimization_status(self) -> Dict:
        """获取当前优化状态"""
        return self.optimizations


class AdaptiveWorker:
    """自适应Worker - 根据实时指标自动调整优化策略"""
    
    def __init__(self, model, optimizer, device, amp_enabled: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.amp_enabled = amp_enabled
        
        # 延迟导入优化组件
        if GradientCompressor is None:
            _lazy_import()
        
        # 优化器组件
        self.gradient_compressor = GradientCompressor(compression_ratio=0.1)
        self.gradient_fusion = GradientFusion(fusion_threshold=1024)
        self.amp_scaler = None
        if amp_enabled and torch.cuda.is_available():
            self.amp_scaler = torch.cuda.amp.GradScaler()
        
        # 优化状态
        self.use_compression = False
        self.use_fusion = False
        self.use_clipping = False
        self.clip_value = 1.0
        
        # 统计数据
        self.step_count = 0
        self.comm_delay_history = []
        self.grad_norm_history = []
        
    def train_step(self, inputs, targets, criterion):
        """
        执行一个训练步骤，自动应用优化策略
        
        Args:
            inputs: 输入数据
            targets: 目标标签
            criterion: 损失函数
            
        Returns:
            loss: 损失值
        """
        self.step_count += 1
        
        self.optimizer.zero_grad()
        
        # 记录通信开始时间
        comm_start = time.time()
        
        # AMP 前向传播
        if self.amp_enabled and self.amp_scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
        
        # 记录通信延迟
        comm_delay_ms = (time.time() - comm_start) * 1000
        self.comm_delay_history.append(comm_delay_ms)
        
        # AMP 反向传播
        if self.amp_enabled and self.amp_scaler is not None:
            self.amp_scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度裁剪（如果启用）
        if self.use_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_value
            )
            self.grad_norm_history.append(grad_norm.item())
        
        # 梯度压缩（如果启用）
        if self.use_compression:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    compressed, meta = self.gradient_compressor.compress(
                        param.grad, name
                    )
                    decompressed = self.gradient_compressor.decompress(compressed, meta)
                    param.grad = decompressed
        
        # 梯度融合（如果启用）
        if self.use_fusion:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.gradient_fusion.add_gradient(name, param.grad)
        
        # 优化器步骤
        if self.amp_enabled and self.amp_scaler is not None:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()
        
        return loss.item()
    
    def update_strategy(self, suggestions: List[Dict]):
        """
        根据优化建议更新策略
        
        Args:
            suggestions: 优化建议列表
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
                    print(f"[ADAPTIVE] 建议增大 batch size: {params.get('suggested_increase')}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "step_count": self.step_count,
            "avg_comm_delay_ms": np.mean(self.comm_delay_history) if self.comm_delay_history else 0,
            "grad_norm_std": np.std(self.grad_norm_history) if self.grad_norm_history else 0,
            "use_compression": self.use_compression,
            "use_fusion": self.use_fusion,
            "use_clipping": self.use_clipping,
        }


def run_adaptive_training_example():
    """运行自适应训练示例"""
    print("=== Adaptive Training Example ===")
    
    # 模拟分析
    advisor = OptimizationAdvisor()
    
    # 模拟分析结果（模拟通信瓶颈场景）
    mock_analysis = {
        "mode": "gossip",
        "sample_count": 100,
        "avg_loss": 0.5,
        "loss_std": 0.1,
        "avg_grad_norm": 2.5,
        "grad_norm_std": 1.0,  # 高波动
        "avg_comm_delay_ms": 50.0,
        "comm_delay_std": 30.0,  # 高波动
        "avg_gpu_utilization": 75.0,  # 低于80%
        "gpu_util_std": 5.0,
    }
    
    print("\n分析结果:")
    print(f"  平均通信延迟: {mock_analysis['avg_comm_delay_ms']:.2f}ms")
    print(f"  梯度范数波动: {mock_analysis['grad_norm_std']:.2f}")
    print(f"  GPU利用率: {mock_analysis['avg_gpu_utilization']:.1f}%")
    
    # 获取建议
    suggestions = advisor.suggest_optimizations(mock_analysis)
    
    print("\n优化建议:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. [{suggestion['type']}] {suggestion['action']} - {suggestion['reason']}")
        if 'params' in suggestion:
            print(f"     参数: {suggestion['params']}")
    
    print("\n当前优化状态:")
    status = advisor.get_optimization_status()
    for opt, enabled in status.items():
        print(f"  {opt}: {'✅ 启用' if enabled else '❌ 禁用'}")


if __name__ == "__main__":
    # 导入 torch（仅在主函数中，避免未安装时出错）
    global torch
    import torch
    
    run_adaptive_training_example()
