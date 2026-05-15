#!/usr/bin/env python3
"""
Network Congestion Module

实现思路四：模拟网络拥塞与动态带宽调整
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import SimulationConfig


@dataclass
class CongestionConfig(SimulationConfig):
    """拥塞模拟配置"""
    congestion_probability: float = 0.2  # 每步发生拥塞的概率
    congestion_factor: float = 0.5  # 拥塞时带宽衰减比例
    initial_compression_ratio: float = 0.1  # 初始压缩率
    min_compression_ratio: float = 0.01  # 最小压缩率
    max_compression_ratio: float = 1.0  # 最大压缩率
    compression_adjust_factor: float = 0.1  # 压缩率调整步长
    smoothing_factor: float = 0.9  # 指数移动平均因子
    simulation_steps: int = 100  # 模拟步数
    congestion_start_step: int = 50  # 拥塞开始步骤
    congestion_end_step: int = 80  # 拥塞结束步骤


class CongestionSimulator:
    """网络拥塞模拟器"""
    
    def __init__(self, config: CongestionConfig):
        self.config = config
        self.current_bandwidth = config.bandwidth_mb_s
        self.current_compression = config.initial_compression_ratio
        self.congested = False
        self.estimated_bandwidth = config.bandwidth_mb_s
        self.loss_history = []
    
    def compute_communication_time(self, compression_ratio: float) -> float:
        """计算通信时间"""
        if self.config.num_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * compression_ratio
        comm_time = 2 * msg_size / self.current_bandwidth * math.log2(self.config.num_workers)
        base_latency = self.config.latency * math.log2(self.config.num_workers)
        
        return comm_time + base_latency
    
    def detect_congestion(self, actual_comm_time: float, expected_comm_time: float) -> bool:
        """
        检测拥塞：比较实际通信时间与预期通信时间
        
        Args:
            actual_comm_time: 实际通信时间
            expected_comm_time: 预期通信时间（基于基准带宽）
        
        Returns:
            congested: 是否拥塞
        """
        # 如果实际时间比预期大超过20%，判定为拥塞
        if actual_comm_time > expected_comm_time * 1.2:
            return True
        return False
    
    def adjust_compression_ratio(self, congested: bool) -> float:
        """
        根据拥塞状态调整压缩率
        
        Args:
            congested: 是否拥塞
        
        Returns:
            new_compression_ratio: 新的压缩率
        """
        if congested:
            # 拥塞时：提高压缩率（减少数据量）
            new_ratio = max(
                self.config.min_compression_ratio,
                self.current_compression * (1 - self.config.compression_adjust_factor)
            )
        else:
            # 非拥塞时：逐渐恢复压缩率
            new_ratio = min(
                self.config.max_compression_ratio,
                self.current_compression * (1 + self.config.compression_adjust_factor * 0.5)
            )
        
        # 使用平滑因子避免剧烈波动
        self.current_compression = self.config.smoothing_factor * self.current_compression + \
                                  (1 - self.config.smoothing_factor) * new_ratio
        
        return self.current_compression
    
    def simulate_static_compression(self) -> List[Dict]:
        """模拟静态压缩策略（压缩率固定）"""
        results = []
        
        for step in range(self.config.simulation_steps):
            # 模拟拥塞注入
            if self.config.congestion_start_step <= step < self.config.congestion_end_step:
                self.current_bandwidth = self.config.bandwidth_mb_s * self.config.congestion_factor
                self.congested = True
            else:
                self.current_bandwidth = self.config.bandwidth_mb_s
                self.congested = False
            
            # 静态压缩：压缩率不变
            compression_ratio = self.config.initial_compression_ratio
            
            comm_time = self.compute_communication_time(compression_ratio)
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
            
            # 模拟精度损失（压缩率越低，损失越大）
            loss = 0.1 + (1 - compression_ratio) * 0.2
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': compression_ratio,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'loss': loss,
                'strategy': 'static'
            })
        
        return results
    
    def simulate_dynamic_compression(self) -> List[Dict]:
        """模拟动态压缩策略（根据拥塞调整压缩率）"""
        results = []
        self.current_compression = self.config.initial_compression_ratio
        
        for step in range(self.config.simulation_steps):
            # 模拟拥塞注入
            if self.config.congestion_start_step <= step < self.config.congestion_end_step:
                self.current_bandwidth = self.config.bandwidth_mb_s * self.config.congestion_factor
                self.congested = True
            else:
                self.current_bandwidth = self.config.bandwidth_mb_s
                self.congested = False
            
            # 根据拥塞状态调整压缩率
            self.adjust_compression_ratio(self.congested)
            compression_ratio = self.current_compression
            
            comm_time = self.compute_communication_time(compression_ratio)
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
            
            # 模拟精度损失
            loss = 0.1 + (1 - compression_ratio) * 0.2
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': compression_ratio,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'loss': loss,
                'strategy': 'dynamic'
            })
        
        return results
    
    def simulate_congestion_detection(self) -> List[Dict]:
        """模拟带拥塞检测的动态压缩策略"""
        results = []
        self.current_compression = self.config.initial_compression_ratio
        
        for step in range(self.config.simulation_steps):
            # 随机发生拥塞
            if random.random() < self.config.congestion_probability:
                self.current_bandwidth = self.config.bandwidth_mb_s * self.config.congestion_factor
            else:
                self.current_bandwidth = self.config.bandwidth_mb_s
            
            # 计算预期通信时间（基于基准带宽）
            expected_comm_time = 2 * self.config.model_size_mb * self.current_compression / \
                                self.config.bandwidth_mb_s * math.log2(self.config.num_workers)
            
            # 计算实际通信时间（基于当前带宽）
            comm_time = self.compute_communication_time(self.current_compression)
            
            # 检测拥塞
            self.congested = self.detect_congestion(comm_time, expected_comm_time)
            
            # 根据检测结果调整压缩率
            self.adjust_compression_ratio(self.congested)
            
            step_time = self.config.compute_time_per_worker + comm_time
            throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
            loss = 0.1 + (1 - self.current_compression) * 0.2
            
            results.append({
                'step': step,
                'bandwidth': self.current_bandwidth,
                'compression_ratio': self.current_compression,
                'congested': self.congested,
                'throughput': throughput,
                'step_time': step_time,
                'loss': loss,
                'strategy': 'detection_based'
            })
        
        return results
    
    def run_comparison(self) -> Dict:
        """运行静态 vs 动态压缩对比"""
        static_results = self.simulate_static_compression()
        dynamic_results = self.simulate_dynamic_compression()
        
        # 计算统计指标
        static_throughputs = [r['throughput'] for r in static_results]
        dynamic_throughputs = [r['throughput'] for r in dynamic_results]
        static_losses = [r['loss'] for r in static_results]
        dynamic_losses = [r['loss'] for r in dynamic_results]
        
        return {
            'config': {
                'num_workers': self.config.num_workers,
                'model_size_mb': self.config.model_size_mb,
                'base_bandwidth': self.config.bandwidth_mb_s,
                'congestion_probability': self.config.congestion_probability,
                'initial_compression': self.config.initial_compression_ratio
            },
            'static_compression': static_results,
            'dynamic_compression': dynamic_results,
            'statistics': {
                'static_avg_throughput': sum(static_throughputs) / len(static_throughputs),
                'static_min_throughput': min(static_throughputs),
                'static_max_throughput': max(static_throughputs),
                'static_avg_loss': sum(static_losses) / len(static_losses),
                'dynamic_avg_throughput': sum(dynamic_throughputs) / len(dynamic_throughputs),
                'dynamic_min_throughput': min(dynamic_throughputs),
                'dynamic_max_throughput': max(dynamic_throughputs),
                'dynamic_avg_loss': sum(dynamic_losses) / len(dynamic_losses)
            }
        }
