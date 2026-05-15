#!/usr/bin/env python3
"""
Communication-Compute Overlap Module

实现思路一：通信与计算重叠的动态优化
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import SimulationConfig


@dataclass
class OverlapConfig(SimulationConfig):
    """重叠模拟配置"""
    num_layers: int = 10  # 模型层数（用于分段计算）
    segment_count: int = 5  # 分段数量
    enable_adaptive: bool = True  # 是否启用自适应调整


class OverlapSimulator:
    """通信与计算重叠模拟器"""
    
    def __init__(self, config: OverlapConfig):
        self.config = config
    
    def compute_communication_time(self) -> float:
        """计算通信时间（DDP Ring All-Reduce）"""
        if self.config.num_workers <= 1:
            return 0.0
        
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        comm_time = 2 * msg_size / self.config.bandwidth_mb_s * math.log2(self.config.num_workers)
        base_latency = self.config.latency * math.log2(self.config.num_workers)
        
        return comm_time + base_latency
    
    def simulate_serial_execution(self) -> Dict:
        """模拟完全串行执行（无重叠）"""
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 串行：先计算，再通信
        step_time = compute_time + comm_time
        throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
        
        return {
            'overlap_factor': 0.0,
            'step_time': step_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'segment_count': 1,
            'description': '完全串行'
        }
    
    def simulate_ideal_overlap(self) -> Dict:
        """模拟理想完全重叠"""
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 完全重叠：取最大值
        step_time = max(compute_time, comm_time)
        throughput = self.config.num_workers * self.config.batch_size_per_worker / step_time
        
        return {
            'overlap_factor': 1.0,
            'step_time': step_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'segment_count': self.config.num_layers,
            'description': '完全重叠'
        }
    
    def simulate_partial_overlap(self, overlap_factor: float) -> Dict:
        """
        模拟部分重叠执行
        
        Args:
            overlap_factor: 重叠因子（0~1）
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 分段计算和通信
        num_segments = max(2, self.config.segment_count)
        segment_compute_time = compute_time / num_segments
        
        # 流水线执行
        total_time = segment_compute_time  # 第一段：只计算
        
        # 中间段：计算和通信重叠
        for _ in range(num_segments - 1):
            segment_comm_time = (comm_time / num_segments) * (1 - overlap_factor)
            total_time += max(segment_compute_time, segment_comm_time)
        
        # 最后一段：剩余通信
        remaining_comm_time = comm_time * overlap_factor
        total_time += remaining_comm_time
        
        throughput = self.config.num_workers * self.config.batch_size_per_worker / total_time
        
        return {
            'overlap_factor': overlap_factor,
            'step_time': total_time,
            'throughput': throughput,
            'compute_time': compute_time,
            'comm_time': comm_time,
            'segment_count': num_segments,
            'description': f'部分重叠 ({overlap_factor*100:.0f}%)'
        }
    
    def simulate_dynamic_overlap(self) -> Dict:
        """
        动态重叠调度策略：根据当前带宽和计算速度自动决定重叠程度
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 根据通信/计算比例决定重叠因子
        total_time = compute_time + comm_time
        comm_ratio = comm_time / total_time if total_time > 0 else 0.5
        
        # 动态计算最优重叠因子
        # 当通信比例高时，需要更多重叠
        dynamic_overlap = min(1.0, max(0.0, 2 * comm_ratio - 0.3))
        
        result = self.simulate_partial_overlap(dynamic_overlap)
        result['description'] = f'动态重叠 ({dynamic_overlap*100:.0f}%)'
        
        return result
    
    def simulate_adaptive_segmentation(self) -> Dict:
        """
        自适应分段调整：根据历史通信与计算比例自动调整分段粒度
        """
        compute_time = self.config.compute_time_per_worker
        comm_time = self.compute_communication_time()
        
        # 根据通信时间决定分段数
        # 带宽高时增大分段（更好重叠），带宽低时减小分段（减少额外开销）
        comm_ratio = comm_time / (compute_time + comm_time) if compute_time + comm_time > 0 else 0.5
        
        # 通信比例越高，分段越多
        optimal_segments = max(2, min(self.config.num_layers, int(2 + 8 * comm_ratio)))
        
        # 临时修改配置进行模拟
        original_segments = self.config.segment_count
        self.config.segment_count = optimal_segments
        
        result = self.simulate_dynamic_overlap()
        result['segment_count'] = optimal_segments
        result['description'] = f'自适应分段 ({optimal_segments}段)'
        
        # 恢复配置
        self.config.segment_count = original_segments
        
        return result
    
    def run_overlap_study(self, overlap_factors: Optional[List[float]] = None) -> Dict:
        """运行重叠因子研究"""
        if overlap_factors is None:
            overlap_factors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        results = {
            'config': {
                'num_workers': self.config.num_workers,
                'model_size_mb': self.config.model_size_mb,
                'bandwidth_mb_s': self.config.bandwidth_mb_s,
                'num_layers': self.config.num_layers
            },
            'fixed_overlap': [],
            'dynamic_overlap': None,
            'adaptive_segmentation': None
        }
        
        # 测试不同重叠因子
        for factor in overlap_factors:
            result = self.simulate_partial_overlap(factor)
            results['fixed_overlap'].append(result)
        
        # 动态重叠
        results['dynamic_overlap'] = self.simulate_dynamic_overlap()
        
        # 自适应分段
        results['adaptive_segmentation'] = self.simulate_adaptive_segmentation()
        
        return results
