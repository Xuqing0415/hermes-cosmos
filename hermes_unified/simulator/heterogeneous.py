#!/usr/bin/env python3
"""
Heterogeneous Worker Load Balancing Module

实现思路三：异构 Worker 负载均衡调度策略
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import SimulationConfig


@dataclass
class WorkerInfo:
    """单个 Worker 的信息"""
    id: int
    speed: float  # 计算速度系数（相对于基准）
    batch_size: int = 64  # 当前分配的 batch size
    compute_time: float = 0.0  # 实际计算时间（用于自适应调节）


@dataclass
class HeterogeneousConfig(SimulationConfig):
    """异构模拟配置"""
    speed_distribution: str = "random"  # random, uniform, tiered
    min_speed: float = 0.5
    max_speed: float = 2.0
    batch_size_min: int = 8  # batch size 下限
    smoothing_factor: float = 0.8  # 指数移动平均因子


class HeterogeneousSimulator:
    """异构分布式训练模拟器"""
    
    def __init__(self, config: HeterogeneousConfig):
        self.config = config
        self.workers = self._initialize_workers()
        self.historical_times = []  # 历史计算时间记录
    
    def _initialize_workers(self) -> List[WorkerInfo]:
        """初始化异构 workers"""
        workers = []
        
        for i in range(self.config.num_workers):
            if self.config.speed_distribution == "uniform":
                speed = 1.0
            elif self.config.speed_distribution == "tiered":
                # 分层分布：前半慢，后半快
                if i < self.config.num_workers // 2:
                    speed = self.config.min_speed
                else:
                    speed = self.config.max_speed
            else:  # random
                speed = random.uniform(self.config.min_speed, self.config.max_speed)
            
            workers.append(WorkerInfo(
                id=i,
                speed=speed,
                batch_size=self.config.batch_size_per_worker
            ))
        
        return workers
    
    def compute_worker_time(self, worker: WorkerInfo, batch_size: int) -> float:
        """计算单个 worker 处理指定 batch 的时间"""
        base_time = self.config.compute_time_per_worker * (batch_size / 64)
        return base_time / worker.speed
    
    def compute_communication_time(self, total_batch_size: int) -> float:
        """计算通信时间（简化的 DDP 风格）"""
        msg_size = self.config.model_size_mb * self.config.compression_ratio
        
        if self.config.num_workers <= 1:
            return 0.0
        
        comm_time = 2 * msg_size / self.config.bandwidth_mb_s * math.log2(self.config.num_workers)
        base_latency = self.config.latency * math.log2(self.config.num_workers)
        
        return comm_time + base_latency
    
    def simulate_static_sharding(self) -> Dict:
        """
        静态分片策略：每个 worker 处理固定 batch size
        """
        batch_size_per_worker = self.config.batch_size_per_worker
        total_batch_size = batch_size_per_worker * self.config.num_workers
        
        # 计算每个 worker 的完成时间
        worker_times = []
        for worker in self.workers:
            compute_time = self.compute_worker_time(worker, batch_size_per_worker)
            worker_times.append(compute_time)
        
        # 整体 step 时间 = 最慢 worker 的计算时间 + 通信时间
        max_compute_time = max(worker_times)
        comm_time = self.compute_communication_time(total_batch_size)
        step_time = max_compute_time + comm_time
        
        # 计算吞吐量
        throughput = total_batch_size / step_time
        
        # 计算闲置时间
        idle_times = [(max_compute_time - t) for t in worker_times]
        avg_idle_time = sum(idle_times) / len(idle_times)
        
        return {
            'strategy': 'static_sharding',
            'throughput': throughput,
            'step_time': step_time,
            'max_compute_time': max_compute_time,
            'comm_time': comm_time,
            'worker_times': worker_times,
            'idle_times': idle_times,
            'avg_idle_time': avg_idle_time,
            'worker_speeds': [w.speed for w in self.workers],
            'batch_sizes': [batch_size_per_worker] * self.config.num_workers
        }
    
    def simulate_dynamic_load_balancing(self) -> Dict:
        """
        动态负载均衡策略：根据 worker 速度动态分配 batch size
        
        目标：使所有 worker 同时完成计算
        """
        # 计算每个 worker 的速度权重
        total_speed = sum(w.speed for w in self.workers)
        target_total_batch = self.config.batch_size_per_worker * self.config.num_workers
        
        # 根据速度比例分配 batch size
        allocated_batches = []
        for worker in self.workers:
            batch_size = int(target_total_batch * (worker.speed / total_speed))
            # 确保不低于下限
            batch_size = max(self.config.batch_size_min, batch_size)
            allocated_batches.append(batch_size)
        
        # 确保总 batch 数接近目标（可能因为整数截断有偏差）
        while sum(allocated_batches) < target_total_batch:
            # 找到速度最快的 worker，多分配一个 batch
            max_speed_idx = max(range(len(self.workers)), key=lambda i: self.workers[i].speed)
            allocated_batches[max_speed_idx] += 1
        
        total_batch_size = sum(allocated_batches)
        
        # 计算每个 worker 的完成时间（应该接近相等）
        worker_times = []
        for i, worker in enumerate(self.workers):
            compute_time = self.compute_worker_time(worker, allocated_batches[i])
            worker_times.append(compute_time)
        
        # 整体 step 时间
        max_compute_time = max(worker_times)
        comm_time = self.compute_communication_time(total_batch_size)
        step_time = max_compute_time + comm_time
        
        # 计算吞吐量
        throughput = total_batch_size / step_time
        
        # 计算闲置时间（负载均衡后应该很小）
        idle_times = [(max_compute_time - t) for t in worker_times]
        avg_idle_time = sum(idle_times) / len(idle_times)
        
        return {
            'strategy': 'dynamic_load_balancing',
            'throughput': throughput,
            'step_time': step_time,
            'max_compute_time': max_compute_time,
            'comm_time': comm_time,
            'worker_times': worker_times,
            'idle_times': idle_times,
            'avg_idle_time': avg_idle_time,
            'worker_speeds': [w.speed for w in self.workers],
            'batch_sizes': allocated_batches
        }
    
    def simulate_online_adaptive(self) -> Dict:
        """
        在线自适应调节：根据历史计算时间动态调整 batch size
        """
        # 使用指数移动平均平滑历史时间
        alpha = self.config.smoothing_factor
        target_total_batch = self.config.batch_size_per_worker * self.config.num_workers
        allocated_batches = []
        
        for worker in self.workers:
            if len(self.historical_times) == 0:
                # 初始分配
                batch_size = self.config.batch_size_per_worker
            else:
                # 使用历史时间的平均值
                avg_hist_time = sum(self.historical_times) / len(self.historical_times)
                # 根据历史时间调整
                batch_size = int(worker.batch_size * (avg_hist_time / (worker.compute_time + 1e-10)))
            
            batch_size = max(self.config.batch_size_min, min(batch_size, target_total_batch))
            allocated_batches.append(batch_size)
        
        # 重新归一化到目标总 batch
        total_allocated = sum(allocated_batches)
        if total_allocated > 0:
            allocated_batches = [int(b * target_total_batch / total_allocated) for b in allocated_batches]
        
        # 模拟执行一次，更新历史记录
        worker_times = []
        for i, worker in enumerate(self.workers):
            compute_time = self.compute_worker_time(worker, allocated_batches[i])
            worker.compute_time = compute_time
            worker_times.append(compute_time)
            
            # 更新历史记录（指数移动平均）
            if len(self.historical_times) > 0:
                self.historical_times.append(alpha * self.historical_times[-1] + (1 - alpha) * compute_time)
            else:
                self.historical_times.append(compute_time)
        
        max_compute_time = max(worker_times)
        comm_time = self.compute_communication_time(sum(allocated_batches))
        step_time = max_compute_time + comm_time
        throughput = sum(allocated_batches) / step_time
        
        return {
            'strategy': 'online_adaptive',
            'throughput': throughput,
            'step_time': step_time,
            'worker_times': worker_times,
            'batch_sizes': allocated_batches,
            'worker_speeds': [w.speed for w in self.workers]
        }
    
    def run_comparison(self) -> Dict:
        """运行三种策略的对比"""
        static_result = self.simulate_static_sharding()
        dynamic_result = self.simulate_dynamic_load_balancing()
        
        return {
            'config': {
                'num_workers': self.config.num_workers,
                'speed_distribution': self.config.speed_distribution,
                'min_speed': self.config.min_speed,
                'max_speed': self.config.max_speed
            },
            'static_sharding': static_result,
            'dynamic_load_balancing': dynamic_result,
            'improvement': {
                'throughput': (dynamic_result['throughput'] - static_result['throughput']) / static_result['throughput'] * 100,
                'avg_idle_time': (static_result['avg_idle_time'] - dynamic_result['avg_idle_time']) / static_result['avg_idle_time'] * 100
            }
        }
