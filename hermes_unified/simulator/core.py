#!/usr/bin/env python3
"""
Core Simulation Module

核心分布式训练模拟器，提供基础的通信时间计算和吞吐量估算
支持动态worker加入/退出、异构网络、稀疏梯度通信
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class WorkerInfo:
    """Worker信息"""
    worker_id: int
    speed: float = 1.0  # 计算速度系数
    batch_size: int = 64
    compute_time: float = 0.02
    alive: bool = True
    join_time: float = 0.0
    leave_time: Optional[float] = None


@dataclass
class NetworkLink:
    """网络链路信息"""
    src_id: int
    dst_id: int
    bandwidth_mb_s: float = 100.0
    latency: float = 0.0001
    congestion_level: float = 0.0  # 0-1，0表示无拥塞


@dataclass
class SimulationConfig:
    """模拟配置参数"""
    num_workers: int = 4
    model_size_mb: float = 40.0  # 模型大小（MB）
    compute_time_per_worker: float = 0.02  # 单个worker纯计算耗时（秒）
    bandwidth_mb_s: float = 100.0  # 网络带宽（MB/s）
    latency: float = 0.0001  # 网络延迟（秒）
    gossip_neighbors: int = 2  # Gossip模式每轮通信邻居数
    compression_ratio: float = 1.0  # 梯度压缩比例
    overlap_factor: float = 0.0  # 计算通信重叠因子
    batch_size_per_worker: int = 64  # 每个worker的batch size
    enable_noise: bool = False  # 是否添加随机噪声
    
    # 动态worker配置
    enable_dynamic_workers: bool = False
    worker_join_rate: float = 0.05  # 每step新增worker的概率
    worker_leave_rate: float = 0.02  # 每step worker离开的概率
    
    # 异构网络配置
    enable_heterogeneous_network: bool = False
    bandwidth_variation: float = 0.3  # 带宽变异系数
    latency_variation: float = 0.5  # 延迟变异系数
    
    # 稀疏梯度通信
    enable_sparse_gradient: bool = False
    sparse_top_k_ratio: float = 0.1  # Top-k比例


class DistributedSimulator:
    """分布式训练模拟器"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.workers: List[WorkerInfo] = []
        self.network_links: Dict[Tuple[int, int], NetworkLink] = {}
        self.step_count = 0
        self.total_throughput = 0.0
        
        # 初始化workers
        self._init_workers()
        
        # 初始化网络拓扑
        if self.config.enable_heterogeneous_network:
            self._init_heterogeneous_network()
    
    def _init_workers(self):
        """初始化worker列表"""
        self.workers = []
        for i in range(self.config.num_workers):
            # 异构worker：随机速度系数
            speed = random.uniform(0.5, 1.5) if self.config.enable_dynamic_workers else 1.0
            self.workers.append(WorkerInfo(
                worker_id=i,
                speed=speed,
                batch_size=self.config.batch_size_per_worker,
                compute_time=self.config.compute_time_per_worker / speed,
                alive=True,
                join_time=0.0
            ))
    
    def _init_heterogeneous_network(self):
        """初始化异构网络拓扑（树形结构）"""
        self.network_links = {}
        
        for i in range(self.config.num_workers):
            for j in range(self.config.num_workers):
                if i != j:
                    # 根节点（worker 0）有更高带宽
                    if i == 0 or j == 0:
                        base_bandwidth = self.config.bandwidth_mb_s * (1.2 + random.uniform(-0.1, 0.1))
                        base_latency = self.config.latency * (0.8 + random.uniform(-0.2, 0.2))
                    else:
                        # 叶子节点之间带宽较低
                        base_bandwidth = self.config.bandwidth_mb_s * (0.6 + random.uniform(-0.3, 0.3))
                        base_latency = self.config.latency * (1.5 + random.uniform(-0.5, 0.5))
                    
                    self.network_links[(i, j)] = NetworkLink(
                        src_id=i,
                        dst_id=j,
                        bandwidth_mb_s=base_bandwidth,
                        latency=base_latency,
                        congestion_level=0.0
                    )
    
    def add_noise(self, value: float, noise_level: float = 0.05) -> float:
        """添加随机噪声"""
        if self.config.enable_noise:
            noise = random.uniform(-noise_level, noise_level)
            return value * (1 + noise)
        return value
    
    def get_network_link(self, src_id: int, dst_id: int) -> NetworkLink:
        """获取网络链路信息"""
        if self.config.enable_heterogeneous_network:
            key = (src_id, dst_id)
            if key in self.network_links:
                return self.network_links[key]
        
        return NetworkLink(
            src_id=src_id,
            dst_id=dst_id,
            bandwidth_mb_s=self.config.bandwidth_mb_s,
            latency=self.config.latency
        )
    
    def update_worker_dynamics(self):
        """更新worker动态（加入/退出）"""
        if not self.config.enable_dynamic_workers:
            return
        
        # Worker退出
        for worker in self.workers:
            if worker.alive and random.random() < self.config.worker_leave_rate:
                worker.alive = False
                worker.leave_time = self.step_count
                print(f"👋 Worker {worker.worker_id} left at step {self.step_count}")
        
        # Worker加入
        if random.random() < self.config.worker_join_rate:
            new_id = max(w.worker_id for w in self.workers) + 1
            speed = random.uniform(0.5, 1.5)
            self.workers.append(WorkerInfo(
                worker_id=new_id,
                speed=speed,
                batch_size=self.config.batch_size_per_worker,
                compute_time=self.config.compute_time_per_worker / speed,
                alive=True,
                join_time=self.step_count
            ))
            print(f"🔔 New worker {new_id} joined at step {self.step_count}")
    
    def get_alive_workers(self) -> List[WorkerInfo]:
        """获取存活的worker列表"""
        return [w for w in self.workers if w.alive]
    
    def compute_sparse_gradient_size(self) -> float:
        """计算稀疏梯度的实际大小"""
        if not self.config.enable_sparse_gradient:
            return self.config.model_size_mb * self.config.compression_ratio
        
        # Top-k稀疏化
        effective_ratio = self.config.compression_ratio * self.config.sparse_top_k_ratio
        return self.config.model_size_mb * effective_ratio
    
    def compute_communication_time_ps(self) -> float:
        """
        计算参数服务器模式的通信时间
        """
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return 0.0
        
        msg_size_send = self.compute_sparse_gradient_size()
        msg_size_recv = self.config.model_size_mb
        
        # 使用最慢的链路作为瓶颈
        max_upload_time = 0.0
        max_download_time = 0.0
        
        for worker in alive_workers:
            link = self.get_network_link(worker.worker_id, 0)  # PS作为虚拟节点0
            upload_time = msg_size_send / link.bandwidth_mb_s + link.latency
            download_time = msg_size_recv / link.bandwidth_mb_s + link.latency
            
            max_upload_time = max(max_upload_time, upload_time)
            max_download_time = max(max_download_time, download_time)
        
        comm_time = max_upload_time + max_download_time
        return self.add_noise(comm_time)
    
    def compute_communication_time_ddp(self) -> float:
        """
        计算DDP模式的通信时间（Ring All-Reduce）
        """
        alive_workers = self.get_alive_workers()
        num_workers = len(alive_workers)
        
        if num_workers <= 1:
            return 0.0
        
        msg_size = self.compute_sparse_gradient_size()
        total_comm_time = 0.0
        
        # 环形通信：每个worker依次与下一个worker交换数据
        for i in range(num_workers):
            src = alive_workers[i]
            dst = alive_workers[(i + 1) % num_workers]
            
            link = self.get_network_link(src.worker_id, dst.worker_id)
            
            # Ring All-Reduce有两个阶段：scatter-reduce和all-gather
            stage_time = 2 * msg_size / link.bandwidth_mb_s + link.latency
            total_comm_time += stage_time
        
        # 取最大值作为瓶颈
        comm_time = total_comm_time / num_workers * math.log2(num_workers)
        return self.add_noise(comm_time)
    
    def compute_communication_time_gossip(self) -> float:
        """
        计算Gossip模式的通信时间
        """
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return 0.0
        
        msg_size = self.compute_sparse_gradient_size()
        k = min(self.config.gossip_neighbors, len(alive_workers) - 1)
        
        max_comm_time = 0.0
        
        for worker in alive_workers:
            # 随机选择k个邻居
            neighbors = random.sample(
                [w for w in alive_workers if w.worker_id != worker.worker_id],
                min(k, len(alive_workers) - 1)
            )
            
            worker_comm_time = 0.0
            for neighbor in neighbors:
                link = self.get_network_link(worker.worker_id, neighbor.worker_id)
                worker_comm_time += 2 * msg_size / link.bandwidth_mb_s + link.latency
            
            max_comm_time = max(max_comm_time, worker_comm_time)
        
        return self.add_noise(max_comm_time)
    
    def compute_step_time(self, comm_time: float) -> float:
        """
        计算单个step的总时间
        """
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return float('inf')
        
        # 取最慢worker的计算时间
        compute_time = max(w.compute_time for w in alive_workers)
        overlap = self.config.overlap_factor
        
        step_time = max(compute_time, comm_time * overlap) + (1 - overlap) * (compute_time + comm_time)
        return max(step_time, 0.0001)
    
    def compute_throughput(self, step_time: float) -> float:
        """
        计算吞吐量（samples/sec）
        """
        alive_workers = self.get_alive_workers()
        total_batch_size = sum(w.batch_size for w in alive_workers)
        return total_batch_size / step_time
    
    def evaluate_protocol(self, protocol: str) -> Dict:
        """评估指定协议的性能"""
        if protocol == 'ps':
            comm_time = self.compute_communication_time_ps()
        elif protocol == 'ddp':
            comm_time = self.compute_communication_time_ddp()
        elif protocol == 'gossip':
            comm_time = self.compute_communication_time_gossip()
        else:
            comm_time = self.compute_communication_time_ddp()
        
        step_time = self.compute_step_time(comm_time)
        throughput = self.compute_throughput(step_time)
        
        return {
            'protocol': protocol,
            'comm_time': comm_time,
            'step_time': step_time,
            'throughput': throughput,
            'alive_workers': len(self.get_alive_workers())
        }
    
    def run_protocol_comparison(self) -> Dict:
        """运行三种协议的对比"""
        results = {}
        for protocol in ['ps', 'ddp', 'gossip']:
            results[protocol] = self.evaluate_protocol(protocol)
        
        best_protocol = max(results, key=lambda p: results[p]['throughput'])
        
        return {
            'config': {
                'num_workers': self.config.num_workers,
                'model_size_mb': self.config.model_size_mb,
                'bandwidth_mb_s': self.config.bandwidth_mb_s
            },
            'results': results,
            'best_protocol': best_protocol,
            'step': self.step_count
        }
    
    def run_dynamic_simulation(self, steps: int = 100) -> List[Dict]:
        """运行动态模拟（包含worker加入/退出）"""
        history = []
        
        for step in range(steps):
            self.step_count = step
            
            # 更新worker动态
            self.update_worker_dynamics()
            
            # 评估所有协议
            result = self.run_protocol_comparison()
            history.append(result)
            
            # 更新网络拥塞状态
            self._update_network_congestion()
        
        return history
    
    def _update_network_congestion(self):
        """更新网络拥塞状态"""
        if not self.config.enable_heterogeneous_network:
            return
        
        alive_workers = self.get_alive_workers()
        num_alive = len(alive_workers)
        
        for (src, dst), link in self.network_links.items():
            # 根据并发流量更新拥塞
            congestion = min(1.0, num_alive * 0.1 + random.uniform(-0.05, 0.05))
            link.congestion_level = max(0.0, min(1.0, congestion))
            
            # 拥塞会降低有效带宽
            effective_bw = link.bandwidth_mb_s * (1 - link.congestion_level * 0.5)
            link.bandwidth_mb_s = max(link.bandwidth_mb_s * 0.5, effective_bw)
