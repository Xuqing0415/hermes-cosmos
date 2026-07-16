#!/usr/bin/env python3
"""
Core Simulation Module


worker/
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class WorkerInfo:
    """Worker"""
    worker_id: int
    speed: float = 1.0  # 
    batch_size: int = 64
    compute_time: float = 0.02
    alive: bool = True
    join_time: float = 0.0
    leave_time: Optional[float] = None


@dataclass
class NetworkLink:
    """"""
    src_id: int
    dst_id: int
    bandwidth_mb_s: float = 100.0
    latency: float = 0.0001
    congestion_level: float = 0.0  # 0-10


@dataclass
class SimulationConfig:
    """"""
    num_workers: int = 4
    model_size_mb: float = 40.0  # MB
    compute_time_per_worker: float = 0.02  # worker
    bandwidth_mb_s: float = 100.0  # MB/s
    latency: float = 0.0001  # 
    gossip_neighbors: int = 2  # Gossip
    compression_ratio: float = 1.0  # 
    overlap_factor: float = 0.0  # 
    batch_size_per_worker: int = 64  # workerbatch size
    enable_noise: bool = False  # 
    
    # worker
    enable_dynamic_workers: bool = False
    worker_join_rate: float = 0.05  # stepworker
    worker_leave_rate: float = 0.02  # step worker
    
    # 
    enable_heterogeneous_network: bool = False
    bandwidth_variation: float = 0.3  # 
    latency_variation: float = 0.5  # 
    
    # 
    enable_sparse_gradient: bool = False
    sparse_top_k_ratio: float = 0.1  # Top-k


class DistributedSimulator:
    """"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.workers: List[WorkerInfo] = []
        self.network_links: Dict[Tuple[int, int], NetworkLink] = {}
        self.step_count = 0
        self.total_throughput = 0.0
        
        # workers
        self._init_workers()
        
        # 
        if self.config.enable_heterogeneous_network:
            self._init_heterogeneous_network()
    
    def _init_workers(self):
        """worker"""
        self.workers = []
        for i in range(self.config.num_workers):
            # worker
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
        """"""
        self.network_links = {}
        
        for i in range(self.config.num_workers):
            for j in range(self.config.num_workers):
                if i != j:
                    # worker 0
                    if i == 0 or j == 0:
                        base_bandwidth = self.config.bandwidth_mb_s * (1.2 + random.uniform(-0.1, 0.1))
                        base_latency = self.config.latency * (0.8 + random.uniform(-0.2, 0.2))
                    else:
                        # 
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
        """"""
        if self.config.enable_noise:
            noise = random.uniform(-noise_level, noise_level)
            return value * (1 + noise)
        return value
    
    def get_network_link(self, src_id: int, dst_id: int) -> NetworkLink:
        """"""
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
        """worker/"""
        if not self.config.enable_dynamic_workers:
            return
        
        # Worker
        for worker in self.workers:
            if worker.alive and random.random() < self.config.worker_leave_rate:
                worker.alive = False
                worker.leave_time = self.step_count
                print(f" Worker {worker.worker_id} left at step {self.step_count}")
        
        # Worker
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
            print(f" New worker {new_id} joined at step {self.step_count}")
    
    def get_alive_workers(self) -> List[WorkerInfo]:
        """worker"""
        return [w for w in self.workers if w.alive]
    
    def compute_sparse_gradient_size(self) -> float:
        """"""
        if not self.config.enable_sparse_gradient:
            return self.config.model_size_mb * self.config.compression_ratio
        
        # Top-k
        effective_ratio = self.config.compression_ratio * self.config.sparse_top_k_ratio
        return self.config.model_size_mb * effective_ratio
    
    def compute_communication_time_ps(self) -> float:
        """
        
        """
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return 0.0
        
        msg_size_send = self.compute_sparse_gradient_size()
        msg_size_recv = self.config.model_size_mb
        
        # 
        max_upload_time = 0.0
        max_download_time = 0.0
        
        for worker in alive_workers:
            link = self.get_network_link(worker.worker_id, 0)  # PS0
            upload_time = msg_size_send / link.bandwidth_mb_s + link.latency
            download_time = msg_size_recv / link.bandwidth_mb_s + link.latency
            
            max_upload_time = max(max_upload_time, upload_time)
            max_download_time = max(max_download_time, download_time)
        
        comm_time = max_upload_time + max_download_time
        return self.add_noise(comm_time)
    
    def compute_communication_time_ddp(self) -> float:
        """
        DDPRing All-Reduce
        """
        alive_workers = self.get_alive_workers()
        num_workers = len(alive_workers)
        
        if num_workers <= 1:
            return 0.0
        
        msg_size = self.compute_sparse_gradient_size()
        total_comm_time = 0.0
        
        # workerworker
        for i in range(num_workers):
            src = alive_workers[i]
            dst = alive_workers[(i + 1) % num_workers]
            
            link = self.get_network_link(src.worker_id, dst.worker_id)
            
            # Ring All-Reducescatter-reduceall-gather
            stage_time = 2 * msg_size / link.bandwidth_mb_s + link.latency
            total_comm_time += stage_time
        
        # 
        comm_time = total_comm_time / num_workers * math.log2(num_workers)
        return self.add_noise(comm_time)
    
    def compute_communication_time_gossip(self) -> float:
        """
        Gossip
        """
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return 0.0
        
        msg_size = self.compute_sparse_gradient_size()
        k = min(self.config.gossip_neighbors, len(alive_workers) - 1)
        
        max_comm_time = 0.0
        
        for worker in alive_workers:
            # k
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
        step
        """
        alive_workers = self.get_alive_workers()
        if not alive_workers:
            return float('inf')
        
        # worker
        compute_time = max(w.compute_time for w in alive_workers)
        overlap = self.config.overlap_factor
        
        step_time = max(compute_time, comm_time * overlap) + (1 - overlap) * (compute_time + comm_time)
        return max(step_time, 0.0001)
    
    def compute_throughput(self, step_time: float) -> float:
        """
        samples/sec
        """
        alive_workers = self.get_alive_workers()
        total_batch_size = sum(w.batch_size for w in alive_workers)
        return total_batch_size / step_time
    
    def evaluate_protocol(self, protocol: str) -> Dict:
        """"""
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
        """"""
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
        """worker/"""
        history = []
        
        for step in range(steps):
            self.step_count = step
            
            # worker
            self.update_worker_dynamics()
            
            # 
            result = self.run_protocol_comparison()
            history.append(result)
            
            # 
            self._update_network_congestion()
        
        return history
    
    def _update_network_congestion(self):
        """"""
        if not self.config.enable_heterogeneous_network:
            return
        
        alive_workers = self.get_alive_workers()
        num_alive = len(alive_workers)
        
        for (src, dst), link in self.network_links.items():
            # 
            congestion = min(1.0, num_alive * 0.1 + random.uniform(-0.05, 0.05))
            link.congestion_level = max(0.0, min(1.0, congestion))
            
            # 
            effective_bw = link.bandwidth_mb_s * (1 - link.congestion_level * 0.5)
            link.bandwidth_mb_s = max(link.bandwidth_mb_s * 0.5, effective_bw)
